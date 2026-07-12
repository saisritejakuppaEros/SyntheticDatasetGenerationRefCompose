"""
Stage 2: Multi-Reference Prompt Composition (stratified, analysis-ready)
------------------------------------------------------------------
Same purpose as the original stage2 script: combine 2-3 GOOD-quality Stage 1
reference captions into a single cinematic "target scene" prompt for a
multi-reference image generation model.

CHANGES vs the original version
================================
1. STRATIFIED / UNIFORM SAMPLING instead of pure rng.choice() per axis.
   With plain random.choice, 100,000 draws converge *close* to uniform but
   never land exactly even, and low-probability corners of the grid can be
   noticeably under/over-represented. This version builds the full
   cartesian grid per axis-group, tiles it to the exact sample count needed,
   and shuffles — so counts per (scale, rotation, lighting, interaction)
   combination are as equal as integer division allows (off by at most 1).

2. n_refs is now 2 or 3 only (1-ref samples removed). Distribution defaults
   to UNIFORM (~50/50) between 2 and 3. Pass
   --ref-count-weights "2:0.5,3:0.5" to override.

3. Category selection is now also cycled (round-robin over a shuffled
   queue, refilled when exhausted) instead of rng.choice, so every category
   AND every individual Stage-1 image gets used an even number of times
   over a large run, rather than some images being drawn far more often
   than others by chance.

4. Every sample record now carries full reference metadata (not just
   category+caption+path) — quality, reason, and any other fields present
   in the Stage 1 record are preserved under `reference_details`.

5. --dry-run mode: skips all LLM calls entirely. It still does 100% of the
   sampling (refs + conditions) and writes:
     - manifest.jsonl        (same schema, combined_caption = null)
     - metadata_preview.csv  (flat, one row per sample — open in Excel/pandas)
     - distribution_summary.txt (crosstab counts per axis, to verify uniformity)
   This lets you validate the whole dataset's composition/metadata BEFORE
   you spend GPU time generating 100k captions.

6. COMPOSITION RULES (new):
   - Every sample is GUARANTEED to include exactly one "landscape" category
     reference. It is always slot #1 in the reference list and is treated
     as the scene's environment/setting rather than a "subject" — the
     prompt template explicitly instructs the LLM to use it as the
     backdrop the other subjects are placed into.
   - At most ONE "person" category reference is allowed per sample (never
     two or three people stacked into one composition).
   - Every sample therefore has 2 or 3 total references: 1 guaranteed
     landscape + 1-2 additional subjects (at most one of which can be a
     person).

Run:
    # Preview only — no LLM calls, no GPU server needed
    python stage2_prompt_composition.py --num-samples 100000 --dry-run

    # Full run (needs Stage 1 manifest + a running vLLM/OpenAI-compatible server)
    python stage2_prompt_composition.py --num-samples 100000 --threads 24
"""

import os
import re
import csv
import json
import time
import random
import argparse
import itertools
import threading
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

BASE_URLS = [
    "http://localhost:8000/v1",
    # "http://localhost:8001/v1",
    # "http://localhost:8002/v1",
    # "http://localhost:8003/v1",
    # "http://localhost:8004/v1",
    # "http://localhost:8005/v1",
]

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct-FP8"

STAGE1_MANIFEST = "outputs/manifest.jsonl"
OUTPUT_ROOT = "outputs/stage2_composed"
NUM_THREADS = 16

# Category names in the Stage 1 manifest that get special composition rules.
# Stage 1 stores environment/setting refs under "landmark" (not "landscape").
LANDSCAPE_CATEGORY = "landmark"
PERSON_CATEGORY = "person"
MAX_PERSON_REFS = 1  # never more than this many "person" refs in one sample

# n_refs is now only ever 2 or 3 (1-ref samples removed, since we always
# need at least: 1 landscape + 1 other subject). Default: uniform.
DEFAULT_REF_COUNT_WEIGHTS = {2: 0.5, 3: 0.5}

# Probability of allowing 2 NON-landscape refs in the same sample to come
# from the SAME category. Still stochastic (not stratified) since it's a
# per-sample coin flip layered on top of an otherwise-cycled category queue.
SAME_CATEGORY_PROB = 0.15

_manifest_lock = threading.Lock()

# ----------------------------------------------------------------------------
# Condition pools — (key, instruction_text)
# ----------------------------------------------------------------------------

SCALE_VARIATIONS = [
    ("natural_scale", "all subjects appear at their natural, real-world relative scale to one another"),
    ("hero_scale_up", "the first-listed subject is scaled up dramatically, dominating the frame compared to the others"),
    ("miniature", "the whole scene appears miniaturized, like a diorama or tilt-shift photograph"),
    ("extreme_closeup", "extreme close-up framing that fills most of the frame with fine surface detail"),
    ("wide_shot", "a wide establishing shot where the subjects appear small within a much larger environment"),
]

ROTATION_VARIATIONS = [
    ("front_facing", "subjects are shown front-facing, directly toward the camera"),
    ("three_quarter", "subjects are shown at a three-quarter angle"),
    ("profile", "subjects are shown in profile view, seen from the side"),
    ("low_angle", "the shot is taken from a low angle looking upward, emphasizing height and scale"),
    ("high_angle", "the shot is taken from a high angle looking downward, a bird's-eye perspective"),
    ("dutch_angle", "the camera is tilted at a dutch angle for a dynamic, off-kilter composition"),
    ("from_behind", "subjects are shown from behind or a rear three-quarter view"),
]

LIGHTING_VARIATIONS = [
    ("golden_hour", "warm golden-hour sunlight casting long, soft shadows"),
    ("harsh_noon", "harsh overhead midday sunlight with hard-edged shadows"),
    ("studio_softbox", "even, diffused studio softbox lighting with minimal shadow"),
    ("neon_night", "moody neon-lit night-time lighting with saturated colored highlights"),
    ("overcast", "soft, overcast diffuse daylight with flat, even illumination"),
    ("candlelight", "warm, flickering candlelight with deep ambient shadow"),
    ("moonlight", "cool blue moonlight with low ambient light"),
    ("rim_light", "dramatic rim lighting separating the subjects from a darker background"),
]

# Since every sample now always has >=2 refs, INTERACTION_VARIATIONS is
# always sampled (no more n_refs==1 special case).
INTERACTION_VARIATIONS = [
    ("holding", "one subject is actively holding, touching, or carrying another element in the scene"),
    ("side_by_side", "the subjects are positioned side by side, coexisting in the same space without direct contact"),
    ("using_object", "a subject is actively using or functionally interacting with an object present in the scene"),
    ("approaching", "one subject is walking toward or approaching another across the scene"),
    ("presenting", "a subject is gesturing toward or presenting another element in the frame"),
    ("consuming", "a subject is in the act of eating or tasting food present in the scene"),
    ("backdrop", "one element (such as a landmark) serves purely as a backdrop behind the other subject(s), with no direct interaction"),
]

COMPOSITION_PROMPT_TEMPLATE = """You are writing a single combined image-generation prompt that will guide a diffusion model to compose ONE new scene using {n} reference subjects together.

Reference #1 below is the LANDSCAPE / ENVIRONMENT reference — it sets the location and setting of the entire scene. It is NOT a subject to place into the scene; everything else is placed INTO it.
The remaining reference(s) are the SUBJECTS to be placed within that landscape.

References (factual descriptions — each one corresponds to a separate input reference image):
{reference_block}

Write ONE cohesive, cinematic scene description (60-100 words) that:
- Uses the landscape reference (#1) as the setting/environment of the whole scene — describe the place itself, grounded in its real features
- Places ALL other reference subjects into that landscape together, in a single unified scene
- Follows these exact composition constraints:
  - Scale: {scale_instruction}
  - Camera angle: {rotation_instruction}
  - Lighting: {lighting_instruction}
  - Interaction: {interaction_instruction}
- Keeps each subject's identity and appearance consistent with its reference description above — only change HOW it is composed into the new scene, never WHAT it looks like
- Is written as a single flowing paragraph, vivid and cinematic, suitable to be used directly as an image-generation prompt
- Does NOT use meta-language like "reference image", "combine", "subject 1", or similar — describe the final target scene directly, as if narrating what the camera sees

Respond with ONLY the combined scene description text. No preamble, no labels, no quotation marks.
"""


# ----------------------------------------------------------------------------
# Stage 1 manifest loading
# ----------------------------------------------------------------------------

def load_good_entries(manifest_path):
    """Load Stage 1 manifest, keep only GOOD-quality, non-errored entries, grouped by category."""
    by_category = defaultdict(list)
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            if rec.get("quality") != "GOOD":
                continue
            if not rec.get("caption"):
                continue
            by_category[rec["category"]].append(rec)

    if LANDSCAPE_CATEGORY not in by_category or not by_category[LANDSCAPE_CATEGORY]:
        raise RuntimeError(
            f"No GOOD '{LANDSCAPE_CATEGORY}' entries found in {manifest_path}. "
            f"Every sample now requires exactly one landmark/landscape reference, so Stage 1 "
            f"must contain at least one GOOD '{LANDSCAPE_CATEGORY}' record."
        )

    return by_category


# ----------------------------------------------------------------------------
# Cyclic (round-robin-over-shuffled-queue) sampler
# Guarantees every item is drawn an even number of times over a long run,
# instead of rng.choice's "correct on average, uneven in practice" behavior.
# ----------------------------------------------------------------------------

class CyclicSampler:
    def __init__(self, items, rng):
        self.items = list(items)
        self.rng = rng
        self.queue = deque()
        self._refill()

    def _refill(self):
        pool = list(self.items)
        self.rng.shuffle(pool)
        self.queue.extend(pool)

    def next(self):
        if not self.queue:
            self._refill()
        return self.queue.popleft()

    def next_excluding(self, exclude):
        """Pop the next item not in `exclude`. Falls back to any item if every
        candidate in one full cycle is excluded (can't avoid a repeat)."""
        tried = []
        for _ in range(len(self.items) + 1):
            item = self.next()
            if item not in exclude:
                self.queue.extendleft(reversed(tried))
                return item
            tried.append(item)
        # everything excluded — just return something
        self.queue.extendleft(reversed(tried))
        return self.next()


# ----------------------------------------------------------------------------
# Stratified condition grid builder
# Builds the exact cartesian grid for a given n_refs, tiles it to the target
# count (as evenly as integer division allows), and shuffles.
# Since n_refs is always >= 2 now, INTERACTION_VARIATIONS is always included.
# ----------------------------------------------------------------------------

def build_condition_grid(n_refs, count, rng):
    axis_lists = [SCALE_VARIATIONS, ROTATION_VARIATIONS, LIGHTING_VARIATIONS, INTERACTION_VARIATIONS]
    axis_names = ["scale", "rotation", "lighting", "interaction"]

    combos = list(itertools.product(*axis_lists))  # each combo: tuple of (key, text) pairs
    n_combos = len(combos)

    full_repeats = count // n_combos
    remainder = count % n_combos

    tiled = combos * full_repeats
    if remainder:
        tiled += rng.sample(combos, remainder)  # distinct combos get the "+1", not always the same ones

    rng.shuffle(tiled)

    grid = []
    for combo in tiled:
        conditions = {}
        instructions = {}
        for name, (key, text) in zip(axis_names, combo):
            conditions[name] = key
            instructions[f"{name}_instruction"] = text
        grid.append((conditions, instructions))
    return grid


# ----------------------------------------------------------------------------
# Ref count allocation (exact, not just weighted-random)
# ----------------------------------------------------------------------------

def allocate_ref_counts(total, weights):
    counts = list(weights.keys())
    raw = {k: total * w for k, w in weights.items()}
    floored = {k: int(v) for k, v in raw.items()}
    allocated = sum(floored.values())
    remainder = total - allocated
    # distribute leftover samples to whichever counts had the largest fractional part
    fracs = sorted(counts, key=lambda k: raw[k] - floored[k], reverse=True)
    for i in range(remainder):
        floored[fracs[i % len(fracs)]] += 1
    return floored  # e.g. {2: 50000, 3: 50000}


def parse_ref_count_weights(arg_str):
    if not arg_str:
        return DEFAULT_REF_COUNT_WEIGHTS
    weights = {}
    for part in arg_str.split(","):
        k, v = part.split(":")
        k = int(k)
        if k < 2:
            raise ValueError(
                f"Invalid ref count {k}: every sample now requires >=2 references "
                f"(1 guaranteed landscape + >=1 other subject)."
            )
        weights[k] = float(v)
    total_w = sum(weights.values())
    return {k: v / total_w for k, v in weights.items()}


# ----------------------------------------------------------------------------
# Reference sampling (category- and image-balanced)
#
# Composition rules enforced here:
#   1. Slot #1 is ALWAYS a "landscape" reference (guaranteed, every sample).
#   2. At most MAX_PERSON_REFS "person" references are allowed in the
#      remaining slots.
#   3. The non-landscape slots are still cycled/balanced via the shared
#      category_sampler + record_samplers, same as before.
# ----------------------------------------------------------------------------

def sample_refs(n_refs, category_sampler, record_samplers, same_category_prob, rng,
                 landscape_category=LANDSCAPE_CATEGORY, person_category=PERSON_CATEGORY,
                 max_person=MAX_PERSON_REFS):
    # 1. Guaranteed landscape reference, always first.
    landscape_rec = record_samplers[landscape_category].next()
    chosen = [landscape_rec]
    used = {landscape_category}
    person_count = 0

    # 2. Remaining slots: pick subjects, respecting the person cap.
    remaining = n_refs - 1
    for _ in range(remaining):
        exclude = {landscape_category}  # never draw a 2nd landscape as a "subject" slot
        if person_count >= max_person:
            exclude.add(person_category)

        allow_repeat = rng.random() < same_category_prob or len(used - {landscape_category}) >= (
            len(record_samplers) - 1
        )
        if not allow_repeat:
            exclude |= used

        cat = category_sampler.next_excluding(exclude)
        rec = record_samplers[cat].next()
        chosen.append(rec)
        used.add(cat)
        if cat == person_category:
            person_count += 1

    return chosen


def build_prompt(refs, instructions):
    n = len(refs)
    reference_lines = []
    for i, rec in enumerate(refs, start=1):
        tag = " (LANDSCAPE/ENVIRONMENT)" if i == 1 else ""
        reference_lines.append(f"{i}. [{rec['category']}]{tag} {rec['caption']}")
    reference_block = "\n".join(reference_lines)

    return COMPOSITION_PROMPT_TEMPLATE.format(
        n=n,
        reference_block=reference_block,
        **instructions,
    )


# ----------------------------------------------------------------------------
# Task preparation (fully deterministic given a seed; no LLM calls here)
# ----------------------------------------------------------------------------

def prepare_tasks(num_samples, seed, ref_count_weights, same_category_prob):
    rng = random.Random(seed)

    by_category = load_good_entries(STAGE1_MANIFEST)

    print("GOOD references available per category:")
    for cat, items in by_category.items():
        print(f"  {cat}: {len(items)}")

    if PERSON_CATEGORY not in by_category:
        print(f"  (note: no '{PERSON_CATEGORY}' category found — samples will simply have 0 people)")

    category_sampler = CyclicSampler(list(by_category.keys()), rng)
    record_samplers = {cat: CyclicSampler(items, rng) for cat, items in by_category.items()}

    ref_counts = allocate_ref_counts(num_samples, ref_count_weights)
    print(f"\nRef-count allocation (2=landscape+1 subject, 3=landscape+2 subjects): {ref_counts}")

    tasks = []
    sample_idx = 0
    for n_refs, group_count in ref_counts.items():
        if group_count == 0:
            continue
        grid = build_condition_grid(n_refs, group_count, rng)
        for conditions, instructions in grid:
            refs = sample_refs(n_refs, category_sampler, record_samplers, same_category_prob, rng)
            if len(refs) < n_refs:
                continue
            prompt_text = build_prompt(refs, instructions)
            sample_id = f"sample_{sample_idx:07d}"
            tasks.append((sample_id, refs, conditions, instructions, prompt_text))
            sample_idx += 1

    rng.shuffle(tasks)  # interleave n_refs groups so they're not written in big contiguous blocks
    return tasks


# ----------------------------------------------------------------------------
# Core worker (LLM call)
# ----------------------------------------------------------------------------

def process_one(client_getter, sample_id, refs, conditions, instructions, prompt_text,
                 samples_dir, dry_run, max_retries=3):
    saved_path = os.path.join(samples_dir, f"{sample_id}.json")

    result = {
        "id": sample_id,
        "num_references": len(refs),
        "reference_images": [r["saved_image_path"] for r in refs],
        "reference_categories": [r["category"] for r in refs],
        "reference_captions": [r["caption"] for r in refs],
        "reference_details": refs,  # full Stage 1 records: quality, reason, any extra fields
        "conditions": conditions,
        "conditions_text": {k: v for k, v in instructions.items()},
        "combined_caption": None,
        "error": None,
        "dry_run": dry_run,
    }

    if not dry_run:
        last_err = None
        for attempt in range(max_retries):
            try:
                response = client_getter().chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=400,
                    temperature=0.9,
                    top_p=0.95,
                )
                raw_text = response.choices[0].message.content.strip()
                raw_text = re.sub(r'^["\']|["\']$', "", raw_text).strip()
                result["combined_caption"] = raw_text
                last_err = None
                break
            except Exception as e:
                last_err = str(e)
                time.sleep(2 ** attempt)
        if last_err:
            result["error"] = last_err

    with open(saved_path, "w") as f:
        json.dump(result, f, indent=2)

    result["saved_path"] = saved_path
    return result


# ----------------------------------------------------------------------------
# Metadata CSV + distribution summary (the "analysis before I get started" part)
# ----------------------------------------------------------------------------

def write_metadata_csv(tasks_or_results, csv_path, is_result=True):
    fieldnames = [
        "id", "num_references", "ref_categories", "ref_captions_preview",
        "scale", "rotation", "lighting", "interaction",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in tasks_or_results:
            if is_result:
                sid = item["id"]
                cats = item["reference_categories"]
                captions = item["reference_captions"]
                cond = item["conditions"]
            else:
                sid, refs, cond, instructions, prompt_text = item
                cats = [r["category"] for r in refs]
                captions = [r["caption"] for r in refs]
            writer.writerow({
                "id": sid,
                "num_references": len(cats),
                "ref_categories": "|".join(cats),
                "ref_captions_preview": " || ".join(c[:60] for c in captions),
                "scale": cond["scale"],
                "rotation": cond["rotation"],
                "lighting": cond["lighting"],
                "interaction": cond["interaction"] or "",
            })


def write_distribution_summary(tasks_or_results, summary_path, is_result=True):
    n_refs_counter = Counter()
    scale_counter = Counter()
    rotation_counter = Counter()
    lighting_counter = Counter()
    interaction_counter = Counter()
    category_counter = Counter()
    category_pair_counter = Counter()
    person_count_counter = Counter()

    for item in tasks_or_results:
        if is_result:
            cats = item["reference_categories"]
            cond = item["conditions"]
        else:
            sid, refs, cond, instructions, prompt_text = item
            cats = [r["category"] for r in refs]

        n_refs_counter[len(cats)] += 1
        scale_counter[cond["scale"]] += 1
        rotation_counter[cond["rotation"]] += 1
        lighting_counter[cond["lighting"]] += 1
        interaction_counter[cond["interaction"] or "N/A"] += 1
        for c in cats:
            category_counter[c] += 1
        if len(set(cats)) > 1:
            category_pair_counter[tuple(sorted(set(cats)))] += 1
        person_count_counter[cats.count(PERSON_CATEGORY)] += 1

    lines = []
    total = sum(n_refs_counter.values())
    lines.append(f"TOTAL SAMPLES: {total}\n")

    def section(title, counter):
        lines.append(f"--- {title} ---")
        for k, v in sorted(counter.items(), key=lambda x: str(x[0])):
            pct = 100 * v / total if total else 0
            lines.append(f"  {k}: {v}  ({pct:.2f}%)")
        lines.append("")

    section("num_references", n_refs_counter)
    section("scale", scale_counter)
    section("rotation", rotation_counter)
    section("lighting", lighting_counter)
    section("interaction", interaction_counter)
    section("reference category usage (raw ref slots, not samples)", category_counter)
    section(f"'{PERSON_CATEGORY}' refs per sample (sanity check, should max out at {MAX_PERSON_REFS})",
            person_count_counter)

    lines.append("--- top 20 category pairings (multi-ref samples) ---")
    for pair, cnt in category_pair_counter.most_common(20):
        lines.append(f"  {pair}: {cnt}")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(lines[:40]))
    print(f"\n(Full summary written to {summary_path})")


# ----------------------------------------------------------------------------
# Pipeline runner
# ----------------------------------------------------------------------------

def run_pipeline(num_samples, num_threads, seed, ref_count_weights, same_category_prob, dry_run):
    samples_dir = os.path.join(OUTPUT_ROOT, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    manifest_path = os.path.join(OUTPUT_ROOT, "manifest.jsonl")
    csv_path = os.path.join(OUTPUT_ROOT, "metadata_preview.csv")
    summary_path = os.path.join(OUTPUT_ROOT, "distribution_summary.txt")

    tasks = prepare_tasks(num_samples, seed, ref_count_weights, same_category_prob)
    print(f"\nPrepared {len(tasks)} composition samples")

    if dry_run:
        print("\n[DRY RUN] Skipping LLM calls. Writing metadata for review only.\n")
        write_metadata_csv(tasks, csv_path, is_result=False)
        write_distribution_summary(tasks, summary_path, is_result=False)

        with open(manifest_path, "w") as manifest_f:
            for sid, refs, cond, instructions, prompt_text in tasks:
                record = {
                    "id": sid,
                    "num_references": len(refs),
                    "reference_images": [r["saved_image_path"] for r in refs],
                    "reference_categories": [r["category"] for r in refs],
                    "reference_captions": [r["caption"] for r in refs],
                    "reference_details": refs,
                    "conditions": cond,
                    "conditions_text": instructions,
                    "combined_caption": None,
                    "error": None,
                    "dry_run": True,
                }
                manifest_f.write(json.dumps(record) + "\n")

        print("\n" + "=" * 60)
        print("DRY RUN COMPLETE — no LLM/GPU calls were made.")
        print(f"Manifest:          {manifest_path}")
        print(f"Metadata CSV:      {csv_path}")
        print(f"Distribution info: {summary_path}")
        print("Review these, then re-run without --dry-run to generate captions.")
        print("=" * 60)
        return

    # ---- Real run: needs the OpenAI-compatible client(s) ----
    from openai import OpenAI
    clients = [OpenAI(api_key="EMPTY", base_url=url, timeout=3600) for url in BASE_URLS]
    rr_lock = threading.Lock()
    rr_idx = [0]

    def get_client():
        with rr_lock:
            idx = rr_idx[0] % len(clients)
            rr_idx[0] += 1
        return clients[idx]

    start = time.time()
    ok_count = 0
    err_count = 0
    results = []

    with open(manifest_path, "w") as manifest_f:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(process_one, get_client, sid, refs, cond, instructions,
                                 prompt, samples_dir, dry_run): sid
                for sid, refs, cond, instructions, prompt in tasks
            }
            with tqdm(total=len(tasks), desc="Composing", unit="sample") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

                    with _manifest_lock:
                        manifest_f.write(json.dumps(result) + "\n")
                        manifest_f.flush()

                    if result.get("error"):
                        err_count += 1
                    else:
                        ok_count += 1

                    pbar.update(1)
                    pbar.set_postfix(OK=ok_count, ERR=err_count, refresh=False)

    write_metadata_csv(results, csv_path, is_result=True)
    write_distribution_summary(results, summary_path, is_result=True)

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"OK={ok_count}  ERROR={err_count}")
    print(f"Manifest: {manifest_path}")
    print(f"Metadata CSV: {csv_path}")
    print(f"Distribution summary: {summary_path}")
    print(f"Per-sample records: {samples_dir}/<sample_id>.json")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=100000,
                         help="How many composed multi-reference prompts to generate (1 Lakh = 100000)")
    parser.add_argument("--threads", type=int, default=NUM_THREADS,
                         help="Number of worker threads")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducible sampling (default fixed for reproducibility)")
    parser.add_argument("--ref-count-weights", type=str, default=None,
                         help='e.g. "2:0.5,3:0.5" (default) to control the 2-ref vs 3-ref split. '
                              'Every sample always has exactly 1 landscape ref, so 2 means '
                              '"landscape + 1 subject" and 3 means "landscape + 2 subjects". '
                              'Values below 2 are rejected.')
    parser.add_argument("--same-category-prob", type=float, default=SAME_CATEGORY_PROB,
                         help="Probability two non-landscape subject refs in one sample share a category")
    parser.add_argument("--dry-run", action="store_true",
                         help="Sample everything and write metadata/CSV/summary, but skip all LLM calls")
    args = parser.parse_args()

    weights = parse_ref_count_weights(args.ref_count_weights)

    run_pipeline(
        num_samples=args.num_samples,
        num_threads=args.threads,
        seed=args.seed,
        ref_count_weights=weights,
        same_category_prob=args.same_category_prob,
        dry_run=args.dry_run,
    )
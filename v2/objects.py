"""
Theme Prompt Composer + Photoreal Renderer — COCO-object, shot-aware,
grounded-composition scenes
---------------------------------------------------------------------------------
Same pipeline as before (shot type -> depth-restricted locations -> grounded
surface -> orientation/color/material -> optional human presence -> LLM
rewrite into one cinematic paragraph), but:

  - Objects now come from the 80 COCO classes instead of 10 hand-picked
    "genre" lists, grouped into 10 physically-coherent SCENES (kitchen,
    dining room, living room, bedroom, home office, city street, sports
    field, beach, savanna wildlife, transportation hub) so the six objects
    in a sample actually belong in the same room/place together.
  - Adds a second stage, `--generate-images`, that reads the manifest
    produced by stage one and turns each `combined_prompt` (the cinematic
    paragraph — never the raw structured bullet list) into a photorealistic
    image with a local diffusers pipeline, so nothing about the composed
    scene gets lost between "prompt" and "pixels".

Two independent stages, run separately (image gen is GPU/VRAM heavy and you
generally want to eyeball the text manifest before burning render time):

  Stage 1 (prompts):
    python objects.py --total 10000 --max-permutations --dry-run
    python objects.py --total 10000 --max-permutations --threads 16
    python objects.py --num-per-theme 200 --dry-run
    python objects.py --total 5000 --dry-run --people-fraction 0.5
    python objects.py --total 5000 --dry-run --shot-types close_up,wide_shot

  Stage 2 (photoreal images from the manifest stage 1 produced):
    python objects.py --generate-images
    python objects.py --generate-images --image-model stabilityai/stable-diffusion-xl-base-1.0
    python objects.py --generate-images --limit 200 --steps 40 --guidance 6.5
"""

import os
import csv
import json
import time
import random
import argparse
import itertools
import threading
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

BASE_URLS = ["http://localhost:8000/v1"]
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct-FP8"
OUTPUT_ROOT = "outputs/theme_prompts"
NUM_THREADS = 16

# ----------------------------------------------------------------------------
# COCO-derived scenes: each is a physically-coherent place, populated with
# exactly 6 COCO object classes that would plausibly all be found there
# together. This is what keeps compositions from reading as "randomly
# displaced" — the objects aren't just diverse, they're co-located by design.
# ----------------------------------------------------------------------------

THEMES = {
    "cozy_kitchen": ["refrigerator", "oven", "sink", "toaster", "bowl", "knife"],
    "dining_room": ["dining table", "wine glass", "bottle", "cup", "fork", "spoon"],
    "living_room": ["couch", "tv", "remote", "vase", "potted plant", "clock"],
    "bedroom": ["bed", "book", "backpack", "cell phone", "teddy bear", "clock"],
    "home_office": ["laptop", "keyboard", "mouse", "chair", "book", "cell phone"],
    "city_street": ["car", "traffic light", "stop sign", "bicycle", "bench", "fire hydrant"],
    "sports_field": ["sports ball", "baseball bat", "baseball glove", "skateboard",
                      "tennis racket", "frisbee"],
    "beach_day": ["surfboard", "kite", "umbrella", "backpack", "bottle", "handbag"],
    "savanna_wildlife": ["elephant", "zebra", "giraffe", "bear", "horse", "cow"],
    "transportation_hub": ["airplane", "bus", "train", "truck", "boat", "motorcycle"],
}

N_OBJECTS = 6  # fixed, every theme has exactly 6

# ----------------------------------------------------------------------------
# Location pool — 12 frame positions, grouped by depth so each SHOT TYPE can
# restrict itself to the depths that are actually plausible for that framing.
# (unchanged from the original — this part already worked well)
# ----------------------------------------------------------------------------

LOCATIONS_LIST = [
    ("foreground_left", "in the immediate foreground, lower-left of frame"),
    ("foreground_right", "in the immediate foreground, lower-right of frame"),
    ("foreground_center", "front and center, closest to camera"),
    ("midground_left", "in the midground, left of center"),
    ("midground_right", "in the midground, right of center"),
    ("midground_center", "in the midground, directly centered"),
    ("background_left", "receding into the background, upper-left"),
    ("background_right", "receding into the background, upper-right"),
    ("background_center", "in the far background, centered and slightly out of focus"),
    ("top_shelf", "resting on a high shelf or ledge near the top of frame"),
    ("floating_midair", "suspended in midair, as if floating"),
    ("bottom_edge", "along the very bottom edge of frame, partially cropped"),
]
LOCATIONS_BY_KEY = {key: (key, text) for key, text in LOCATIONS_LIST}

# ----------------------------------------------------------------------------
# Shot types — each shot restricts which depths are usable and tells the LLM
# how to handle focus/blur, so the six objects land in a composition that
# actually matches how that framing would really look.
# ----------------------------------------------------------------------------

SHOT_TYPES = {
    "close_up": {
        "label": "an extreme close-up shot",
        "camera_text": (
            "an extreme close-up shot with a shallow depth of field — the camera "
            "sits just inches from the nearest object, which is razor-sharp, while "
            "anything past the mid-ground dissolves into a soft, creamy blur"
        ),
        "location_keys": [
            "foreground_left", "foreground_right", "foreground_center",
            "midground_left", "midground_right", "midground_center", "bottom_edge",
        ],
    },
    "mid_shot": {
        "label": "a medium shot",
        "camera_text": (
            "a medium shot, framed like a well-composed tabletop or workspace "
            "still-life — the camera sits a few feet back so every object stays "
            "clearly legible, with only gentle, natural falloff toward the edges"
        ),
        "location_keys": [
            "foreground_left", "foreground_right", "foreground_center",
            "midground_left", "midground_right", "midground_center",
            "top_shelf", "bottom_edge",
        ],
    },
    "wide_shot": {
        "label": "a wide establishing shot",
        "camera_text": (
            "a wide establishing shot that takes in the entire room or environment — "
            "deep focus keeps both near and far objects readable across the full "
            "depth of the space"
        ),
        "location_keys": [
            "foreground_left", "foreground_right", "foreground_center",
            "midground_left", "midground_right", "midground_center",
            "background_left", "background_right", "background_center",
            "top_shelf", "floating_midair", "bottom_edge",
        ],
    },
}
SHOT_TYPE_KEYS = list(SHOT_TYPES.keys())

for _shot, _cfg in SHOT_TYPES.items():
    _cfg["locations"] = [LOCATIONS_BY_KEY[k] for k in _cfg["location_keys"]]
    assert len(_cfg["locations"]) >= N_OBJECTS, f"{_shot} location pool too small"

# ----------------------------------------------------------------------------
# Orientation pool — how each object is turned/posed.
# ----------------------------------------------------------------------------

ORIENTATIONS = [
    ("facing_camera", "facing directly toward the camera"),
    ("profile_left", "turned in left profile"),
    ("profile_right", "turned in right profile"),
    ("three_quarter_left", "angled three-quarters to the left"),
    ("three_quarter_right", "angled three-quarters to the right"),
    ("facing_away", "turned away from the camera, back to the viewer"),
    ("tilted_45", "tilted at a 45-degree angle, as if mid-fall"),
    ("lying_flat", "lying flat on its side"),
    ("upright_vertical", "standing perfectly upright and vertical"),
    ("upside_down", "inverted, upside down"),
    ("leaning", "leaning against a nearby surface"),
    ("mid_motion", "caught mid-motion, slightly blurred with implied movement"),
]

# ----------------------------------------------------------------------------
# Color + material/finish pools — pure diversity axes, no constraints.
# ----------------------------------------------------------------------------

COLORS = [
    "deep crimson", "burnt orange", "emerald green", "midnight blue", "ivory white",
    "charcoal black", "tarnished gold", "dusty rose", "steel gray", "amber",
    "violet", "faded teal", "copper", "bone white", "obsidian black",
]

MATERIALS = [
    "weathered and worn", "polished to a high shine", "rusted and corroded",
    "finely detailed", "matte and understated", "gently glowing",
    "translucent and glassy", "scuffed and well-used", "leather-bound and creased",
    "dust-covered", "freshly cleaned", "chipped and battered",
]

# ----------------------------------------------------------------------------
# Per-scene flavor: a short mood/setting phrase, so different scenes read as
# distinct places rather than re-skins of the same room.
# ----------------------------------------------------------------------------

THEME_SETTINGS = {
    "cozy_kitchen": "inside a warm, sunlit farmhouse kitchen with tiled countertops and hanging pots",
    "dining_room": "inside an elegant dining room lit by a low-hanging chandelier",
    "living_room": "inside a cozy living room lit by warm lamplight and the flicker of a television",
    "bedroom": "inside a quiet, softly lit bedroom with rumpled linens and morning light through curtains",
    "home_office": "inside a cluttered home office lit by a desk lamp and a glowing laptop screen",
    "city_street": "along a rain-slicked city street lit by streetlights and passing headlights",
    "sports_field": "on a sunlit outdoor sports field with worn grass and faded chalk lines",
    "beach_day": "on a sun-drenched beach with soft sand and rolling surf in the distance",
    "savanna_wildlife": "across an open golden savanna under a wide, dusty sky",
    "transportation_hub": "inside a bustling transportation terminal with distant announcements and shifting crowds",
}

# ----------------------------------------------------------------------------
# Grounded, scene-appropriate surfaces. Every object is anchored to something
# physically real in that place, not floating at an abstract coordinate.
# ----------------------------------------------------------------------------

THEME_SURFACES = {
    "cozy_kitchen": [
        "set on the tiled countertop", "tucked inside the open cabinet",
        "resting on the stovetop", "hanging from a wall hook",
        "set on the wooden cutting board", "tucked into a corner nook by the window",
    ],
    "dining_room": [
        "set at a place setting on the table", "resting beside a folded napkin",
        "set on the side buffet", "placed near the centerpiece",
        "on a woven placemat", "set beside a lit candle at the table's edge",
    ],
    "living_room": [
        "set on the coffee table", "resting on the couch cushions",
        "on a floating wall shelf", "set on the windowsill",
        "beside the TV stand", "set on the area rug",
    ],
    "bedroom": [
        "set on the nightstand", "resting atop the rumpled bedsheets",
        "on the dresser", "propped against a pillow",
        "on the windowsill", "half-tucked under the edge of the bed",
    ],
    "home_office": [
        "set on the cluttered desk", "resting on a stack of papers",
        "beside the laptop", "on a low shelf within arm's reach",
        "set on the desk's armrest", "in a drawer left half-open",
    ],
    "city_street": [
        "parked at the curb", "propped against a lamppost",
        "set on the sidewalk", "near the crosswalk",
        "beside a storm drain", "resting against a brick building wall",
    ],
    "sports_field": [
        "set on the sideline bench", "resting on the worn grass",
        "near the goalpost", "on the chalk line",
        "propped against the chain-link fence", "tucked into an open equipment bag",
    ],
    "beach_day": [
        "half-buried in the sand", "propped against a beach towel",
        "near the water's edge", "resting on a striped blanket",
        "tucked beside a cooler", "planted upright in the sand",
    ],
    "savanna_wildlife": [
        "standing in the tall dry grass", "near a shallow watering hole",
        "beneath the shade of a lone acacia tree", "on a dusty worn trail",
        "at the edge of the herd", "silhouetted against the horizon",
    ],
    "transportation_hub": [
        "parked at the gate", "sitting on the platform",
        "near the boarding ramp", "beside the luggage carousel",
        "idling in its bay", "waiting at the terminal curb",
    ],
}

# ----------------------------------------------------------------------------
# Some COCO objects are furniture-scale, fixed-in-place, or ARE the thing the
# scene surface would normally describe (the couch can't "rest on the couch").
# These get a dedicated, physically sane override instead.
# ----------------------------------------------------------------------------

OBJECT_SURFACE_OVERRIDES = {
    # cozy_kitchen
    "refrigerator": ["standing against the kitchen wall, door closed",
                      "humming quietly in the corner of the kitchen"],
    "oven": ["built into the counter, door closed",
             "standing against the far wall, dials glinting"],
    "sink": ["built into the countertop beneath the window",
             "set into the counter, faucet dripping"],
    # dining_room
    "dining table": ["standing at the center of the room, set for a meal",
                      "positioned as the room's centerpiece, linens laid out"],
    # living_room
    "couch": ["positioned against the living room wall",
              "set as the room's centerpiece seating"],
    "tv": ["mounted on the wall facing the couch",
           "set atop a low media console"],
    # bedroom
    "bed": ["positioned against the bedroom wall, linens rumpled",
            "set beneath the window, sheets in disarray"],
    # home_office
    "chair": ["pulled up to the desk", "positioned facing the monitor"],
    # city_street
    "traffic light": ["mounted on its post above the intersection",
                       "standing fixed at the corner, lens glowing"],
    "stop sign": ["planted on its post at the corner",
                  "fixed at the edge of the crosswalk"],
    "fire hydrant": ["fixed to the sidewalk at the corner",
                     "standing at the curb's edge"],
    "bench": ["bolted along the sidewalk", "fixed near the crosswalk, paint peeling"],
}

# ----------------------------------------------------------------------------
# Optional human presence — gives the frame scale, a focal point, and a
# reason to read as a *story* rather than a laid-out inventory of props.
# ----------------------------------------------------------------------------

THEME_CHARACTERS = {
    "cozy_kitchen": [
        "a home cook wiping their hands on an apron, glancing toward the stove",
        "a child sneaking a taste from a bowl on the counter",
        "an elderly figure quietly tidying the kitchen before dawn",
    ],
    "dining_room": [
        "a host straightening the table setting before guests arrive",
        "a lone figure seated at the far end of the table, lost in thought",
        "someone refilling a glass in the low chandelier light",
    ],
    "living_room": [
        "a person curled up on the couch, half-lit by the television",
        "a figure standing at the window, watching the evening settle in",
        "someone reaching for the remote without looking up",
    ],
    "bedroom": [
        "a figure just waking, sheets tangled around them",
        "someone sitting on the edge of the bed, phone in hand",
        "a person packing quietly before sunrise",
    ],
    "home_office": [
        "a remote worker hunched over the laptop, coffee gone cold",
        "a figure rubbing tired eyes at the cluttered desk",
        "someone on a late call, papers scattered around them",
    ],
    "city_street": [
        "a pedestrian crossing briskly beneath the traffic light",
        "a cyclist pausing at the corner, checking their phone",
        "a lone figure walking past the bench in the rain",
    ],
    "sports_field": [
        "a player crouched at the sideline, catching their breath",
        "a coach gesturing instructions from the bench",
        "a kid chasing after a rolling ball mid-play",
    ],
    "beach_day": [
        "a surfer waxing their board at the water's edge",
        "a sunbather shading their eyes beneath the umbrella",
        "a child piling up a small mound of sand nearby",
    ],
    "savanna_wildlife": [
        "a lone ranger observing quietly through binoculars from a distance",
        "a guide pointing out the herd to unseen tourists",
        "a photographer crouched low, camera raised toward the horizon",
    ],
    "transportation_hub": [
        "a traveler hurrying toward the gate, bag slung over one shoulder",
        "a conductor pausing near the boarding ramp",
        "a family waiting anxiously near the luggage carousel",
    ],
}

_manifest_lock = threading.Lock()

# ----------------------------------------------------------------------------
# Cyclic sampler — even coverage over many samples instead of pure rng.choice
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

    def sample_n_distinct(self, n):
        """Pop n distinct items (refilling/reshuffling as needed)."""
        out, seen = [], set()
        guard = 0
        while len(out) < n:
            guard += 1
            if guard > (len(self.items) + 1) * (n + 2):
                raise RuntimeError("Not enough distinct items in pool for sample_n_distinct")
            item = self.next()
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out


# ----------------------------------------------------------------------------
# Per-object assignment: distinct location (drawn from the shot type's own
# depth-appropriate pool) per object, plus a grounded surface, orientation,
# color, and material.
# ----------------------------------------------------------------------------

def assign_object_details(objects, theme, shot_type, location_sampler, surface_sampler,
                           orientation_sampler, color_sampler, material_sampler,
                           override_samplers=None):
    locations = location_sampler.sample_n_distinct(len(objects))
    override_samplers = override_samplers or {}
    assignments = []
    for obj, (loc_key, loc_text) in zip(objects, locations):
        orient_key, orient_text = orientation_sampler.next()
        color = color_sampler.next()
        material = material_sampler.next()
        if obj in override_samplers:
            surface_text = override_samplers[obj].next()
        else:
            surface_text = surface_sampler.next()
        assignments.append({
            "object": obj,
            "surface_text": surface_text,
            "location_key": loc_key,
            "location_text": loc_text,
            "orientation_key": orient_key,
            "orientation_text": orient_text,
            "color": color,
            "material": material,
        })
    return assignments


def build_structured_description(assignments):
    lines = []
    for a in assignments:
        article = "An" if a["color"][0].lower() in "aeiou" else "A"
        lines.append(
            f"- {article} {a['color']} {a['material']} {a['object']}, "
            f"{a['surface_text']}, positioned {a['location_text']}, {a['orientation_text']}."
        )
    return "\n".join(lines)


COMPOSITION_PROMPT_TEMPLATE = """You are writing a single cinematic image/video generation prompt for a "{theme}" scene, captured as {shot_label}.

Setting: {setting}
Camera & framing: {camera_text}
{character_block}
The scene must contain exactly these {n} objects, each grounded on a real surface with its own location + orientation, exactly as specified (do not add, remove, merge, or relocate any of them, and do not invent extra props):
{structured_block}

Write ONE cohesive, cinematic scene description (70-110 words) that:
- Opens by establishing the setting and this shot's framing / depth of field
- Places each of the {n} objects believably on the physical surface it's grounded to (table, wall, floor, shelf, curb, sand, grass, etc.), preserving its stated color, material/finish, location, and orientation
- {character_instruction}
- Hints at a story or a mood — a sense of who has been here, what just happened, or what's about to happen — rather than reading like a flat inventory list
- Reads as a single flowing paragraph, vivid and cinematic, suitable to be used directly as a photorealistic image/video generation prompt
- Does NOT use meta-language like "the objects are", "positioned at", "reference"
- Does NOT add any props beyond the {n} objects listed{character_object_note}

Respond with ONLY the scene description text. No preamble, no labels, no quotation marks.
"""


def build_prompt_text(theme, shot_type, assignments, character_text):
    structured_block = build_structured_description(assignments)
    shot_cfg = SHOT_TYPES[shot_type]

    if character_text:
        character_block = f"A human presence is also in frame (not one of the {N_OBJECTS} objects): {character_text}.\n"
        character_instruction = (
            "Weaves the human figure into the scene naturally — their pose, position, and "
            "relationship to the objects should reinforce the story, without ever being described as one of the objects"
        )
        character_object_note = " (the human figure is scene dressing, not a 7th object)"
    else:
        character_block = ""
        character_instruction = "Keeps the scene unpeopled — no human figures, just the environment and the objects"
        character_object_note = ""

    return COMPOSITION_PROMPT_TEMPLATE.format(
        theme=theme.replace("_", " "),
        shot_label=shot_cfg["label"],
        setting=THEME_SETTINGS[theme],
        camera_text=shot_cfg["camera_text"],
        character_block=character_block,
        n=N_OBJECTS,
        structured_block=structured_block,
        character_instruction=character_instruction,
        character_object_note=character_object_note,
    )


# ----------------------------------------------------------------------------
# Max-permutation indexing — enumerate pool^repeat without materializing
# ----------------------------------------------------------------------------

def nth_product(pool, repeat, index):
    """Return the index-th element of itertools.product(pool, repeat=repeat)."""
    n = len(pool)
    return tuple(pool[(index // (n ** j)) % n] for j in range(repeat))


def _per_theme_counts(total_samples, num_themes):
    base, remainder = divmod(total_samples, num_themes)
    return [base + (1 if i < remainder else 0) for i in range(num_themes)]


def _location_perms_by_shot(shot_type_keys, rng):
    perms = {}
    for shot in shot_type_keys:
        pool = SHOT_TYPES[shot]["locations"]
        plist = list(itertools.permutations(pool, N_OBJECTS))
        rng.shuffle(plist)
        perms[shot] = plist
    return perms


def prepare_tasks_max_permutations(total_samples, seed, shot_type_keys, people_fraction):
    """
    Systematically walk the largest independent permutation axes so samples
    stay as unique as possible up to total_samples.
    """
    rng = random.Random(seed)
    num_themes = len(THEMES)
    per_theme = _per_theme_counts(total_samples, num_themes)

    location_perms = _location_perms_by_shot(shot_type_keys, rng)
    object_indices = list(range(N_OBJECTS))

    tasks = []
    sample_idx = 0
    for theme_i, (theme, objects) in enumerate(THEMES.items()):
        n_samples = per_theme[theme_i]
        theme_stride = theme_i * 1_000_003  # large coprime-ish offset per theme
        surfaces = THEME_SURFACES[theme]
        characters = THEME_CHARACTERS[theme]

        for i in range(n_samples):
            global_i = theme_stride + i
            shot_type = shot_type_keys[global_i % len(shot_type_keys)]
            n_loc = len(location_perms[shot_type])
            loc_perm = location_perms[shot_type][global_i % n_loc]

            slot_perm = list(object_indices)
            rng_obj = random.Random(seed + global_i * 97)
            rng_obj.shuffle(slot_perm)
            ordered_objects = [objects[j] for j in slot_perm]

            orient_tuples = nth_product(ORIENTATIONS, N_OBJECTS, global_i)
            color_tuples = nth_product(COLORS, N_OBJECTS, global_i * 7 + theme_i)
            material_tuples = nth_product(MATERIALS, N_OBJECTS, global_i * 13 + theme_i)
            surface_tuples = nth_product(surfaces, N_OBJECTS, global_i * 19 + theme_i)

            has_character = rng_obj.random() < people_fraction
            character_text = rng_obj.choice(characters) if has_character else None

            assignments = []
            for obj, loc, orient, color, material, surface in zip(
                ordered_objects, loc_perm, orient_tuples, color_tuples, material_tuples, surface_tuples
            ):
                loc_key, loc_text = loc
                orient_key, orient_text = orient
                if obj in OBJECT_SURFACE_OVERRIDES:
                    override_pool = OBJECT_SURFACE_OVERRIDES[obj]
                    surface = override_pool[rng_obj.randrange(len(override_pool))]
                assignments.append({
                    "object": obj,
                    "surface_text": surface,
                    "location_key": loc_key,
                    "location_text": loc_text,
                    "orientation_key": orient_key,
                    "orientation_text": orient_text,
                    "color": color,
                    "material": material,
                })

            prompt_text = build_prompt_text(theme, shot_type, assignments, character_text)
            sample_id = f"{theme}_{sample_idx:06d}"
            tasks.append((sample_id, theme, shot_type, assignments, character_text, prompt_text))
            sample_idx += 1

    rng.shuffle(tasks)
    return tasks


# ----------------------------------------------------------------------------
# Task preparation (deterministic given seed; no LLM calls here)
# ----------------------------------------------------------------------------

def _build_task(theme, objects, shot_type_sampler, samplers_by_shot, surface_sampler,
                 character_sampler, orientation_sampler, color_sampler, material_sampler,
                 people_fraction, rng, sample_idx, override_samplers=None):
    shot_type = shot_type_sampler.next()
    location_sampler = samplers_by_shot[shot_type]
    assignments = assign_object_details(
        objects, theme, shot_type, location_sampler, surface_sampler,
        orientation_sampler, color_sampler, material_sampler, override_samplers,
    )
    character_text = character_sampler.next() if rng.random() < people_fraction else None
    prompt_text = build_prompt_text(theme, shot_type, assignments, character_text)
    sample_id = f"{theme}_{sample_idx:06d}"
    return (sample_id, theme, shot_type, assignments, character_text, prompt_text)


def prepare_tasks(num_per_theme, seed, shot_type_keys, people_fraction):
    rng = random.Random(seed)

    samplers_by_shot = {s: CyclicSampler(SHOT_TYPES[s]["locations"], rng) for s in shot_type_keys}
    shot_type_sampler = CyclicSampler(shot_type_keys, rng)
    orientation_sampler = CyclicSampler(ORIENTATIONS, rng)
    color_sampler = CyclicSampler(COLORS, rng)
    material_sampler = CyclicSampler(MATERIALS, rng)

    override_samplers = {
        obj: CyclicSampler(texts, rng) for obj, texts in OBJECT_SURFACE_OVERRIDES.items()
    }

    tasks = []
    sample_idx = 0
    for theme, objects in THEMES.items():
        assert len(objects) == N_OBJECTS, f"{theme} must have exactly {N_OBJECTS} objects"
        surface_sampler = CyclicSampler(THEME_SURFACES[theme], rng)
        character_sampler = CyclicSampler(THEME_CHARACTERS[theme], rng)
        for _ in range(num_per_theme):
            tasks.append(_build_task(
                theme, objects, shot_type_sampler, samplers_by_shot, surface_sampler,
                character_sampler, orientation_sampler, color_sampler, material_sampler,
                people_fraction, rng, sample_idx, override_samplers,
            ))
            sample_idx += 1

    rng.shuffle(tasks)  # interleave themes rather than writing big contiguous blocks
    return tasks


def prepare_tasks_total(total_samples, seed, shot_type_keys, people_fraction):
    """Cyclic-sampler variant balanced across themes for an exact total count."""
    rng = random.Random(seed)
    samplers_by_shot = {s: CyclicSampler(SHOT_TYPES[s]["locations"], rng) for s in shot_type_keys}
    shot_type_sampler = CyclicSampler(shot_type_keys, rng)
    orientation_sampler = CyclicSampler(ORIENTATIONS, rng)
    color_sampler = CyclicSampler(COLORS, rng)
    material_sampler = CyclicSampler(MATERIALS, rng)

    override_samplers = {
        obj: CyclicSampler(texts, rng) for obj, texts in OBJECT_SURFACE_OVERRIDES.items()
    }

    per_theme = _per_theme_counts(total_samples, len(THEMES))
    tasks = []
    sample_idx = 0
    for (theme, objects), n_samples in zip(THEMES.items(), per_theme):
        assert len(objects) == N_OBJECTS, f"{theme} must have exactly {N_OBJECTS} objects"
        surface_sampler = CyclicSampler(THEME_SURFACES[theme], rng)
        character_sampler = CyclicSampler(THEME_CHARACTERS[theme], rng)
        for _ in range(n_samples):
            tasks.append(_build_task(
                theme, objects, shot_type_sampler, samplers_by_shot, surface_sampler,
                character_sampler, orientation_sampler, color_sampler, material_sampler,
                people_fraction, rng, sample_idx, override_samplers,
            ))
            sample_idx += 1

    rng.shuffle(tasks)
    return tasks


# ----------------------------------------------------------------------------
# LLM worker (text stage — turns structured description into one paragraph)
# ----------------------------------------------------------------------------

def process_one(client_getter, sample_id, theme, shot_type, assignments, character_text,
                 prompt_text, samples_dir, dry_run, max_retries=3):
    saved_path = os.path.join(samples_dir, f"{sample_id}.json")

    result = {
        "id": sample_id,
        "theme": theme,
        "shot_type": shot_type,
        "has_character": character_text is not None,
        "character_text": character_text,
        "num_objects": len(assignments),
        "objects": [a["object"] for a in assignments],
        "object_details": assignments,
        "structured_prompt": prompt_text,
        "combined_prompt": None,
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
                raw_text = response.choices[0].message.content.strip().strip('"\'')
                result["combined_prompt"] = raw_text
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
# Metadata CSV + distribution summary
# ----------------------------------------------------------------------------

def write_metadata_csv(tasks_or_results, csv_path, is_result):
    fieldnames = ["id", "theme", "shot_type", "has_character", "objects",
                  "locations", "orientations", "colors", "materials", "surfaces"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in tasks_or_results:
            if is_result:
                sid, theme = item["id"], item["theme"]
                shot_type = item["shot_type"]
                has_character = item["has_character"]
                assignments = item["object_details"]
            else:
                sid, theme, shot_type, assignments, character_text, _ = item
                has_character = character_text is not None
            writer.writerow({
                "id": sid,
                "theme": theme,
                "shot_type": shot_type,
                "has_character": has_character,
                "objects": "|".join(a["object"] for a in assignments),
                "locations": "|".join(a["location_key"] for a in assignments),
                "orientations": "|".join(a["orientation_key"] for a in assignments),
                "colors": "|".join(a["color"] for a in assignments),
                "materials": "|".join(a["material"] for a in assignments),
                "surfaces": "|".join(a["surface_text"] for a in assignments),
            })


def write_distribution_summary(tasks_or_results, summary_path, is_result):
    theme_counter = Counter()
    shot_counter = Counter()
    character_counter = Counter()
    location_counter = Counter()
    orientation_counter = Counter()
    color_counter = Counter()
    material_counter = Counter()

    for item in tasks_or_results:
        if is_result:
            theme = item["theme"]
            shot_type = item["shot_type"]
            has_character = item["has_character"]
            assignments = item["object_details"]
        else:
            _, theme, shot_type, assignments, character_text, _ = item
            has_character = character_text is not None
        theme_counter[theme] += 1
        shot_counter[shot_type] += 1
        character_counter["with_character" if has_character else "no_character"] += 1
        for a in assignments:
            location_counter[a["location_key"]] += 1
            orientation_counter[a["orientation_key"]] += 1
            color_counter[a["color"]] += 1
            material_counter[a["material"]] += 1

    total = sum(theme_counter.values())
    lines = [f"TOTAL SAMPLES: {total}\n"]

    def section(title, counter):
        lines.append(f"--- {title} ---")
        for k, v in sorted(counter.items(), key=lambda x: str(x[0])):
            lines.append(f"  {k}: {v}")
        lines.append("")

    section("samples per theme", theme_counter)
    section("shot type usage", shot_counter)
    section("character presence", character_counter)
    section("location usage (across all object slots)", location_counter)
    section("orientation usage", orientation_counter)
    section("color usage", color_counter)
    section("material usage", material_counter)

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(lines[:50]))
    print(f"\n(Full summary written to {summary_path})")


# ----------------------------------------------------------------------------
# STAGE 2 — photorealistic image generation from the manifest.
#
# Deliberately reads `combined_prompt` (the cinematic paragraph an LLM wrote
# from the structured description), never the raw bullet list — that's the
# whole point: the composition, grounding, and story context are already
# baked into that paragraph, so the image model gets a coherent scene
# description instead of a prop inventory.
# ----------------------------------------------------------------------------

PHOTOREAL_SUFFIX = (
    ", photorealistic, shot on 35mm film, natural lighting, shallow depth of field, "
    "ultra-detailed, high dynamic range, professional photography, 8k"
)

NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, drawing, sketch, cgi, 3d render, video game, "
    "plastic skin, wax figure, overexposed, underexposed, blurry, low quality, low detail, "
    "watermark, signature, text, logo, deformed, disfigured, extra limbs, bad anatomy, "
    "duplicate, frame, border"
)


def load_image_pipeline(model_id, device):
    """
    Loads a local diffusers text-to-image pipeline once. Works with any
    diffusers-compatible checkpoint; SDXL and FLUX.1-dev both give strong
    photoreal results. Swap model_id via --image-model.
    """
    import torch
    from diffusers import DiffusionPipeline

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True)
    pipe = pipe.to(device)
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe.enable_vae_slicing()
    return pipe


def generate_images_from_manifest(manifest_path, images_dir, model_id, steps, guidance,
                                   width, height, seed, limit, skip_existing):
    import torch

    os.makedirs(images_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device found — running on CPU will be extremely slow.")

    print(f"Loading image model '{model_id}' on {device} ...")
    pipe = load_image_pipeline(model_id, device)
    generator = torch.Generator(device=device)

    records = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if limit:
        records = records[:limit]

    ok, skipped, failed = 0, 0, 0
    for rec in tqdm(records, desc="Rendering", unit="image"):
        sample_id = rec["id"]
        out_path = os.path.join(images_dir, f"{sample_id}.png")
        if skip_existing and os.path.exists(out_path):
            skipped += 1
            continue

        cinematic_prompt = rec.get("combined_prompt")
        if not cinematic_prompt:
            # Fall back to the structured description only if the LLM stage
            # was skipped (dry-run manifests) — still keeps all objects,
            # surfaces, locations, and orientations, just less flowing prose.
            cinematic_prompt = rec["structured_prompt"]

        full_prompt = cinematic_prompt.strip() + PHOTOREAL_SUFFIX
        generator.manual_seed(seed + hash(sample_id) % (2**31))

        try:
            image = pipe(
                prompt=full_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
            ).images[0]
            image.save(out_path)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[{sample_id}] generation failed: {e}")

    print("\n" + "=" * 60)
    print(f"Image render complete. OK={ok}  SKIPPED={skipped}  FAILED={failed}")
    print(f"Images written to: {images_dir}")
    print("=" * 60)


# ----------------------------------------------------------------------------
# Pipeline runner (Stage 1: prompt composition)
# ----------------------------------------------------------------------------

def run_pipeline(num_per_theme, total_samples, max_permutations, num_threads, seed, dry_run,
                  shot_type_keys, people_fraction):
    samples_dir = os.path.join(OUTPUT_ROOT, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    manifest_path = os.path.join(OUTPUT_ROOT, "manifest.jsonl")
    csv_path = os.path.join(OUTPUT_ROOT, "metadata_preview.csv")
    summary_path = os.path.join(OUTPUT_ROOT, "distribution_summary.txt")

    if total_samples is not None:
        if max_permutations:
            tasks = prepare_tasks_max_permutations(total_samples, seed, shot_type_keys, people_fraction)
            mode = "max-permutations"
        else:
            tasks = prepare_tasks_total(total_samples, seed, shot_type_keys, people_fraction)
            mode = "cyclic"
        per_theme_actual = _per_theme_counts(total_samples, len(THEMES))
        print(f"Prepared {len(tasks)} samples across {len(THEMES)} scenes "
              f"({per_theme_actual[0]}-{per_theme_actual[-1]} per scene, "
              f"{N_OBJECTS} objects each, shots={shot_type_keys}, "
              f"people_fraction={people_fraction}, mode={mode})")
    else:
        tasks = prepare_tasks(num_per_theme, seed, shot_type_keys, people_fraction)
        print(f"Prepared {len(tasks)} samples across {len(THEMES)} scenes "
              f"({num_per_theme} per scene, {N_OBJECTS} objects each, "
              f"shots={shot_type_keys}, people_fraction={people_fraction})")

    if dry_run:
        print("\n[DRY RUN] Skipping LLM calls. Writing metadata for review only.\n")
        write_metadata_csv(tasks, csv_path, is_result=False)
        write_distribution_summary(tasks, summary_path, is_result=False)

        with open(manifest_path, "w") as manifest_f:
            for sid, theme, shot_type, assignments, character_text, prompt_text in tasks:
                record = {
                    "id": sid,
                    "theme": theme,
                    "shot_type": shot_type,
                    "has_character": character_text is not None,
                    "character_text": character_text,
                    "num_objects": len(assignments),
                    "objects": [a["object"] for a in assignments],
                    "object_details": assignments,
                    "structured_prompt": prompt_text,
                    "combined_prompt": None,
                    "error": None,
                    "dry_run": True,
                }
                manifest_f.write(json.dumps(record) + "\n")

        print("\n" + "=" * 60)
        print("DRY RUN COMPLETE — no LLM/GPU calls were made.")
        print(f"Manifest:          {manifest_path}")
        print(f"Metadata CSV:      {csv_path}")
        print(f"Distribution info: {summary_path}")
        print("=" * 60)
        return

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
                executor.submit(process_one, get_client, sid, theme, shot_type, assignments,
                                 character_text, prompt_text, samples_dir, dry_run): sid
                for sid, theme, shot_type, assignments, character_text, prompt_text in tasks
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num-per-theme", type=int, default=None,
                       help="Prompts per scene (x10 scenes). Ignored if --total is set.")
    group.add_argument("--total", type=int, default=None,
                       help="Total prompts across all scenes (e.g. 10000)")
    parser.add_argument("--max-permutations", action="store_true",
                        help="Systematically enumerate location/object/orientation/color/material "
                             "combinations for maximum diversity (recommended with --total)")
    parser.add_argument("--shot-types", type=str, default="close_up,mid_shot,wide_shot",
                        help="Comma-separated subset of: close_up, mid_shot, wide_shot")
    parser.add_argument("--people-fraction", type=float, default=0.4,
                        help="Fraction of samples (0-1) that include a human figure in the scene")
    parser.add_argument("--threads", type=int, default=NUM_THREADS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Sample everything and write metadata/CSV/summary, but skip LLM calls")

    # Stage 2: image generation
    parser.add_argument("--generate-images", action="store_true",
                        help="Skip prompt composition; render photoreal images from an existing manifest.jsonl")
    parser.add_argument("--image-model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Any diffusers-compatible text-to-image checkpoint (SDXL, FLUX.1-dev, etc.)")
    parser.add_argument("--steps", type=int, default=40, help="Diffusion inference steps")
    parser.add_argument("--guidance", type=float, default=6.5,
                        help="Classifier-free guidance scale — lower (~5-7) tends to look more photoreal")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="Only render the first N manifest rows")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Re-render images even if a PNG for that sample id already exists")

    args = parser.parse_args()

    if args.generate_images:
        manifest_path = os.path.join(OUTPUT_ROOT, "manifest.jsonl")
        images_dir = os.path.join(OUTPUT_ROOT, "images")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"No manifest found at {manifest_path}. Run stage 1 first "
                f"(e.g. `python objects.py --total 5000 --max-permutations`)."
            )
        generate_images_from_manifest(
            manifest_path=manifest_path,
            images_dir=images_dir,
            model_id=args.image_model,
            steps=args.steps,
            guidance=args.guidance,
            width=args.width,
            height=args.height,
            seed=args.seed,
            limit=args.limit,
            skip_existing=not args.no_skip_existing,
        )
    else:
        num_per_theme = args.num_per_theme if args.num_per_theme is not None else 100

        shot_type_keys = [s.strip() for s in args.shot_types.split(",") if s.strip()]
        for s in shot_type_keys:
            if s not in SHOT_TYPES:
                raise ValueError(f"Unknown shot type '{s}'. Choose from {list(SHOT_TYPES.keys())}")
        if not (0.0 <= args.people_fraction <= 1.0):
            raise ValueError("--people-fraction must be between 0 and 1")

        run_pipeline(
            num_per_theme=num_per_theme,
            total_samples=args.total,
            max_permutations=args.max_permutations,
            num_threads=args.threads,
            seed=args.seed,
            dry_run=args.dry_run,
            shot_type_keys=shot_type_keys,
            people_fraction=args.people_fraction,
        )
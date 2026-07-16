#!/usr/bin/env python3
"""
Sweep guidance_scale for stage3_lora_infer.py (single model load).

Default scales: 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5

Example::

    cd infer_v2
    python run_guidance_sweep.py

    python run_guidance_sweep.py --output outputs/guidance_sweep/image_gen_lora.png

    python run_guidance_sweep.py --scales 1 2 3 --seed 42 --no-canvas_depth_coarse_augment

Outputs (default base ``outputs/guidance_sweep/image_gen_lora.png``)::

    outputs/guidance_sweep/image_gen_lora_gs1.png
    outputs/guidance_sweep/image_gen_lora_gs1p5.png
    ...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_INFER_DIR = Path(__file__).resolve().parent
_STAGE3 = _INFER_DIR / "stage3_lora_infer.py"

DEFAULT_SCALES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,5 ,5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10)
DEFAULT_OUTPUT = _INFER_DIR / "outputs_black_canvas" / "guidance_sweep" / "image_gen_lora.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stage3 LoRA inference across multiple guidance_scale values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=list(DEFAULT_SCALES),
        help="Guidance scale values to sweep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Base output path; each scale writes <stem>_gs<scale><suffix>.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scales whose output PNG already exists.",
    )
    args, extra = parser.parse_known_args()

    out_base = args.output.expanduser().resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)

    scales: list[float] = []
    for scale in args.scales:
        scale = float(scale)
        if scale in scales:
            continue
        label = f"{scale:g}".replace(".", "p")
        out_path = out_base.with_name(f"{out_base.stem}_gs{label}{out_base.suffix}")
        if args.skip_existing and out_path.is_file():
            print(f"Skip existing gs={scale:g} → {out_path}")
            continue
        scales.append(scale)

    if not scales:
        print("All requested scale outputs already exist.")
        return

    cmd = [
        sys.executable,
        str(_STAGE3),
        "--guidance_scales",
        *[str(s) for s in scales],
        "--output",
        str(out_base),
        *extra,
    ]

    print(f"Scales: {scales}")
    print(f"Output base: {out_base}")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_INFER_DIR))


if __name__ == "__main__":
    main()

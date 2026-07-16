#!/usr/bin/env python3
"""
CLI client for ``lora_server.py``.

Sends a folder path and guidance scale; the server reads inputs from that folder
and writes ``image_gen_lora.png`` back into it.

Example::

    python lora_client.py \\
      --folder methodology_image \\
      --guidance-scale 7.0

    python lora_client.py \\
      --folder /path/to/generated_image \\
      --guidance-scale 8.5 \\
      --url http://127.0.0.1:8767

    # Black canvas reference (depth-only conditioning)
    python lora_client.py --folder methodology_image --guidance-scale 7.0 --no-canvas

    # Black depth reference (canvas-only conditioning)
    python lora_client.py --folder methodology_image --guidance-scale 7.0 --no-depth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing package 'requests'. Install via: pip install requests", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Client for stage3 LoRA inference server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8767",
        help="Server base URL",
    )
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Folder containing black_canvas.jpg, image_gen_depth.png, image_gen.txt",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        required=True,
        dest="guidance_scale",
        help="CFG guidance scale for this generation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28, help="Denoising steps")
    parser.add_argument(
        "--lora-first-depth-steps",
        type=int,
        default=0,
        help="If >0, depth LoRA priority for first N steps",
    )
    parser.add_argument(
        "--no-canvas",
        action="store_true",
        help="Use a black image for the canvas/subject reference instead of black_canvas.jpg",
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Use a black image for the depth reference instead of image_gen_depth.png",
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Only call GET /health and exit",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    if args.health_only:
        try:
            r = requests.get(f"{base_url}/health", timeout=30)
            health = r.json()
            print(json.dumps(health, indent=2))
            if r.status_code == 200 and health.get("api_version", 0) < 2:
                print(
                    "Warning: server api_version < 2; restart lora_server.py for no_canvas/no_depth support.",
                    file=sys.stderr,
                )
            return 0 if r.status_code == 200 else 1
        except requests.exceptions.ConnectionError:
            print(f"Error: cannot connect to {base_url}", file=sys.stderr)
            return 1

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Error: folder not found: {folder}", file=sys.stderr)
        return 1

    payload = {
        "folder_path": str(folder),
        "guidance_scale": float(args.guidance_scale),
        "seed": int(args.seed),
        "num_inference_steps": int(args.steps),
        "lora_first_depth_steps": int(args.lora_first_depth_steps),
        "no_canvas": bool(args.no_canvas),
        "no_depth": bool(args.no_depth),
    }

    print(f"Server: {base_url}")
    print(f"Folder: {folder}")
    print(f"Guidance scale: {args.guidance_scale:g}")
    if args.no_canvas:
        print("Canvas reference: black image")
    if args.no_depth:
        print("Depth reference: black image")

    try:
        r = requests.post(f"{base_url}/generate", json=payload, timeout=3600)
    except requests.exceptions.ConnectionError:
        print(f"Error: cannot connect to {base_url}. Is lora_server.py running?", file=sys.stderr)
        return 1

    if r.status_code != 200:
        detail = r.text
        try:
            detail = r.json().get("detail", detail)
        except Exception:
            pass
        print(f"Generation failed (HTTP {r.status_code}): {detail}", file=sys.stderr)
        return 1

    data = r.json()
    print(json.dumps(data, indent=2))

    if args.no_canvas and data.get("canvas_source") != "black":
        print(
            "Error: server still used a canvas file. Restart lora_server.py to pick up no_canvas support.",
            file=sys.stderr,
        )
        return 1
    if args.no_depth and data.get("depth_source") != "black":
        print(
            "Error: server still used a depth file. Restart lora_server.py to pick up no_depth support.",
            file=sys.stderr,
        )
        return 1

    print(f"Canvas: {data.get('canvas_source', '?')}")
    print(f"Depth:  {data.get('depth_source', '?')}")
    print(f"Saved -> {data['output_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

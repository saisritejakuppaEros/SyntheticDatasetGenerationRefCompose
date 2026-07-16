"""JSONL manifest dataset for v2 synthetic scenes (gt, canvas, depth, caption)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logzero
import numpy as np
import torch
from PIL import Image

from .canvas_dataset import (
    coarse_degrade_depth_rgb_pil,
    collate_fn_canvas,
    load_depth_image_as_rgb_pil,
    pil_to_model_tensor,
    resize_cover_pil,
)
from .jsonl_datasets import get_random_resolution, load_image_safely, multiple_16


def parse_manifest_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one v2 manifest JSON object to training paths."""
    canvas = record.get("canvas") or {}
    depth = record.get("depth") or {}
    return {
        "id": record.get("id", ""),
        "caption": (record.get("caption") or "").strip(),
        "gt_image": record.get("gt_image") or "",
        "canvas_image": record.get("canvas_image") or canvas.get("image") or "",
        "depth_image": record.get("depth_image") or depth.get("image") or "",
        "image_width": int(record.get("image_width") or 0),
        "image_height": int(record.get("image_height") or 0),
    }


def load_manifest_records(manifest_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no} of {path}") from exc
            records.append(parse_manifest_record(raw))
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def _path_exists(p: str) -> bool:
    if not p or not str(p).strip():
        return False
    try:
        return Path(p).is_file()
    except OSError:
        return False


def filter_valid_manifest_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Keep rows with gt + canvas + depth + caption and existing files on disk."""
    stats = {
        "total": len(records),
        "kept": 0,
        "missing_caption": 0,
        "missing_gt": 0,
        "missing_canvas": 0,
        "missing_depth": 0,
        "missing_file": 0,
    }
    kept: List[Dict[str, Any]] = []
    for rec in records:
        if not rec["caption"]:
            stats["missing_caption"] += 1
            continue
        if not rec["gt_image"]:
            stats["missing_gt"] += 1
            continue
        if not rec["canvas_image"]:
            stats["missing_canvas"] += 1
            continue
        if not rec["depth_image"]:
            stats["missing_depth"] += 1
            continue
        if not (
            _path_exists(rec["gt_image"])
            and _path_exists(rec["canvas_image"])
            and _path_exists(rec["depth_image"])
        ):
            stats["missing_file"] += 1
            continue
        kept.append(rec)
        stats["kept"] += 1
    return kept, stats


def log_manifest_asset_pairing_report(args: Any) -> None:
    manifest_path = getattr(args, "manifest_path", "") or getattr(args, "train_data_dir", "") or ""
    if not manifest_path:
        logzero.logger.warning("[train][manifest] skip report: manifest_path unset")
        return
    records = load_manifest_records(manifest_path)
    _, stats = filter_valid_manifest_records(records)

    def lz(msg: str) -> None:
        logzero.logger.info("[train][manifest] %s", msg)

    lz(f"manifest={manifest_path}")
    lz(f"rows read: {stats['total']}")
    lz(f"rows kept (caption + gt + canvas + depth + files): {stats['kept']}")
    if stats["missing_caption"]:
        lz(f"dropped missing caption: {stats['missing_caption']}")
    if stats["missing_gt"]:
        lz(f"dropped missing gt_image: {stats['missing_gt']}")
    if stats["missing_canvas"]:
        lz(f"dropped missing canvas_image: {stats['missing_canvas']}")
    if stats["missing_depth"]:
        lz(f"dropped missing depth_image: {stats['missing_depth']}")
    if stats["missing_file"]:
        lz(f"dropped missing file on disk: {stats['missing_file']}")


class ManifestSceneDataset(torch.utils.data.Dataset):
    """One sample per manifest row: gt RGB target, canvas subject cond, depth spatial cond."""

    _missing_depth_warning_emitted = False

    def __init__(self, manifest_path: str, args: Any):
        super().__init__()
        self.args = args
        records = load_manifest_records(manifest_path)
        self.records, self._stats = filter_valid_manifest_records(records)
        if not self.records:
            raise ValueError(
                f"no valid manifest rows in {manifest_path} "
                f"(stats={self._stats}); need caption, gt_image, canvas_image, depth_image with existing files"
            )

        self._noise_cap = args.noise_size
        self._random_target_res = bool(getattr(args, "canvas_random_target_resolution", False))
        uw = int(getattr(args, "unified_train_width", 1280))
        uh = int(getattr(args, "unified_train_height", 720))
        self._unified_wh: Optional[Tuple[int, int]] = None
        if not self._random_target_res:
            self._unified_wh = (multiple_16(uw), multiple_16(uh))

        mode = getattr(args, "canvas_unified_resize", "cover")
        if mode not in ("cover", "contain"):
            raise ValueError(f"canvas_unified_resize must be 'cover' or 'contain', got {mode!r}")
        self._unified_resize_mode = mode

        self._depth_keep_prob = float(getattr(args, "depth_keep_prob", 1.0))
        if not (0.0 <= self._depth_keep_prob <= 1.0):
            raise ValueError(f"depth_keep_prob must be in [0, 1], got {self._depth_keep_prob}")

        self._canvas_keep_prob = float(getattr(args, "canvas_keep_prob", 1.0))
        if not (0.0 <= self._canvas_keep_prob <= 1.0):
            raise ValueError(f"canvas_keep_prob must be in [0, 1], got {self._canvas_keep_prob}")

        self._depth_coarse_augment = bool(getattr(args, "canvas_depth_coarse_augment", True))
        self._canvas_augment_enabled = True

        if self._random_target_res:
            raise ValueError(
                "manifest dataset uses unified training resolution; "
                "do not pass --canvas_random_target_resolution"
            )

    def __len__(self) -> int:
        return len(self.records)

    def _depth_coarse_augment_active(self) -> bool:
        if not self._depth_coarse_augment:
            return False
        if not getattr(self.args, "canvas_augment", True):
            return False
        if getattr(self, "_canvas_augment_enabled", True):
            return True
        return bool(getattr(self.args, "validation_depth_coarse_augment", True))

    def _finalize_depth_cond(self, depth_pil: Image.Image, tw: int, th: int) -> Image.Image:
        kp = float(self._depth_keep_prob)
        if kp >= 1.0:
            return depth_pil
        if kp <= 0.0:
            return Image.new("RGB", (tw, th), (0, 0, 0))
        if float(np.random.random()) < kp:
            return depth_pil
        return Image.new("RGB", (tw, th), (0, 0, 0))

    def _finalize_subject_pixel_tensor(self, t: torch.Tensor) -> torch.Tensor:
        kp = float(self._canvas_keep_prob)
        if kp >= 1.0:
            return t
        if kp <= 0.0:
            return torch.full_like(t, -1.0)
        if float(np.random.random()) < kp:
            return t
        return torch.full_like(t, -1.0)

    def _load_depth_pil(self, depth_path: str, tw: int, th: int) -> Image.Image:
        if not _path_exists(depth_path):
            if not ManifestSceneDataset._missing_depth_warning_emitted:
                ManifestSceneDataset._missing_depth_warning_emitted = True
                warnings.warn(
                    f"depth file missing: {depth_path!r}; using black placeholder",
                    stacklevel=2,
                )
            depth_pil = Image.new("RGB", (tw, th), (0, 0, 0))
        else:
            depth_pil = resize_cover_pil(load_depth_image_as_rgb_pil(depth_path), tw, th)
            if self._depth_coarse_augment_active():
                depth_pil = coarse_degrade_depth_rgb_pil(depth_pil, self.args)
        return self._finalize_depth_cond(depth_pil, tw, th)

    def _target_transform(self, image: Image.Image, noise_size: int) -> torch.Tensor:
        from torchvision import transforms

        tfm = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: img.resize(
                        (
                            multiple_16(noise_size * img.size[0] / max(img.size)),
                            multiple_16(noise_size * img.size[1] / max(img.size)),
                        ),
                        resample=Image.BILINEAR,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        return tfm(image)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        tw, th = self._unified_wh or (1280, 720)

        target = load_image_safely(rec["gt_image"], self.args.cond_size)
        canvas = load_image_safely(rec["canvas_image"], self.args.cond_size)

        tgt_u = resize_cover_pil(target.convert("RGB"), tw, th)
        canvas_u = resize_cover_pil(canvas.convert("RGB"), tw, th)
        depth_u = self._load_depth_pil(rec["depth_image"], tw, th)

        out: Dict[str, Any] = {
            "pixel_values": pil_to_model_tensor(tgt_u),
            "prompts": rec["caption"],
            "subject_pixel_values": self._finalize_subject_pixel_tensor(pil_to_model_tensor(canvas_u)),
            "cond_pixel_values": pil_to_model_tensor(depth_u),
        }
        return out


def make_manifest_train_dataset(args: Any, accelerator=None) -> ManifestSceneDataset:
    manifest_path = getattr(args, "manifest_path", "") or getattr(args, "train_data_dir", "") or ""
    if not manifest_path:
        raise ValueError("dataset_type=manifest requires --manifest_path or --train_data_dir")
    if accelerator is not None:
        with accelerator.main_process_first():
            return ManifestSceneDataset(manifest_path, args)
    return ManifestSceneDataset(manifest_path, args)


__all__ = [
    "ManifestSceneDataset",
    "collate_fn_canvas",
    "filter_valid_manifest_records",
    "load_manifest_records",
    "log_manifest_asset_pairing_report",
    "make_manifest_train_dataset",
    "parse_manifest_record",
]

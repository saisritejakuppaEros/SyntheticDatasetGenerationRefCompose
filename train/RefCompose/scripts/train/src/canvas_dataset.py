"""CSV-driven dataset for canvas / scene training (Flux2)."""

from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path

import numpy as np
import logzero
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TVF

from .canvas_bbox_compose import (
    CANVAS_H,
    CANVAS_W,
    build_canvas_for_image_rel,
    map_rows_xyxy_to_cover_crop_space,
    prepare_groups_from_flat_rows,
    prompt_for_group,
)
from .jsonl_datasets import get_random_resolution, load_image_safely, multiple_16

Image.MAX_IMAGE_PIXELS = None

# Relative layout under each ``--canvas_data_roots`` entry (same as dataset_prep output trees).
CANVAS_DATA_ROOT_CSV = Path("bbox_results") / "yolo26_detections.csv"


def parse_canvas_data_roots(raw: Optional[str]) -> List[str]:
    """Split ``--canvas_data_roots`` (comma/semicolon/newline separated) into non-empty absolute or relative roots."""
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    return [p.strip() for p in re.split(r"[\n,;]+", s) if p.strip()]


def resolve_canvas_data_paths(root: Path) -> Dict[str, Path]:
    """Paths under one dataset ``output`` root for merged training."""
    root = root.expanduser().resolve()
    return {
        "csv": root / CANVAS_DATA_ROOT_CSV,
        "images": root / "images",
        "depth": root / "depth",
        "captions": root / "image_captions",
        "multiview": root / "multiview_out",
    }


class CanvasConcatDataset(torch.utils.data.ConcatDataset):
    """Multiple ``CanvasSceneDataset`` roots; proxies validation toggles to every child."""

    _PROXY_ATTRS = frozenset({"_canvas_augment_enabled", "_depth_keep_prob", "_canvas_keep_prob"})

    def __init__(self, datasets: List["CanvasSceneDataset"]):
        object.__setattr__(self, "_canvas_children", list(datasets))
        super().__init__(datasets)

    def __getattr__(self, name: str) -> Any:
        if name in CanvasConcatDataset._PROXY_ATTRS:
            ch = object.__getattribute__(self, "_canvas_children")
            if not ch:
                raise AttributeError(name)
            return getattr(ch[0], name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in CanvasConcatDataset._PROXY_ATTRS and hasattr(self, "_canvas_children"):
            for ds in object.__getattribute__(self, "_canvas_children"):
                setattr(ds, name, value)
            return
        super().__setattr__(name, value)


def concat_dataset_child_lengths(ds: torch.utils.data.Dataset) -> Optional[list[int]]:
    """If ``ds`` is a multi-child :class:`torch.utils.data.ConcatDataset`, return ``[len(c) for c in ds.datasets]``."""
    if not isinstance(ds, torch.utils.data.ConcatDataset):
        return None
    parts = getattr(ds, "datasets", None) or []
    if len(parts) < 2:
        return None
    return [len(c) for c in parts]


def balanced_sampling_weights_for_concat(ds: torch.utils.data.ConcatDataset) -> torch.Tensor:
    """Weights for :class:`torch.utils.data.WeightedRandomSampler`: each concat child is chosen with probability ``1/k``.

    Child ``j`` has ``n_j`` indices; each gets weight ``1/n_j`` so the total mass on child ``j`` is ``n_j * (1/n_j) = 1``,
    normalized against ``k`` children gives equal root probability.
    """
    lengths = concat_dataset_child_lengths(ds)
    if lengths is None:
        raise ValueError("balanced_sampling_weights_for_concat expects a ConcatDataset with at least two children")
    weights: list[float] = []
    for n in lengths:
        if n <= 0:
            raise ValueError("balanced_sampling_weights_for_concat: empty child dataset")
        weights.extend([1.0 / float(n)] * n)
    if len(weights) != len(ds):
        raise RuntimeError("internal: weight length != concat dataset length")
    return torch.tensor(weights, dtype=torch.double)


def resolve_depth_asset_path(depth_root: Path, rel: str) -> Optional[Path]:
    """Resolve depth file for CSV ``rel`` (same rules as training).

    Targets are often ``*.jpg`` while depth exports are ``*.png`` with the same **stem**;
    stem + ``.png`` is tried before the CSV basename.
    """
    rel_p = Path(rel)
    stem = rel_p.stem
    parent = rel_p.parent
    depth_exts = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")
    candidates: list[Path] = []
    for ext in depth_exts:
        candidates.append(depth_root / f"{stem}{ext}")
    if str(parent) not in (".", ""):
        for ext in depth_exts:
            candidates.append(depth_root / parent / f"{stem}{ext}")
    candidates.append(depth_root / rel_p.name)
    candidates.append(depth_root / rel)

    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None


def resolve_caption_asset_path(caption_root: Path, rel: str) -> Optional[Path]:
    """First existing caption file for ``rel`` (same search as ``_read_caption_for_rel``)."""
    stem = Path(rel).stem
    candidates = [
        caption_root / f"{stem}.txt",
        caption_root / Path(rel).with_suffix(".txt").name,
    ]
    parent = Path(rel).parent
    if str(parent) not in (".", ""):
        candidates.append(caption_root / parent / f"{stem}.txt")
    for p in candidates:
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def pairing_fs_image_path(rel: str, image_root: Optional[str]) -> Path:
    """Filesystem path for the original/target image (aligned with ``resolve_image_path`` / training)."""
    if not rel or not str(rel).strip():
        return Path()
    ir = image_root.strip() if isinstance(image_root, str) and image_root.strip() else ""
    p = Path(rel)
    if ir and not p.is_absolute():
        return (Path(ir).expanduser() / p).resolve()
    return p.resolve()


def log_canvas_asset_pairing_report(args: Any) -> None:
    """Before training: log how many CSV images pair with depth maps and caption files (matched by stem/name)."""
    csv_path = getattr(args, "csv_path", "") or ""
    if not csv_path or not Path(csv_path).is_file():
        logzero.logger.warning("[train][pairing] skip report: csv_path missing or not a file")
        return

    target_column = getattr(args, "canvas_target_column", "image_path")
    conditioning = getattr(args, "canvas_conditioning", "precomputed")
    image_root = getattr(args, "canvas_image_root", None)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        flat_rows = [{k.strip(): (v or "").strip() for k, v in row.items()} for row in reader]

    if conditioning == "bbox_multiview":
        bbox_min = float(getattr(args, "canvas_bbox_min_side", 0.0))
        mv_min = float(getattr(args, "canvas_multiview_match_min_side", 200.0))
        groups = prepare_groups_from_flat_rows(flat_rows, target_column, bbox_min, mv_min)
        unique_rels = [g[0] for g in groups]
    else:
        unique_rels = list(
            dict.fromkeys(
                row[target_column] for row in flat_rows if row.get(target_column, "").strip()
            )
        )

    n = len(unique_rels)
    depth_root: Optional[Path] = None
    dr = getattr(args, "depth_image_root", None) or ""
    if isinstance(dr, str) and dr.strip():
        depth_root = Path(dr).resolve()
    cap_root: Optional[Path] = None
    cap = getattr(args, "caption_dir", None)
    if isinstance(cap, str) and cap.strip():
        cap_root = Path(cap).resolve()

    need_depth = depth_root is not None and depth_root.is_dir()
    need_cap = cap_root is not None and cap_root.is_dir()
    if depth_root is not None and not depth_root.is_dir():
        logzero.logger.warning(
            "[train][pairing] depth_image_root is not a directory or missing: %s", depth_root
        )
    if cap_root is not None and not cap_root.is_dir():
        logzero.logger.warning(
            "[train][pairing] caption_dir is not a directory or missing: %s", cap_root
        )

    n_img = n_depth = n_cap = n_triple = 0
    missing_depth: list[str] = []
    missing_cap: list[str] = []
    missing_img: list[str] = []

    for rel in unique_rels:
        ip = pairing_fs_image_path(rel, image_root)
        has_img = ip.is_file()
        if has_img:
            n_img += 1
        else:
            if len(missing_img) < 12:
                missing_img.append(rel)

        has_d = False
        if need_depth:
            dp = resolve_depth_asset_path(depth_root, rel)
            has_d = dp is not None
            if has_d:
                n_depth += 1
            elif len(missing_depth) < 12:
                missing_depth.append(rel)

        has_c = False
        if need_cap:
            cp = resolve_caption_asset_path(cap_root, rel)
            has_c = cp is not None
            if has_c:
                n_cap += 1
            elif len(missing_cap) < 12:
                missing_cap.append(rel)

        depth_ok = not need_depth or has_d
        cap_ok = not need_cap or has_c
        if has_img and depth_ok and cap_ok:
            n_triple += 1

    def lz(msg: str) -> None:
        logzero.logger.info("[train][pairing] %s", msg)

    lz(f"CSV unique target images ({target_column!r}, conditioning={conditioning!r}): {n}")
    lz(f"Original image file exists (under canvas_image_root): {n_img} / {n}")
    if need_depth:
        lz(f"Depth map resolved (depth_image_root={depth_root}): {n_depth} / {n}")
        nd = sum(1 for _ in depth_root.rglob("*.png"))
        nj = sum(1 for _ in depth_root.rglob("*.jpg"))
        lz(f"Depth folder scan: {nd} *.png, {nj} *.jpg under {depth_root} (recursive)")
    elif depth_root is not None:
        lz(f"depth_image_root invalid — depth pairing skipped ({depth_root})")
    else:
        lz("depth_image_root unset — depth pairing not checked")

    if need_cap:
        lz(f"Caption .txt resolved (caption_dir={cap_root}): {n_cap} / {n}")
        nt = sum(1 for _ in cap_root.rglob("*.txt"))
        lz(f"Caption folder scan: {nt} *.txt under {cap_root} (recursive)")
    elif cap_root is not None:
        lz(f"caption_dir invalid — caption pairing skipped ({cap_root})")
    else:
        lz("caption_dir unset — caption pairing not checked")

    parts = ["image file on disk"]
    if need_depth:
        parts.append("depth")
    if need_cap:
        parts.append("caption")
    lz(f"Paired rows ({' + '.join(parts)}): {n_triple} / {n}")

    if missing_img:
        lz(f"example missing original image (up to 12): {missing_img}")
    if missing_depth and depth_root is not None:
        lz(f"example missing depth for CSV rel (up to 12): {missing_depth}")
    if missing_cap and cap_root is not None:
        lz(f"example missing caption .txt (up to 12): {missing_cap}")


def resize_cover_pil(im: Image.Image, out_w: int, out_h: int) -> Image.Image:
    """Scale so the image covers out_w×out_h, then center-crop to exact size (LANCZOS)."""
    im = im.convert("RGB")
    iw, ih = im.size
    if iw == out_w and ih == out_h:
        return im
    scale = max(out_w / iw, out_h / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = max(0, (nw - out_w) // 2)
    top = max(0, (nh - out_h) // 2)
    return im.crop((left, top, left + out_w, top + out_h))


def resize_contain_letterbox_pil(im: Image.Image, out_w: int, out_h: int, fill: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Scale uniformly so the image fits inside out_w×out_h, then center-pad to exact size (LANCZOS). No crop."""
    im = im.convert("RGB")
    iw, ih = im.size
    if iw == out_w and ih == out_h:
        return im
    scale = min(out_w / iw, out_h / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (out_w, out_h), fill)
    left = (out_w - nw) // 2
    top = (out_h - nh) // 2
    canvas.paste(im, (left, top))
    return canvas


def resize_unified_pil(im: Image.Image, out_w: int, out_h: int, mode: str) -> Image.Image:
    if mode == "cover":
        return resize_cover_pil(im, out_w, out_h)
    if mode == "contain":
        return resize_contain_letterbox_pil(im, out_w, out_h)
    raise ValueError(f"resize_unified_pil: unknown mode {mode!r} (use 'cover' or 'contain')")


def pil_to_model_tensor(pil: Image.Image) -> torch.Tensor:
    """RGB PIL → (3, H, W) in [-1, 1], matching legacy dataset normalization."""
    t = transforms.ToTensor()(pil)
    return transforms.Normalize([0.5], [0.5])(t)


def load_depth_image_as_rgb_pil(path: Path | str) -> Image.Image:
    """Load a depth PNG as RGB uint8 for ``pil_to_model_tensor``.

    Depth maps are often **16-bit grayscale** (``I;16``). Pillow ``convert('RGB')`` pushes most
    samples toward 255 (almost flat white); tensors then saturate and validation dumps read as white/black.

    Linear **uint16 → [0, 1]** scaling uses ``/ 65535`` so ``ToTensor`` + ``Normalize(0.5, 0.5)``
    matches ordinary RGB semantics. Same values are replicated across R/G/B channels.
    """
    p = Path(path)
    im = Image.open(p)
    im.load()
    mode = im.mode

    if mode == "RGB":
        return im
    if mode == "RGBA":
        return im.convert("RGB")

    if mode == "L":
        return Image.merge("RGB", (im, im, im))

    arr = np.asarray(im)

    if mode in ("I;16", "I;16L", "I;16B"):
        x = arr.astype(np.float32) * (1.0 / 65535.0)
    elif mode == "I":
        xf = arr.astype(np.float32)
        xmax = float(xf.max()) if xf.size else 1.0
        if xmax <= 65535.0:
            xf = xf * (1.0 / 65535.0)
        else:
            xf = xf / xmax if xmax > 0.0 else np.zeros_like(xf)
        x = xf
    elif mode == "F":
        xf = arr.astype(np.float32)
        mn = float(xf.min())
        mx = float(xf.max())
        x = (xf - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(xf)
    else:
        return im.convert("RGB")

    x = np.clip(x, 0.0, 1.0)
    u8 = np.round(x * 255.0).astype(np.uint8)
    rgb = np.stack([u8, u8, u8], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _clamp_odd_blur_kernel(k: int, w: int, h: int) -> int:
    """Largest odd kernel ≤ min(k, w, h), at least 3, or 0 if the map is too small to blur."""
    cap = int(min(w, h))
    if cap < 3:
        return 0
    mk = int(min(k, cap))
    if mk < 3:
        return 0
    if mk % 2 == 0:
        mk -= 1
    return mk if mk >= 3 else 0


def coarse_degrade_depth_rgb_pil(depth_rgb: Image.Image, args: Any) -> Image.Image:
    """Blur / noise / soft patch dropout in [0,1] space — encourages using canvas for fine appearance.

    Used when ``CanvasSceneDataset._depth_coarse_augment_active`` is true (training batches and, if enabled,
    validation / inference scripts that pass the same ``args`` fields).
    """
    w, h = depth_rgb.size
    arr = np.asarray(depth_rgb.convert("RGB"), dtype=np.float32) * (1.0 / 255.0)
    t = torch.from_numpy(arr).permute(2, 0, 1)

    p_blur = float(getattr(args, "canvas_depth_coarse_blur_prob", 0.7))
    if float(np.random.random()) < p_blur:
        kernels = tuple(getattr(args, "canvas_depth_coarse_blur_kernels", (11, 15, 21, 31, 41)))
        k = int(np.random.choice(kernels))
        k = _clamp_odd_blur_kernel(k, w, h)
        if k >= 3:
            sigma = max(0.15, float(k) / 6.0)
            t = TVF.gaussian_blur(t, kernel_size=[k, k], sigma=[sigma, sigma])

    p_noise = float(getattr(args, "canvas_depth_coarse_noise_prob", 0.4))
    if float(np.random.random()) < p_noise:
        lo = float(getattr(args, "canvas_depth_coarse_noise_std_lo", 0.03))
        hi = float(getattr(args, "canvas_depth_coarse_noise_std_hi", 0.10))
        scale = float(np.random.uniform(lo, hi))
        t = (t + torch.randn_like(t) * scale).clamp(0.0, 1.0)

    p_patch = float(getattr(args, "canvas_depth_coarse_patch_prob", 0.3))
    if float(np.random.random()) < p_patch:
        drop_p = float(getattr(args, "canvas_depth_coarse_patch_drop_strength", 0.3))
        m = torch.bernoulli(torch.full((1, 1, h, w), 1.0 - drop_p, dtype=t.dtype))
        mk = _clamp_odd_blur_kernel(21, w, h)
        if mk >= 3:
            sig = max(1.0, float(mk) / 6.0)
            m = TVF.gaussian_blur(m, kernel_size=[mk, mk], sigma=[sig, sig])
        t = t * m.squeeze(0)

    u8 = (t.clamp(0.0, 1.0).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy())
    return Image.fromarray(np.asarray(u8), mode="RGB")


def _make_subject_transform(cond_size: int, *, random_geom_aug: bool) -> transforms.Compose:
    """Resize (long side → cond_size, /16), optional flip/small rotate, square pad, [-1,1] tensor."""
    steps: List[Any] = [
        transforms.Lambda(
            lambda img: img.resize(
                (
                    multiple_16(cond_size * img.size[0] / max(img.size)),
                    multiple_16(cond_size * img.size[1] / max(img.size)),
                ),
                resample=Image.BILINEAR,
            )
        ),
    ]
    if random_geom_aug:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
        ]
    steps += [
        transforms.Lambda(
            lambda img: transforms.Pad(
                padding=(
                    int((cond_size - img.size[0]) / 2),
                    int((cond_size - img.size[1]) / 2),
                    int((cond_size - img.size[0]) / 2),
                    int((cond_size - img.size[1]) / 2),
                ),
                fill=0,
            )(img)
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    return transforms.Compose(steps)


class CanvasSceneDataset(torch.utils.data.Dataset):
    """
    Two conditioning strategies (``args.canvas_conditioning``):

    - ``precomputed``: one sample per CSV row. Target = ``canvas_target_column`` image;
      optional ``canvas_column`` / ``subject_path`` file for subject pixels.

    - ``bbox_multiview``: one sample per distinct scene image (grouped by ``canvas_target_column``).
      **Unified mode:** the source frame is **center-cropped** to ``(unified_train_width, unified_train_height)``
      (cover / no letterbox bars). CSV boxes are mapped into that viewport, then bbox **crops** are augmented
      (brightness, contrast, flip, rotation, shear) before pasting onto the conditioning canvas.
      **Target RGB and depth** use the same center crop **without** geometric augmentations.

    **Legacy unified contain / random resolution:** unchanged older paths still use ``canvas_unified_resize`` where
    applicable.

    Training noise is applied in train.py to **main** latents only; canvas/depth latents stay clean encoder inputs.

    """

    _missing_depth_warning_emitted = False

    def __init__(
        self,
        csv_path: str,
        args: Any,
        canvas_column: str = "canvas_path",
        target_column: str = "image_path",
        prompt_column: str = "prompt",
        *,
        asset_image_root: Optional[str] = None,
        asset_depth_root: Optional[str] = None,
        asset_caption_root: Optional[str] = None,
        asset_multiview_root: Optional[str] = None,
    ):
        super().__init__()
        self.args = args
        self.canvas_column = canvas_column
        self.target_column = target_column
        self.prompt_column = prompt_column
        img_src = (
            asset_image_root
            if asset_image_root is not None
            else getattr(args, "canvas_image_root", None)
        )
        self._image_root = img_src.strip() if isinstance(img_src, str) and img_src.strip() else None
        self.conditioning = getattr(args, "canvas_conditioning", "precomputed")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            flat_rows = [{k.strip(): (v or "").strip() for k, v in row.items()} for row in reader]

        size = args.cond_size
        self._noise_cap = args.noise_size
        self._random_target_res = bool(getattr(args, "canvas_random_target_resolution", False))
        uw = int(getattr(args, "unified_train_width", 1920))
        uh = int(getattr(args, "unified_train_height", 1080))
        self._unified_wh: Optional[Tuple[int, int]] = None
        if not self._random_target_res:
            self._unified_wh = (multiple_16(uw), multiple_16(uh))
        mode = getattr(args, "canvas_unified_resize", "contain")
        if mode not in ("cover", "contain"):
            raise ValueError(f"canvas_unified_resize must be 'cover' or 'contain', got {mode!r}")
        self._unified_resize_mode: str = mode
        # Precomputed canvas path: geom aug only on bbox crops elsewhere; square letterbox stays layout-only here.
        self.subject_transform = _make_subject_transform(size, random_geom_aug=False)
        # Composed bbox canvas: preserve layout (only scale + letterbox to cond_size for VAE).
        self.subject_transform_layout_preserving = _make_subject_transform(size, random_geom_aug=False)

        if self.conditioning == "bbox_multiview":
            mv_raw = (
                asset_multiview_root
                if asset_multiview_root is not None
                else (getattr(args, "canvas_multiview_dir", "") or "")
            )
            self._multiview_root = Path(mv_raw).resolve() if isinstance(mv_raw, str) and mv_raw.strip() else None
            if self._multiview_root is not None and not self._multiview_root.is_dir():
                self._multiview_root = None
            self._multiview_prob = float(getattr(args, "canvas_multiview_prob", 0.5))
            self._canvas_background = getattr(args, "canvas_background", "black")
            self._bbox_min_side = float(getattr(args, "canvas_bbox_min_side", 0.0))
            self._mv_match_min = float(getattr(args, "canvas_multiview_match_min_side", 200.0))
            if self._unified_wh is not None:
                self._compose_canvas_wh = self._unified_wh
            else:
                self._compose_canvas_wh = (CANVAS_W, CANVAS_H)
            self._canvas_augment_enabled = True
            dr = (
                asset_depth_root
                if asset_depth_root is not None
                else (getattr(args, "depth_image_root", None) or "")
            )
            self._depth_root: Optional[Path] = Path(dr).resolve() if isinstance(dr, str) and dr.strip() else None
            if self._depth_root is not None and not self._depth_root.is_dir():
                raise ValueError(f"depth_image_root is not a directory: {self._depth_root}")
            self.groups: List[Tuple[str, List[Dict[str, Any]]]] = prepare_groups_from_flat_rows(
                flat_rows,
                target_column,
                self._bbox_min_side,
                self._mv_match_min,
            )
            if not self.groups:
                raise ValueError(
                    "canvas_conditioning=bbox_multiview: no valid image groups (need image_path + x1,y1,x2,y2)."
                )
            self.rows = []  # unused
        else:
            self.groups = []
            self.rows = flat_rows
            self._multiview_root = None
            self._compose_canvas_wh = (CANVAS_W, CANVAS_H)
            self._canvas_augment_enabled = True
            dr = (
                asset_depth_root
                if asset_depth_root is not None
                else (getattr(args, "depth_image_root", None) or "")
            )
            self._depth_root = Path(dr).resolve() if isinstance(dr, str) and dr.strip() else None
            if self._depth_root is not None and not self._depth_root.is_dir():
                raise ValueError(f"depth_image_root is not a directory: {self._depth_root}")

        if self._depth_root is not None and self._random_target_res:
            raise ValueError(
                "depth_image_root is set: use unified training resolution (do not pass --canvas_random_target_resolution)."
            )

        self._depth_keep_prob = float(getattr(args, "depth_keep_prob", 1.0))
        if not (0.0 <= self._depth_keep_prob <= 1.0):
            raise ValueError(f"depth_keep_prob must be in [0, 1], got {self._depth_keep_prob}")

        self._canvas_keep_prob = float(getattr(args, "canvas_keep_prob", 1.0))
        if not (0.0 <= self._canvas_keep_prob <= 1.0):
            raise ValueError(f"canvas_keep_prob must be in [0, 1], got {self._canvas_keep_prob}")

        self._depth_coarse_augment = bool(getattr(args, "canvas_depth_coarse_augment", True))

        cap = asset_caption_root if asset_caption_root is not None else getattr(args, "caption_dir", None)
        self._caption_root: Optional[Path] = None
        if isinstance(cap, str) and cap.strip():
            self._caption_root = Path(cap).resolve()
            if not self._caption_root.is_dir():
                raise ValueError(f"caption_dir is not a directory: {self._caption_root}")

    def __len__(self):
        if self.conditioning == "bbox_multiview":
            return len(self.groups)
        return len(self.rows)

    def _target_transform(self, image: Image.Image, noise_size: int):
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
        if self.conditioning == "bbox_multiview":
            return self._getitem_bbox_multiview(idx)
        return self._getitem_precomputed(idx)

    def _canvas_augment_kwargs(self) -> Dict[str, Any]:
        a = self.args
        augment = bool(
            getattr(self, "_canvas_augment_enabled", True) and getattr(a, "canvas_augment", True)
        )
        # Scene-wide brightness changes canvas vs clean target crop — default off (center-crop targets stay reference).
        gbp = float(getattr(a, "canvas_augment_global_brightness_prob", 0.0))
        if not augment:
            gbp = 0.0
        return {
            "augment": augment,
            "bbox_margin_prob": float(getattr(a, "canvas_augment_bbox_margin_prob", 1.0)),
            "bbox_margin_min": int(getattr(a, "canvas_augment_bbox_margin_min", 10)),
            "bbox_margin_max": int(getattr(a, "canvas_augment_bbox_margin_max", 30)),
            "bbox_expand_prob": float(getattr(a, "canvas_augment_bbox_expand_prob", 0.5)),
            "brightness_prob": float(getattr(a, "canvas_augment_brightness_prob", 0.88)),
            "brightness_bimodal": bool(getattr(a, "canvas_augment_brightness_bimodal", True)),
            "brightness_min": float(getattr(a, "canvas_augment_brightness_min", 0.45)),
            "brightness_max": float(getattr(a, "canvas_augment_brightness_max", 1.85)),
            "brightness_dim_min": float(getattr(a, "canvas_augment_brightness_dim_min", 0.32)),
            "brightness_dim_max": float(getattr(a, "canvas_augment_brightness_dim_max", 0.58)),
            "brightness_lit_min": float(getattr(a, "canvas_augment_brightness_lit_min", 1.42)),
            "brightness_lit_max": float(getattr(a, "canvas_augment_brightness_lit_max", 2.08)),
            "global_brightness_prob": gbp,
            "contrast_prob": float(getattr(a, "canvas_augment_contrast_prob", 0.55)),
            "contrast_min": float(getattr(a, "canvas_augment_contrast_min", 0.78)),
            "contrast_max": float(getattr(a, "canvas_augment_contrast_max", 1.28)),
            "crop_flip_prob": float(getattr(a, "canvas_augment_crop_flip_prob", 0.5)),
            "crop_rotate_prob": float(getattr(a, "canvas_augment_crop_rotate_prob", 0.55)),
            "crop_rotate_deg_mild": float(getattr(a, "canvas_augment_crop_rotate_deg_mild", 8.0)),
            "crop_rotate_deg_extreme": float(getattr(a, "canvas_augment_crop_rotate_deg_extreme", 14.0)),
            "shear_prob": float(getattr(a, "canvas_augment_shear_prob", 0.45)),
            "shear_deg_mild": float(getattr(a, "canvas_augment_shear_deg_mild", 10.0)),
            "shear_deg_extreme": float(getattr(a, "canvas_augment_shear_deg_extreme", 18.0)),
        }

    def _read_caption_for_rel(self, rel: str, fallback: str) -> str:
        """Prefer ``caption_dir/<stem>.txt`` (optional nested layout)."""
        if self._caption_root is None:
            return fallback
        stem = Path(rel).stem
        candidates = [
            self._caption_root / f"{stem}.txt",
            self._caption_root / Path(rel).with_suffix(".txt").name,
        ]
        parent = Path(rel).parent
        if str(parent) not in (".", ""):
            candidates.append(self._caption_root / parent / f"{stem}.txt")
        for p in candidates:
            try:
                if p.is_file():
                    return p.read_text(encoding="utf-8").strip()
            except OSError:
                continue
        return fallback

    def _resolve_depth_path(self, rel: str) -> Optional[Path]:
        if self._depth_root is None:
            return None
        return resolve_depth_asset_path(self._depth_root, rel)

    def _load_depth_crop_pil_with_flag(self, rel: str, tw: int, th: int) -> Optional[Tuple[Image.Image, bool]]:
        """Load depth, center-crop to ``tw×th``. Returns ``(pil, from_disk)`` or ``None`` if depth stream disabled."""
        if self._depth_root is None:
            return None
        p = self._resolve_depth_path(rel)
        if p is None:
            if not CanvasSceneDataset._missing_depth_warning_emitted:
                CanvasSceneDataset._missing_depth_warning_emitted = True
                warnings.warn(
                    f"No depth image resolved for image_path={rel!r} under depth_image_root={self._depth_root}; "
                    f"tried stem+.png first (jpg target → png depth), then nested paths and CSV basename — "
                    f"using a black placeholder.",
                    stacklevel=2,
                )
            return (Image.new("RGB", (tw, th), (0, 0, 0)), False)
        depth = load_depth_image_as_rgb_pil(p)
        return (resize_cover_pil(depth, tw, th), True)

    def _depth_coarse_augment_active(self) -> bool:
        """Depth degradation: on for normal training batches; also on in validation when canvas aug is off if
        ``args.validation_depth_coarse_augment`` (default True) so depth matches inference/stage3."""
        if not self._depth_coarse_augment:
            return False
        if not getattr(self.args, "canvas_augment", True):
            return False
        if getattr(self, "_canvas_augment_enabled", True):
            return True
        return bool(getattr(self.args, "validation_depth_coarse_augment", True))

    def _prepare_depth_for_sample(self, rel: str, tw: int, th: int) -> Optional[Image.Image]:
        """Load depth → optional coarse degradation → ``_depth_keep_prob`` full dropout."""
        pack = self._load_depth_crop_pil_with_flag(rel, tw, th)
        if pack is None:
            return None
        depth_pil, from_disk = pack
        if from_disk and self._depth_coarse_augment_active():
            depth_pil = coarse_degrade_depth_rgb_pil(depth_pil, self.args)
        return self._finalize_depth_cond(depth_pil, tw, th)

    def _finalize_depth_cond(self, depth_pil: Image.Image, tw: int, th: int) -> Image.Image:
        """Stochastic depth: keep real map with ``_depth_keep_prob``, else black RGB (same size as target crop)."""
        if self._depth_root is None:
            return depth_pil
        kp = float(self._depth_keep_prob)
        if kp >= 1.0:
            return depth_pil
        if kp <= 0.0:
            return Image.new("RGB", (tw, th), (0, 0, 0))
        if float(np.random.random()) < kp:
            return depth_pil
        return Image.new("RGB", (tw, th), (0, 0, 0))

    def _finalize_subject_pixel_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Stochastic canvas: keep real conditioning with ``_canvas_keep_prob``, else black (-1 in normalized space)."""
        kp = float(self._canvas_keep_prob)
        if kp >= 1.0:
            return t
        if kp <= 0.0:
            return torch.full_like(t, -1.0)
        if float(np.random.random()) < kp:
            return t
        return torch.full_like(t, -1.0)

    def _getitem_bbox_multiview(self, idx: int) -> Dict[str, Any]:
        rel, group_rows = self.groups[idx]
        full = load_image_safely(rel, self.args.cond_size, root_dir=self._image_root)

        cw, ch = self._compose_canvas_wh
        geom_extreme = int(getattr(self.args, "canvas_geom_extreme_max_crops", 3))

        depth_u: Optional[Image.Image] = None
        if self._unified_wh is not None:
            tw, th = self._unified_wh
            iw, ih = full.size
            cropped_full = resize_cover_pil(full.convert("RGB"), tw, th)
            mapped_rows = map_rows_xyxy_to_cover_crop_space(group_rows, iw, ih, tw, th)

            canvas_pil = build_canvas_for_image_rel(
                rel,
                cropped_full,
                mapped_rows,
                self._multiview_root,
                self._multiview_prob,
                self._canvas_background,
                getattr(self.args, "seed", None),
                self._bbox_min_side,
                tw,
                th,
                extreme_geom_max_crops=geom_extreme,
                canvas_layout="contain",
                **self._canvas_augment_kwargs(),
            )

            pixel_values = pil_to_model_tensor(cropped_full)
            subject_pixel_values = self._finalize_subject_pixel_tensor(pil_to_model_tensor(canvas_pil))
            depth_u = self._prepare_depth_for_sample(rel, tw, th)
        else:
            canvas_pil = build_canvas_for_image_rel(
                rel,
                full,
                group_rows,
                self._multiview_root,
                self._multiview_prob,
                self._canvas_background,
                getattr(self.args, "seed", None),
                self._bbox_min_side,
                cw,
                ch,
                extreme_geom_max_crops=geom_extreme,
                canvas_layout=self._unified_resize_mode,
                **self._canvas_augment_kwargs(),
            )
            noise_size = get_random_resolution(max_size=self.args.noise_size)
            pixel_values = self._target_transform(full, noise_size)
            subject_pixel_values = self._finalize_subject_pixel_tensor(
                self.subject_transform_layout_preserving(canvas_pil)
            )

        base_prompt = prompt_for_group(group_rows, self.prompt_column)
        prompt = self._read_caption_for_rel(rel, base_prompt)

        out: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "prompts": prompt,
            "subject_pixel_values": subject_pixel_values,
        }
        if self._unified_wh is not None and depth_u is not None:
            out["cond_pixel_values"] = pil_to_model_tensor(depth_u)

        return out

    def _getitem_precomputed(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        rel = row[self.target_column]
        target = load_image_safely(rel, self.args.cond_size, root_dir=self._image_root)

        depth_u_for_cond: Optional[Image.Image] = None
        if self._unified_wh is not None:
            tw, th = self._unified_wh
            tgt_u = resize_cover_pil(target.convert("RGB"), tw, th)
            depth_u_for_cond = self._prepare_depth_for_sample(rel, tw, th)
            pixel_values = pil_to_model_tensor(tgt_u)
        else:
            noise_size = get_random_resolution(max_size=self.args.noise_size)
            pixel_values = self._target_transform(target, noise_size)

        base_prompt = row.get(self.prompt_column, "") or ""
        prompt = self._read_caption_for_rel(rel, base_prompt)

        out: Dict[str, Any] = {"pixel_values": pixel_values, "prompts": prompt}

        canvas_key = self.canvas_column if self.canvas_column in row else "subject_path"
        if canvas_key in row and row[canvas_key]:
            canvas = load_image_safely(row[canvas_key], self.args.cond_size, root_dir=self._image_root)
            if self._unified_wh is not None:
                tw, th = self._unified_wh
                out["subject_pixel_values"] = self._finalize_subject_pixel_tensor(
                    pil_to_model_tensor(resize_cover_pil(canvas.convert("RGB"), tw, th))
                )
            else:
                out["subject_pixel_values"] = self._finalize_subject_pixel_tensor(self.subject_transform(canvas))

        if depth_u_for_cond is not None:
            out["cond_pixel_values"] = pil_to_model_tensor(depth_u_for_cond)

        for k in ("x1", "y1", "x2", "y2", "confidence"):
            if k in row and row[k] != "":
                try:
                    out[k] = float(row[k])
                except ValueError:
                    out[k] = row[k]

        return out


def make_canvas_train_dataset(args, accelerator=None):
    roots = parse_canvas_data_roots(getattr(args, "canvas_data_roots", None) or "")
    if roots:
        pieces: List[CanvasSceneDataset] = []
        for root_s in roots:
            root_p = Path(root_s).expanduser().resolve()
            paths = resolve_canvas_data_paths(root_p)
            if not paths["csv"].is_file():
                raise FileNotFoundError(
                    f"canvas_data_roots entry {root_p}: expected CSV at {paths['csv']}"
                )
            pieces.append(
                CanvasSceneDataset(
                    str(paths["csv"]),
                    args,
                    canvas_column=args.canvas_column,
                    target_column=getattr(args, "canvas_target_column", "image_path"),
                    prompt_column=getattr(args, "canvas_prompt_column", "prompt"),
                    asset_image_root=str(paths["images"]),
                    asset_depth_root=str(paths["depth"]),
                    asset_caption_root=str(paths["captions"]),
                    asset_multiview_root=str(paths["multiview"]),
                )
            )
        if len(pieces) == 1:
            return pieces[0]
        return CanvasConcatDataset(pieces)

    return CanvasSceneDataset(
        args.csv_path,
        args,
        canvas_column=args.canvas_column,
        target_column=getattr(args, "canvas_target_column", "image_path"),
        prompt_column=getattr(args, "canvas_prompt_column", "prompt"),
    )


def collate_fn_canvas(examples: List[Dict[str, Any]]):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples]).float()
    prompts = [ex["prompts"] for ex in examples]
    batch: Dict[str, Any] = {"pixel_values": pixel_values, "prompts": prompts}
    if examples[0].get("subject_pixel_values") is not None:
        batch["subject_pixel_values"] = torch.stack([ex["subject_pixel_values"] for ex in examples]).float()
    if examples[0].get("cond_pixel_values") is not None:
        batch["cond_pixel_values"] = torch.stack([ex["cond_pixel_values"] for ex in examples]).float()
    return batch

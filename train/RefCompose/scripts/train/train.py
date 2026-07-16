import argparse
import copy
import logging
import logzero
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
import re
from safetensors.torch import save_file

import numpy as np
import torch
import torch.utils.checkpoint
from torch.utils.data import WeightedRandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from tqdm.auto import tqdm
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor

import diffusers

from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel as Flux2Transformer2DModelBase
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import check_min_version

from src.prompt_helper import encode_prompts_flux2
from src.transformer_flux import FluxTransformer2DModel
from src.jsonl_datasets import make_train_dataset, collate_fn, multiple_16
from src.canvas_dataset import (
    CANVAS_DATA_ROOT_CSV,
    balanced_sampling_weights_for_concat,
    collate_fn_canvas,
    concat_dataset_child_lengths,
    log_canvas_asset_pairing_report,
    make_canvas_train_dataset,
    parse_canvas_data_roots,
)
from src.manifest_dataset import (
    log_manifest_asset_pairing_report,
    make_manifest_train_dataset,
)
from src.layers_flux2 import MultiDoubleStreamBlockFlux2LoraProcessor, MultiSingleStreamBlockFlux2LoraProcessor
from src.flux2_train_helpers import encode_flux2_latents, prepare_subject_latent_ids, unpack_main_latents

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0")

logger = get_logger(__name__)


def _log_train(msg: str) -> None:
    """Human-readable progress on stdout via logzero (all ranks — prefix with rank in _train_lz when needed)."""
    logzero.logger.info("[train] %s", msg)


def _train_lz(accelerator: Accelerator, msg: str) -> None:
    logzero.logger.info("[RANK %s] [train] %s", accelerator.process_index, msg)


def _tracker_config_from_args(args: argparse.Namespace) -> dict:
    """TensorBoard add_hparams only allows int, float, str, bool, or torch.Tensor."""
    out = {}
    for k, v in vars(args).items():
        if isinstance(v, torch.Tensor):
            out[k] = v
        elif isinstance(v, bool):
            out[k] = v
        elif isinstance(v, int):
            out[k] = v
        elif isinstance(v, float):
            out[k] = v
        elif isinstance(v, str):
            out[k] = v
        elif v is None:
            out[k] = ""
        else:
            out[k] = str(v)
    return out


def log_validation(
        pipeline,
        args,
        accelerator,
        pipeline_args,
        step,
        torch_dtype,
        is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    autocast_ctx = nullcontext()

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, step, dataformats="NHWC")

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--lora_num", type=int, default=1, help="number of the lora.")
    parser.add_argument("--cond_size", type=int, default=512, help="size of the condition data.")
    parser.add_argument("--noise_size", type=int, default=1280, help="max side of the training data.")
    parser.add_argument("--test_h", type=int, default=1024, help="max side of the training data.")
    parser.add_argument("--test_w", type=int, default=1024, help="max side of the training data.")
    parser.add_argument("--mode",type=str,default=None,help="The mode of the controller. Choose between ['depth', 'pose', 'canny'].")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--spatial_column",
        type=str,
        default="None",
        help="The column of the dataset containing the canny image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--subject_column",
        type=str,
        default="image",
        help="The column of the dataset containing the subject image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption_left,caption_right",
        help="The column of the dataset containing the instance prompt for each image",
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length for Mistral / Pixtral chat template tokenization (Flux2).",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="jsonl",
        choices=["jsonl", "canvas", "manifest"],
        help=(
            "jsonl: HuggingFace JSON dataset; canvas: CSV-driven CanvasSceneDataset; "
            "manifest: v2 dataset_manifest.jsonl with gt_image, canvas_image, depth_image, caption."
        ),
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="",
        help=(
            "When dataset_type=manifest: path to dataset_manifest.jsonl "
            "(falls back to --train_data_dir if unset)."
        ),
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help="Used when dataset_type=canvas: CSV with columns image_path, prompt (and optional bbox columns). "
        "Ignored when --canvas_data_roots is non-empty (paths are derived per root).",
    )
    parser.add_argument(
        "--canvas_data_roots",
        type=str,
        default="",
        help=(
            "When dataset_type=canvas and non-empty: comma/semicolon/newline-separated list of dataset **output** "
            "directories. Each must contain bbox_results/yolo26_detections.csv, images/, depth/, image_captions/, "
            "and multiview_out/ (same layout as dataset_prep output). Builds one CanvasSceneDataset per root and "
            "concatenates them. With multiple roots, default training uses a balanced sampler (equal P per root); "
            "see --canvas_balanced_concat_sampling."
        ),
    )
    parser.add_argument(
        "--canvas_balanced_concat_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When the canvas dataset is a ConcatDataset from multiple --canvas_data_roots: use "
            "WeightedRandomSampler so each root is drawn with equal probability (e.g. half from each of two roots) "
            "even if CSV lengths differ (5k vs 50k). Single-process training only; with DDP (num_processes>1) "
            "this is skipped and shuffle=True is used. --no-canvas_balanced_concat_sampling restores "
            "length-proportional sampling (shuffle)."
        ),
    )
    parser.add_argument(
        "--canvas_column",
        type=str,
        default="canvas_path",
        help="CSV column for canvas / subject composite image (canvas dataset).",
    )
    parser.add_argument(
        "--canvas_target_column",
        type=str,
        default="image_path",
        help="CSV column for target scene image path.",
    )
    parser.add_argument(
        "--canvas_prompt_column",
        type=str,
        default="prompt",
        help="CSV column for text prompt.",
    )
    parser.add_argument(
        "--canvas_image_root",
        type=str,
        default="",
        help=(
            "When dataset_type=canvas: directory prepended to relative image_path / canvas_path values from the CSV."
        ),
    )
    parser.add_argument(
        "--depth_image_root",
        type=str,
        default="",
        help=(
            "When dataset_type=canvas and non-empty: depth PNG directory (basename matches target image_path). "
            "Encoded as cond latents after canvas; train bumps --lora_num to 2 when unset below 2."
        ),
    )
    parser.add_argument(
        "--depth_keep_prob",
        type=float,
        default=0.5,
        help=(
            "When depth_image_root is set: per-sample probability to use the real depth map for cond_pixel_values; "
            "otherwise a black RGB image of the same crop size (depth LoRA stream still receives a blank block). "
            "Use 1.0 to always keep real depth. Dataloader validation forces 1.0 while sampling for clearer PNGs."
        ),
    )
    parser.add_argument(
        "--canvas_keep_prob",
        type=float,
        default=0.5,
        help=(
            "Per-sample probability to keep the real conditioning canvas in subject_pixel_values; otherwise replace "
            "with a black canvas in model space (same tensor shape, values -1). Use 1.0 to disable. "
            "Dataloader validation forces 1.0 while sampling for clearer PNGs."
        ),
    )
    parser.add_argument(
        "--canvas_depth_coarse_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When dataset uses depth: during training only, randomly degrade kept depth (large Gaussian blur, "
            "additive noise, smoothed patch dropout) so fine appearance must come from the canvas; full-depth "
            "dropout remains controlled by --depth_keep_prob. Off when validation disables canvas aug or when "
            "--no-canvas_augment."
        ),
    )
    parser.add_argument(
        "--canvas_depth_coarse_blur_prob",
        type=float,
        default=0.7,
        help="Training-only: probability to apply heavy Gaussian blur to depth (per sample).",
    )
    parser.add_argument(
        "--canvas_depth_coarse_noise_prob",
        type=float,
        default=0.4,
        help="Training-only: probability to add Gaussian noise to depth in [0,1] space (per sample).",
    )
    parser.add_argument(
        "--canvas_depth_coarse_patch_prob",
        type=float,
        default=0.3,
        help="Training-only: probability for smoothed multiplicative patch dropout on depth (per sample).",
    )
    parser.add_argument(
        "--caption_dir",
        type=str,
        default="/mnt/data0/teja/research_multiref/dataset_preparation/output/image_captions",
        help=(
            "When dataset_type=canvas: directory of .txt files (filename stem matches target image_path stem). "
            "Overrides CSV text for that row. Set to empty string to disable and use CSV / class_name only."
        ),
    )
    parser.add_argument(
        "--prompt_warmup_steps",
        type=int,
        default=3000,
        help=(
            "For the first N global steps (at loop start, before the forward), use an empty string prompt; "
            "from step N onward, use captions from --caption_dir (or CSV). Same rule for dataloader validation."
        ),
    )
    parser.add_argument(
        "--canvas_target_align_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="canvas unified mode: paired flip + small rotation on target RGB and depth.",
    )
    parser.add_argument(
        "--canvas_target_align_flip_p",
        type=float,
        default=0.5,
        help="Probability of paired horizontal flip on target+depth.",
    )
    parser.add_argument(
        "--canvas_target_align_rot_deg",
        type=float,
        default=8.0,
        help="Uniform ±degrees rotation on target+depth (paired).",
    )
    parser.add_argument(
        "--canvas_conditioning",
        type=str,
        default="precomputed",
        choices=["precomputed", "bbox_multiview"],
        help=(
            "precomputed: one row per sample, optional canvas_path file. "
            "bbox_multiview: one sample per scene image; build black canvas + bbox crops (stage4-style), optional multiview."
        ),
    )
    parser.add_argument(
        "--canvas_bbox_min_side",
        type=float,
        default=0.0,
        help="bbox_multiview: drop detections whose box w or h is <= this (0 keeps all positive-area boxes).",
    )
    parser.add_argument(
        "--canvas_multiview_match_min_side",
        type=float,
        default=200.0,
        help="bbox_multiview: only boxes larger than this get multiview folder indices (same as stage4).",
    )
    parser.add_argument(
        "--canvas_multiview_dir",
        type=str,
        default="",
        help="bbox_multiview: stage3 root (stem/crop_no/*.png); empty disables multiview.",
    )
    parser.add_argument(
        "--canvas_multiview_prob",
        type=float,
        default=0.5,
        help="bbox_multiview: when multiview PNGs exist, probability to use one vs source crop.",
    )
    parser.add_argument(
        "--canvas_background",
        type=str,
        default="black",
        choices=["black", "scaled"],
        help="bbox_multiview: canvas base (black, or full frame letterboxed under crops — same layout as bbox mapping).",
    )
    parser.add_argument(
        "--canvas_geom_extreme_max_crops",
        type=int,
        default=3,
        help=(
            "bbox_multiview: per image, at most this many bbox crops get stronger rotation/shear bands; "
            "others use mild geometry. 0 = all crops use mild crop aug."
        ),
    )
    parser.add_argument(
        "--canvas_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "bbox_multiview: augmentations apply to **bbox crop patches** only (brightness, contrast, flip, rotation, "
            "shear); target RGB and depth stay **center-cropped without geom aug**. "
            "Validation snapshots follow training augments by default (--validation_canvas_augment)."
        ),
    )
    parser.add_argument(
        "--canvas_augment_bbox_margin_prob",
        type=float,
        default=1.0,
        help="bbox_multiview: when canvas_augment, probability per box to apply margin expand/shrink.",
    )
    parser.add_argument(
        "--canvas_augment_bbox_margin_min",
        type=int,
        default=10,
        help="bbox_multiview: margin min (pixels in source image space).",
    )
    parser.add_argument(
        "--canvas_augment_bbox_margin_max",
        type=int,
        default=30,
        help="bbox_multiview: margin max (pixels in source image space).",
    )
    parser.add_argument(
        "--canvas_augment_bbox_expand_prob",
        type=float,
        default=0.5,
        help="bbox_multiview: probability to expand vs shrink the box (when margin is applied).",
    )
    parser.add_argument(
        "--canvas_augment_brightness_prob",
        type=float,
        default=0.88,
        help="bbox_multiview: per-crop probability for brightness jitter (bimodal dim vs lit by default).",
    )
    parser.add_argument(
        "--canvas_augment_brightness_bimodal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "bbox_multiview: sample brightness in two bands (dim vs heavily lit). If disabled, uniform in "
            "[--canvas_augment_brightness_min, --canvas_augment_brightness_max]."
        ),
    )
    parser.add_argument(
        "--canvas_augment_brightness_min",
        type=float,
        default=0.45,
        help="bbox_multiview: brightness lower bound when bimodal is off (uniform mode).",
    )
    parser.add_argument(
        "--canvas_augment_brightness_max",
        type=float,
        default=1.85,
        help="bbox_multiview: brightness upper bound when bimodal is off (uniform mode).",
    )
    parser.add_argument(
        "--canvas_augment_brightness_dim_min",
        type=float,
        default=0.32,
        help="bbox_multiview: bimodal — very dim crops/scene lower factor (PIL ImageEnhance.Brightness).",
    )
    parser.add_argument(
        "--canvas_augment_brightness_dim_max",
        type=float,
        default=0.58,
        help="bbox_multiview: bimodal — very dim upper factor.",
    )
    parser.add_argument(
        "--canvas_augment_brightness_lit_min",
        type=float,
        default=1.42,
        help="bbox_multiview: bimodal — heavily lit lower factor.",
    )
    parser.add_argument(
        "--canvas_augment_brightness_lit_max",
        type=float,
        default=2.08,
        help="bbox_multiview: bimodal — heavily lit upper factor.",
    )
    parser.add_argument(
        "--canvas_augment_global_brightness_prob",
        type=float,
        default=0.0,
        help=(
            "bbox_multiview: probability of global brightness on the **composed canvas image** before pasting crops "
            "(default 0: unified target stays the clean center crop; enable only if you want matched scene tint)."
        ),
    )
    parser.add_argument(
        "--canvas_augment_perspective_prob",
        type=float,
        default=0.48,
        help="Unused (perspective warp removed from crop aug). Ignored.",
    )
    parser.add_argument(
        "--canvas_augment_perspective_distortion",
        type=float,
        default=0.62,
        help="Unused. Ignored.",
    )
    parser.add_argument(
        "--canvas_augment_contrast_prob",
        type=float,
        default=0.55,
        help="bbox_multiview: per-crop probability for PIL contrast jitter.",
    )
    parser.add_argument(
        "--canvas_augment_contrast_min",
        type=float,
        default=0.78,
        help="bbox_multiview: contrast enhance lower bound.",
    )
    parser.add_argument(
        "--canvas_augment_contrast_max",
        type=float,
        default=1.28,
        help="bbox_multiview: contrast enhance upper bound.",
    )
    parser.add_argument(
        "--canvas_augment_crop_flip_prob",
        type=float,
        default=0.5,
        help="bbox_multiview: horizontal flip probability on each crop tensor.",
    )
    parser.add_argument(
        "--canvas_augment_crop_rotate_prob",
        type=float,
        default=0.55,
        help="bbox_multiview: probability to apply planar rotation on each crop.",
    )
    parser.add_argument(
        "--canvas_augment_crop_rotate_deg_mild",
        type=float,
        default=8.0,
        help="bbox_multiview: ±degrees rotation on non-extreme crops.",
    )
    parser.add_argument(
        "--canvas_augment_crop_rotate_deg_extreme",
        type=float,
        default=14.0,
        help="bbox_multiview: ±degrees rotation on extreme_geom crops.",
    )
    parser.add_argument(
        "--canvas_augment_shear_prob",
        type=float,
        default=0.45,
        help="bbox_multiview: per-crop probability for affine shear (both axes).",
    )
    parser.add_argument(
        "--canvas_augment_shear_deg_mild",
        type=float,
        default=10.0,
        help="bbox_multiview: max shear degrees per axis (mild crops).",
    )
    parser.add_argument(
        "--canvas_augment_shear_deg_extreme",
        type=float,
        default=18.0,
        help="bbox_multiview: max shear degrees per axis (extreme_geom crops).",
    )
    parser.add_argument(
        "--unified_train_width",
        type=int,
        default=1920,
        help="canvas dataset: fixed width when not using --canvas_random_target_resolution. Snapped to multiple of 16 in train.py.",
    )
    parser.add_argument(
        "--unified_train_height",
        type=int,
        default=1080,
        help=(
            "canvas dataset: nominal height (default 1080p). Snapped to a multiple of 16 in train.py (1080→1088). "
            "With default --canvas_unified_resize contain, black bars pad that gap so 16:9 content stays uncropped."
        ),
    )
    parser.add_argument(
        "--canvas_unified_resize",
        type=str,
        default="contain",
        choices=["cover", "contain"],
        help=(
            "canvas unified mode: 'contain' (default) = scale to fit + black letterbox — same box for target and "
            "bbox canvas (1920×1080 composite → 1920×1088 tensor with thin bars). "
            "'cover' = scale to fill then center-crop (can cut edges)."
        ),
    )
    parser.add_argument(
        "--canvas_random_target_resolution",
        action="store_true",
        help="canvas dataset: random long-side resize for target (noise_size) and cond_size² letterbox for canvas; use batch size 1 unless all targets share size.",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip Flux2 validation (dataloader metrics + samples, or legacy pipeline) during training.",
    )
    parser.add_argument(
        "--validation_from_dataloader",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When True (default), every validation step samples a batch from the train dataset, logs "
            "validation/loss to TensorBoard, saves target/canvas/generated PNGs under output_dir, and "
            "writes a composite image to TensorBoard. Use --no-validation_from_dataloader for legacy "
            "text-only Flux2Pipeline validation."
        ),
    )
    parser.add_argument(
        "--validation_num_samples",
        type=int,
        default=4,
        help=(
            "Dataloader validation: number of dataset samples collated into one validation batch (PNG dumps + TB). "
            "Peak VRAM scales with this for the loss forward unless --validation_loss_micro_batch_size splits it."
        ),
    )
    parser.add_argument(
        "--validation_loss_micro_batch_size",
        type=int,
        default=1,
        help=(
            "Dataloader validation: forward the flow-matching loss this many samples at a time (default 1). "
            "Use 1 on tight GPUs when validation_num_samples > 1 — avoids one huge joint-attention forward."
        ),
    )
    parser.add_argument(
        "--validation_inference_steps",
        type=int,
        default=28,
        help="Dataloader validation: denoising steps when saving generated images.",
    )
    parser.add_argument(
        "--validation_samples_subdir",
        type=str,
        default="validation_samples",
        help="Dataloader validation: folder under output_dir for PNGs (per-step subfolders inside).",
    )
    parser.add_argument(
        "--validation_canvas_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Dataloader validation: when True (default), saved target/canvas PNGs use the same canvas augmentations "
            "as training (when --canvas_augment). When False, disable augments for clean, repeatable conditioning."
        ),
    )
    parser.add_argument(
        "--validation_depth_coarse_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When canvas aug is off for validation (--no-validation_canvas_augment), still apply depth blur/noise/"
            "patch dropout if True (default), matching stage3 inference. Use --no-validation_depth_coarse_augment "
            "for clean depth PNGs in validation."
        ),
    )
    parser.add_argument(
        "--text_encoder_out_layers",
        type=int,
        nargs="+",
        default=[10, 20, 30],
        help="Mistral hidden layer indices used for Flux2 prompt embeddings.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="Legacy only (--no-validation_from_dataloader): fixed prompt for Flux2Pipeline validation.",
    )
    parser.add_argument(
        "--subject_test_images",
        type=str,
        nargs="+",
        default=["/tiamat-NAS/zhangyuxuan/datasets/benchmark_dataset/decoritems_woodenpot/0.png"],
        help="A list of subject test image paths.",
    )
    parser.add_argument(
        "--spatial_test_images",
        type=str,
        nargs="+",
        default=[],
        help="A list of spatial test image paths.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Legacy only: images per validation when using Flux2Pipeline with --validation_prompt.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=20,
        help=(
            "Run validation every N optimizer steps. Default path: dataloader batch, TensorBoard metrics, "
            "saved PNGs. Legacy path: fixed prompt pipeline (--no-validation_from_dataloader)."
        ),
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[128],
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alphas",
        type=int,
        nargs="+",
        default=[128],
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tiamat-NAS/zhangyuxuan/projects2/Easy_Control_0120/single_models/subject_model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
             "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
             "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='Experiment tracker integration (e.g. `"tensorboard"`, `"comet_ml"`). See Accelerate `log_with`.',
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def main(args):
    _log_train(f"main() starting pid={os.getpid()} cwd={os.getcwd()}")
    _log_train(
        f"config: dataset_type={args.dataset_type} output_dir={args.output_dir!r} "
        f"mixed_precision={args.mixed_precision} guidance_scale={args.guidance_scale}"
    )
    if args.dataset_type == "manifest":
        _manifest = getattr(args, "manifest_path", "") or args.train_data_dir or ""
        _log_train(f"manifest path: {_manifest!r}")
    elif args.dataset_type == "canvas":
        _merged_roots = parse_canvas_data_roots(getattr(args, "canvas_data_roots", None) or "")
        if _merged_roots:
            _log_train(
                f"canvas merged data roots ({len(_merged_roots)}): {_merged_roots!r} — per root: "
                "bbox_results/yolo26_detections.csv, images/, depth/, image_captions/, multiview_out/"
            )
        else:
            _log_train(
                f"canvas paths: csv={args.csv_path!r} "
                f"canvas_image_root={getattr(args, 'canvas_image_root', '')!r} "
                f"depth_image_root={getattr(args, 'depth_image_root', '')!r} "
                f"caption_dir={getattr(args, 'caption_dir', '')!r}"
            )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)
    logging_dir = Path(args.output_dir, args.logging_dir)

    if args.subject_column == "None":
        args.subject_column = None
    if args.spatial_column == "None":
        args.spatial_column = None

    if args.dataset_type in ("canvas", "manifest") and not args.canvas_random_target_resolution:
        args.unified_train_width = multiple_16(args.unified_train_width)
        args.unified_train_height = multiple_16(args.unified_train_height)

    _canvas_data_roots_for_depth = parse_canvas_data_roots(getattr(args, "canvas_data_roots", None) or "")
    if args.dataset_type == "manifest":
        if args.lora_num < 2:
            logger.info(
                "manifest dataset includes depth conditioning: using lora_num=2 (was %s).",
                args.lora_num,
            )
            _log_train("manifest dataset → bumping lora_num to 2 for depth cond stream")
            args.lora_num = 2
    elif args.dataset_type == "canvas" and (
        (getattr(args, "depth_image_root", "") or "").strip() or _canvas_data_roots_for_depth
    ):
        if args.lora_num < 2:
            logger.info(
                "depth_image_root is set: using lora_num=2 for canvas + depth conditioning blocks (was %s).",
                args.lora_num,
            )
            _log_train("depth_image_root set → bumping lora_num to 2 for depth cond stream")
            args.lora_num = 2

    _log_train("creating Accelerator (DDP / mixed precision / logging dir)")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    _train_lz(
        accelerator,
        f"accelerator ready: num_processes={accelerator.num_processes} device={accelerator.device} "
        f"distributed_type={getattr(accelerator, 'distributed_type', 'n/a')}",
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    _train_lz(accelerator, "verbosity configured; beginning model/tokenizer load")

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    tokenizer_path = os.path.join(args.pretrained_model_name_or_path, "tokenizer")
    _train_lz(accelerator, f"loading PixtralProcessor from {tokenizer_path!r}")
    tokenizer = PixtralProcessor.from_pretrained(
        tokenizer_path,
        revision=args.revision,
    )
    _train_lz(accelerator, "PixtralProcessor loaded; loading text_encoder")
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    _train_lz(accelerator, "text_encoder loaded; loading scheduler")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    _train_lz(accelerator, "scheduler ready; loading VAE (can take a while)")
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    _train_lz(accelerator, "VAE loaded; loading Flux2 transformer weights (can take a while)")
    transformer_base = Flux2Transformer2DModelBase.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )
    _train_lz(accelerator, "transformer checkpoint in memory; building FluxTransformer2DModel from config")
    transformer = FluxTransformer2DModel.from_config(transformer_base.config)
    transformer.load_state_dict(transformer_base.state_dict(), strict=True)
    del transformer_base
    _train_lz(accelerator, "transformer state_dict applied; setting requires_grad flags")

    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    _train_lz(
        accelerator,
        f"moving VAE, transformer, text_encoder to {accelerator.device} (dtype={weight_dtype})",
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    _train_lz(accelerator, "models on device")

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        _train_lz(accelerator, "gradient checkpointing enabled on transformer")

    if args.pretrained_lora_path is not None:
        logger.warning(
            "pretrained_lora_path is ignored for Flux2 training (FLUX.1 LoRA checkpoints are incompatible)."
        )

    dim = transformer.inner_dim
    sa0 = transformer.single_transformer_blocks[0].attn
    s_inner, s_mlp_h, s_mlp_mf = sa0.inner_dim, sa0.mlp_hidden_dim, sa0.mlp_mult_factor
    double_blocks_idx = list(range(len(transformer.transformer_blocks)))
    single_blocks_idx = list(range(len(transformer.single_transformer_blocks)))
    lora_w = [1.0 for _ in range(args.lora_num)]

    if args.dataset_type in ("canvas", "manifest") and not args.canvas_random_target_resolution:
        lora_cond_w, lora_cond_h = args.unified_train_width, args.unified_train_height
    else:
        lora_cond_w = lora_cond_h = args.cond_size

    # MultiDoubleStreamBlockFlux2LoraProcessor indexes ranks[i] / network_alphas[i] for i in range(n_loras).
    _ranks = list(args.ranks)
    _alphas = list(args.network_alphas)
    n_need = args.lora_num
    if len(_ranks) < n_need or len(_alphas) < n_need:
        logger.info(
            "Padding ranks / network_alphas to length %s (repeat last value) for lora_num=%s.",
            n_need,
            n_need,
        )
    while len(_ranks) < n_need:
        _ranks.append(_ranks[-1] if _ranks else 32)
    while len(_alphas) < n_need:
        _alphas.append(_alphas[-1] if _alphas else 32)
    if len(_ranks) > args.lora_num or len(_alphas) > args.lora_num:
        logger.warning(
            "Truncating ranks (%s) / network_alphas (%s) to lora_num=%s.",
            len(_ranks),
            len(_alphas),
            args.lora_num,
        )
        _ranks = _ranks[: args.lora_num]
        _alphas = _alphas[: args.lora_num]
    args.ranks = _ranks
    args.network_alphas = _alphas

    _train_lz(
        accelerator,
        f"building LoRA attn processors (double_blocks={len(double_blocks_idx)}, single_blocks={len(single_blocks_idx)})",
    )
    lora_attn_procs = {}
    for name, attn_processor in transformer.attn_processors.items():
        match = re.search(r"\.(\d+)\.", name)
        layer_index = int(match.group(1)) if match else -1
        if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
            lora_attn_procs[name] = MultiDoubleStreamBlockFlux2LoraProcessor(
                dim=dim,
                ranks=args.ranks,
                network_alphas=args.network_alphas,
                lora_weights=lora_w,
                device=accelerator.device,
                dtype=weight_dtype,
                cond_width=lora_cond_w,
                cond_height=lora_cond_h,
                n_loras=args.lora_num,
            )
        elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
            lora_attn_procs[name] = MultiSingleStreamBlockFlux2LoraProcessor(
                dim=dim,
                inner_dim=s_inner,
                mlp_hidden_dim=s_mlp_h,
                mlp_mult_factor=s_mlp_mf,
                ranks=args.ranks,
                network_alphas=args.network_alphas,
                lora_weights=lora_w,
                device=accelerator.device,
                dtype=weight_dtype,
                cond_width=lora_cond_w,
                cond_height=lora_cond_h,
                n_loras=args.lora_num,
            )
        else:
            lora_attn_procs[name] = attn_processor
    transformer.set_attn_processor(lora_attn_procs)
    transformer.train()
    lora_substrings = ("q_loras", "k_loras", "v_loras", "proj_loras", "qkv_mlp_loras")
    for n, param in transformer.named_parameters():
        if not any(s in n for s in lora_substrings):
            param.requires_grad = False
    trainable_m = sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1_000_000
    _train_lz(accelerator, f"LoRA processors attached; trainable params ~{trainable_m:.3f}M")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        global_step = int(path.split("-")[-1])
        initial_global_step = global_step
        _log_train(f"resume_from_checkpoint={path!r} → initial global_step={global_step}")
    else:
        initial_global_step = 0
        global_step = 0
        first_epoch = 0
        _log_train("training from scratch (no resume_from_checkpoint)")

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # Optimization parameters
    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
    transformer_parameters_with_lr = {"params": params_to_optimize, "lr": args.learning_rate}
    _train_lz(accelerator, "creating AdamW optimizer")

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        [transformer_parameters_with_lr],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    _train_lz(accelerator, "optimizer ready")

    if args.dataset_type == "manifest":
        manifest_path = getattr(args, "manifest_path", "") or args.train_data_dir or ""
        if not manifest_path:
            raise ValueError(
                "dataset_type=manifest requires --manifest_path or --train_data_dir "
                "(path to dataset_manifest.jsonl)."
            )
        args.manifest_path = manifest_path
        _train_lz(accelerator, "manifest asset pairing report (gt / canvas / depth / caption)")
        log_manifest_asset_pairing_report(args)
        _train_lz(accelerator, "building manifest train dataset")
        train_dataset = make_manifest_train_dataset(args, accelerator)
        cfn = collate_fn_canvas
        _train_lz(
            accelerator,
            f"manifest unified RGB {args.unified_train_width}x{args.unified_train_height} "
            f"(gt+canvas+depth, resize={args.canvas_unified_resize}); "
            f"LoRA cond grid {args.unified_train_width}x{args.unified_train_height}, lora_num={args.lora_num}",
        )
    elif args.dataset_type == "canvas":
        _roots = parse_canvas_data_roots(getattr(args, "canvas_data_roots", None) or "")
        if not _roots and not args.csv_path:
            raise ValueError(
                "dataset_type=canvas requires --csv_path or non-empty --canvas_data_roots "
                "(comma-separated output directories)."
            )
        _train_lz(accelerator, "canvas asset pairing report (images / depth / captions by stem)")
        if _roots:
            for root_s in _roots:
                r = Path(root_s).expanduser().resolve()
                sub_args = copy.copy(args)
                sub_args.csv_path = str(r / CANVAS_DATA_ROOT_CSV)
                sub_args.canvas_image_root = str(r / "images")
                sub_args.depth_image_root = str(r / "depth")
                sub_args.caption_dir = str(r / "image_captions")
                sub_args.canvas_multiview_dir = str(r / "multiview_out")
                log_canvas_asset_pairing_report(sub_args)
            _train_lz(
                accelerator,
                f"building merged canvas train dataset ({len(_roots)} root(s); "
                f"balanced per-root sampling={'on' if getattr(args, 'canvas_balanced_concat_sampling', True) else 'off'} "
                f"when num_processes==1)",
            )
        else:
            log_canvas_asset_pairing_report(args)
            _train_lz(accelerator, "building canvas train dataset")
        train_dataset = make_canvas_train_dataset(args, accelerator)
        cfn = collate_fn_canvas
        if not args.canvas_random_target_resolution:
            dextra = (
                "; depth as 2nd cond stream"
                if (getattr(args, "depth_image_root", "") or "").strip()
                else ""
            )
            _train_lz(
                accelerator,
                f"canvas unified RGB {args.unified_train_width}x{args.unified_train_height} "
                f"(target+canvas, resize={args.canvas_unified_resize}){dextra}; "
                f"LoRA cond grid {args.unified_train_width}x{args.unified_train_height}, lora_num={args.lora_num}",
            )
        else:
            _train_lz(accelerator, "canvas random target resolution + cond_size² subject (prefer batch_size=1)")
    else:
        _train_lz(accelerator, "building jsonl train dataset")
        train_dataset = make_train_dataset(args, accelerator)
        cfn = collate_fn
    _train_lz(accelerator, f"dataset built: len={len(train_dataset)}")

    _concat_lens = concat_dataset_child_lengths(train_dataset)
    _use_balanced = (
        args.dataset_type == "canvas"
        and getattr(args, "canvas_balanced_concat_sampling", True)
        and _concat_lens is not None
        and accelerator.num_processes == 1
    )
    if (
        args.dataset_type == "canvas"
        and getattr(args, "canvas_balanced_concat_sampling", True)
        and _concat_lens is not None
        and accelerator.num_processes > 1
    ):
        _train_lz(
            accelerator,
            "canvas_balanced_concat_sampling is disabled when num_processes>1 (fallback: shuffle=True); "
            "run single-GPU to balance unequal concat roots.",
        )

    _dl_kwargs: dict = {
        "dataset": train_dataset,
        "batch_size": args.train_batch_size,
        "collate_fn": cfn,
        "num_workers": args.dataloader_num_workers,
    }
    if _use_balanced:
        _w = balanced_sampling_weights_for_concat(train_dataset)
        _g = torch.Generator()
        if args.seed is not None:
            _g.manual_seed(int(args.seed))
        _sampler = WeightedRandomSampler(
            weights=_w,
            num_samples=len(train_dataset),
            replacement=True,
            generator=_g,
        )
        _dl_kwargs["sampler"] = _sampler
        _dl_kwargs["shuffle"] = False
        _train_lz(
            accelerator,
            f"DataLoader: WeightedRandomSampler (balanced roots), child lengths={_concat_lens}, "
            f"num_samples/epoch={len(train_dataset)}",
        )
    else:
        _dl_kwargs["shuffle"] = True

    train_dataloader = torch.utils.data.DataLoader(**_dl_kwargs)
    _train_lz(
        accelerator,
        f"DataLoader ready (batch_size={args.train_batch_size}, num_workers={args.dataloader_num_workers})",
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.resume_from_checkpoint:
        first_epoch = global_step // num_update_steps_per_epoch
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    _train_lz(accelerator, "lr_scheduler created; calling accelerator.prepare (DDP wrap — may pause until all ranks sync)")

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    _train_lz(accelerator, "accelerator.prepare finished")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "Easy_Control"
        accelerator.init_trackers(tracker_name, config=_tracker_config_from_args(args))
    _train_lz(accelerator, "trackers initialized (main only); entering training loop")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if accelerator.is_main_process:
        _log_train(
            f"run summary: samples={len(train_dataset)} batches/epoch={len(train_dataloader)} "
            f"epochs={args.num_train_epochs} max_train_steps={args.max_train_steps} "
            f"train_batch_size={args.train_batch_size} accum={args.gradient_accumulation_steps} "
            f"prompt_warmup_steps={getattr(args, 'prompt_warmup_steps', 0)}"
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    logged_first_batch = False
    logged_first_optim_step = False
    for epoch in range(first_epoch, args.num_train_epochs):
        if accelerator.is_main_process and (epoch < 3 or epoch % 50 == 0):
            _log_train(f"epoch {epoch + 1}/{args.num_train_epochs} starting (global_step={global_step})")
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            if not logged_first_batch:
                _train_lz(
                    accelerator,
                    f"first batch from DataLoader (epoch={epoch}, in-epoch_step={step}); encoding prompts and latents",
                )
                logged_first_batch = True
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["prompts"]
                wu = int(getattr(args, "prompt_warmup_steps", 0) or 0)
                if wu > 0 and global_step < wu:
                    n = len(prompts) if isinstance(prompts, list) else int(batch["pixel_values"].shape[0])
                    prompts = [""] * n
                prompt_embeds, text_ids = encode_prompts_flux2(
                    text_encoder,
                    tokenizer,
                    prompts,
                    accelerator.device,
                    args.max_sequence_length,
                    weight_dtype,
                    tuple(args.text_encoder_out_layers),
                )
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=accelerator.device)
                text_ids = text_ids.to(dtype=weight_dtype, device=accelerator.device)

                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=torch.float32)
                model_input = encode_flux2_latents(vae, pixel_values, weight_dtype)
                latent_image_ids = Flux2Pipeline._prepare_latent_ids(model_input)

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = Flux2Pipeline._pack_latents(noisy_model_input)

                latent_image_ids_to_concat = [latent_image_ids]
                packed_cond_model_input_to_concat = []

                if batch.get("subject_pixel_values") is not None:
                    subject_pixel_values = batch["subject_pixel_values"].to(
                        device=accelerator.device, dtype=torch.float32
                    )
                    subject_latents = encode_flux2_latents(vae, subject_pixel_values, weight_dtype)
                    latent_image_ids_to_concat.append(prepare_subject_latent_ids(subject_latents))
                    packed_cond_model_input_to_concat.append(Flux2Pipeline._pack_latents(subject_latents))

                if batch.get("cond_pixel_values") is not None:
                    cond_pixel_values = batch["cond_pixel_values"].to(device=accelerator.device, dtype=torch.float32)
                    cond_latents = encode_flux2_latents(vae, cond_pixel_values, weight_dtype)
                    latent_image_ids_to_concat.append(prepare_subject_latent_ids(cond_latents))
                    packed_cond_model_input_to_concat.append(Flux2Pipeline._pack_latents(cond_latents))

                latent_image_ids = torch.cat(latent_image_ids_to_concat, dim=1)
                if packed_cond_model_input_to_concat:
                    cond_packed_noisy_model_input = torch.cat(packed_cond_model_input_to_concat, dim=1)
                else:
                    cond_packed_noisy_model_input = None

                guidance = torch.full(
                    (bsz,), args.guidance_scale, device=accelerator.device, dtype=weight_dtype
                )

                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    cond_hidden_states=cond_packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                main_seq = packed_noisy_model_input.shape[1]
                main_latent_ids = latent_image_ids[:, :main_seq, :]
                model_pred = unpack_main_latents(model_pred, main_latent_ids)

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                target = noise - model_input

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )

                loss = loss.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (transformer.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process and not logged_first_optim_step:
                    _log_train(f"first optimizer step finished global_step={global_step}")
                    logged_first_optim_step = True

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        unwrapped_model_state = accelerator.unwrap_model(transformer).state_dict()
                        lora_state_dict = {
                            k: unwrapped_model_state[k]
                            for k in unwrapped_model_state.keys()
                            if any(s in k for s in lora_substrings)
                        }
                        save_file(
                            lora_state_dict,
                            os.path.join(save_path, "lora.safetensors")
                        )
                        logger.info(f"Saved state to {save_path}")
                        logzero.logger.info("[train] wrote LoRA checkpoint global_step=%s path=%s", global_step, save_path)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process and not args.skip_validation and global_step % args.validation_steps == 0:
                _log_train(
                    f"running validation (global_step={global_step}, "
                    f"from_dataloader={args.validation_from_dataloader})"
                )
                did_validate = False
                if args.validation_from_dataloader:
                    from src.flux2_dataloader_validation import run_dataloader_validation

                    run_dataloader_validation(
                        accelerator=accelerator,
                        args=args,
                        train_dataset=train_dataset,
                        collate_fn=cfn,
                        vae=vae,
                        transformer=accelerator.unwrap_model(transformer),
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        noise_scheduler_copy=noise_scheduler_copy,
                        weight_dtype=weight_dtype,
                        global_step=global_step,
                    )
                    did_validate = True
                elif args.validation_prompt is not None:
                    pipeline = Flux2Pipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        tokenizer=tokenizer,
                        transformer=accelerator.unwrap_model(transformer),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )
                    pipeline_args = {
                        "prompt": args.validation_prompt,
                        "height": args.test_h,
                        "width": args.test_w,
                        "guidance_scale": args.guidance_scale,
                        "num_inference_steps": 20,
                        "max_sequence_length": args.max_sequence_length,
                        "text_encoder_out_layers": tuple(args.text_encoder_out_layers),
                    }
                    images = log_validation(
                        pipeline=pipeline,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=pipeline_args,
                        step=global_step,
                        torch_dtype=weight_dtype,
                    )
                    save_path = os.path.join(args.output_dir, "validation")
                    os.makedirs(save_path, exist_ok=True)
                    save_folder = os.path.join(save_path, f"checkpoint-{global_step}")
                    os.makedirs(save_folder, exist_ok=True)
                    for idx, img in enumerate(images):
                        img.save(os.path.join(save_folder, f"{idx}.jpg"))
                    del pipeline
                    did_validate = True
                if did_validate:
                    _log_train(f"validation finished for global_step={global_step}")
                else:
                    _log_train(
                        "validation skipped (no dataloader path and validation_prompt is None)"
                    )

    _train_lz(accelerator, "training loop exited; syncing all processes …")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _log_train("done — accelerator.end_training()")
    accelerator.end_training()


if __name__ == "__main__":
    logzero.loglevel(logging.INFO)
    _log_train("train.py __main__: parsing CLI …")
    args = parse_args()
    _log_train("CLI parsed — entering main()")
    main(args)
    _log_train("train.py normal exit")
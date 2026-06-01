#!/usr/bin/env python
# encoding: utf-8
"""Synthesize realistic low-light ReID images from normal-light images."""

from __future__ import print_function

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance

try:
    import numpy as np
except ImportError:
    np = None


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a synthetic low-light copy of a normal-light image tree."
    )
    parser.add_argument("--src-root", required=True, help="Input normal-light dataset root.")
    parser.add_argument("--dark-root", required=True, help="Output root for low-light images.")
    parser.add_argument(
        "--gt-root",
        default=None,
        help="Optional output root for unchanged normal-light GT images.",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated image extensions to process.",
    )
    parser.add_argument(
        "--include-dirs",
        default="",
        help="Optional comma-separated top-level subdirectories to process.",
    )
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done.")
    parser.add_argument("--quality", type=int, default=95, help="JPEG save quality.")
    parser.add_argument(
        "--brightness",
        type=float,
        nargs=2,
        default=(0.10, 0.38),
        metavar=("MIN", "MAX"),
        help="Base darkness control range. Lower values produce darker images.",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        nargs=2,
        default=(0.20, 0.35),
        metavar=("MIN", "MAX"),
        help="Base saturation control range.",
    )
    parser.add_argument(
        "--hue",
        type=float,
        nargs=2,
        default=(7.0, 30.0),
        metavar=("MIN_DEG", "MAX_DEG"),
        help="Base absolute hue-shift range in degrees.",
    )
    return parser.parse_args()


def validate_range(name, values):
    low, high = values
    if low > high:
        raise ValueError("{} min must be <= max".format(name))
    if name != "hue" and low < 0:
        raise ValueError("{} factors must be non-negative".format(name))


def iter_images(src_root, extensions):
    for path in sorted(src_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def shift_hue(image, degrees):
    if degrees == 0:
        return image

    hsv = image.convert("HSV")
    hue, saturation, value = hsv.split()
    shift = int(round(degrees / 360.0 * 255.0))
    hue = hue.point(lambda pixel: (pixel + shift) % 256)
    return Image.merge("HSV", (hue, saturation, value)).convert("RGB")


def synthesize_image(image, rng, args):
    if np is None:
        raise RuntimeError("This script requires numpy for realistic low-light synthesis.")

    arr = np.asarray(image).astype(np.float32) / 255.0
    height, width = arr.shape[:2]

    exposure_min = max(0.18, args.brightness[0] * 2.0)
    exposure_max = min(0.95, args.brightness[1] * 2.2)
    if exposure_min > exposure_max:
        exposure_min, exposure_max = exposure_max, exposure_min
    exposure = rng.uniform(exposure_min, exposure_max)
    gamma = rng.uniform(1.45, 2.35)
    black_level = rng.uniform(0.00, 0.045)

    arr = np.clip((arr - black_level) / max(1.0 - black_level, 1e-6), 0.0, 1.0)
    arr = np.power(arr, gamma) * exposure

    color_casts = (
        (0.88, 0.96, 1.12),  # cool street/night cast
        (1.08, 0.96, 0.78),  # sodium lamp cast
        (0.92, 1.02, 0.92),  # greenish surveillance cast
        (1.00, 0.98, 0.95),  # near-neutral dim light
    )
    gains = np.array(rng.choice(color_casts), dtype=np.float32)
    jitter = np.array(
        [rng.uniform(0.94, 1.06), rng.uniform(0.94, 1.06), rng.uniform(0.94, 1.06)],
        dtype=np.float32,
    )
    arr *= gains * jitter

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    center_x = rng.uniform(0.35, 0.65)
    center_y = rng.uniform(0.35, 0.65)
    dist = ((xx - center_x) ** 2 + (yy - center_y) ** 2) / 0.55
    vignette = 1.0 - rng.uniform(0.18, 0.42) * np.clip(dist, 0.0, 1.0)

    grad_x = np.linspace(rng.uniform(0.86, 1.10), rng.uniform(0.86, 1.10), width)
    grad_y = np.linspace(rng.uniform(0.90, 1.08), rng.uniform(0.90, 1.08), height)
    illumination = vignette * grad_y[:, None] * grad_x[None, :]
    arr *= illumination[:, :, None].astype(np.float32)

    noise_rng = np.random.RandomState(rng.randint(0, 2 ** 31 - 1))
    noise_sigma = rng.uniform(0.004, 0.018)
    noise = noise_rng.normal(0.0, noise_sigma, arr.shape).astype(np.float32)
    arr += noise * np.sqrt(np.maximum(arr, 0.02))

    arr = np.clip(arr, 0.0, 1.0)
    image = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")

    saturation = rng.uniform(
        max(0.35, args.saturation[0] * 1.8),
        max(0.65, args.saturation[1] * 2.2),
    )
    image = ImageEnhance.Color(image).enhance(saturation)
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.12))

    hue = rng.uniform(0.0, min(args.hue[1] * 0.5, 15.0))
    if rng.random() < 0.5:
        hue = -hue
    return shift_hue(image, hue)


def save_image(image, output_path, quality):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        image.save(output_path, quality=quality)
    else:
        image.save(output_path)


def copy_gt(src_path, gt_path, overwrite, dry_run):
    if gt_path.exists() and not overwrite:
        return False
    if dry_run:
        return True
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src_path), str(gt_path))
    return True


def main():
    args = parse_args()
    for name in ("brightness", "saturation", "hue"):
        validate_range(name, getattr(args, name))

    src_root = Path(args.src_root).resolve()
    dark_root = Path(args.dark_root).resolve()
    gt_root = Path(args.gt_root).resolve() if args.gt_root else None
    extensions = tuple(ext.strip().lower() for ext in args.extensions.split(",") if ext.strip())
    include_dirs = tuple(
        name.strip() for name in args.include_dirs.split(",") if name.strip()
    )

    if not src_root.exists():
        raise RuntimeError("Source root does not exist: {}".format(src_root))

    rng = random.Random(args.seed)
    total = 0
    written_dark = 0
    written_gt = 0

    for src_path in iter_images(src_root, extensions):
        rel_path = src_path.relative_to(src_root)
        if include_dirs and (not rel_path.parts or rel_path.parts[0] not in include_dirs):
            continue
        dark_path = dark_root / rel_path
        gt_path = gt_root / rel_path if gt_root else None
        total += 1

        if not dark_path.exists() or args.overwrite:
            written_dark += 1
            if not args.dry_run:
                with Image.open(src_path) as image:
                    image = image.convert("RGB")
                    dark_image = synthesize_image(image, rng, args)
                    save_image(dark_image, dark_path, args.quality)

        if gt_path is not None:
            if copy_gt(src_path, gt_path, args.overwrite, args.dry_run):
                written_gt += 1

    action = "would write" if args.dry_run else "wrote"
    print("Scanned {} images under {}".format(total, src_root))
    print("{} {} low-light images to {}".format(action, written_dark, dark_root))
    if gt_root is not None:
        print("{} {} GT images to {}".format(action, written_gt, gt_root))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# encoding: utf-8
"""Build Syn_dark_market_msmt_v3 from Market1501 and MSMT17.

Market1501 keeps the original file names and split directories. MSMT17 train
and test images are merged into bounding_box_train with new ReID-style names.
Both synthetic low-light images and unchanged normal-light GT images are saved.
"""

from __future__ import print_function

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

from synthesize_lowlight import save_image, synthesize_image, validate_range


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MARKET_DIRS = ("bounding_box_train", "query", "bounding_box_test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a merged synthetic low-light Market1501+MSMT17 dataset."
    )
    parser.add_argument("--market-root", required=True, help="Market1501 dataset root.")
    parser.add_argument(
        "--msmt-root",
        required=True,
        help="MSMT17 root, or a parent directory containing MSMT17_V1/train and test.",
    )
    parser.add_argument("--dark-root", required=True, help="Output root for low-light images.")
    parser.add_argument("--gt-root", required=True, help="Output root for normal-light GT images.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done.")
    parser.add_argument("--quality", type=int, default=95, help="JPEG save quality.")
    parser.add_argument(
        "--msmt-pid-offset",
        type=int,
        default=1502,
        help="MSMT17 new_pid = this offset + msmt_global_pid.",
    )
    parser.add_argument(
        "--msmt-cam-offset",
        type=int,
        default=6,
        help="MSMT17 new_camid = old_camid + this offset.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting global image index for renamed MSMT17 images.",
    )
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


def iter_images(root):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def is_market_junk(src_path):
    return src_path.name.split("_", 1)[0] == "-1"


def resolve_msmt_v1_root(msmt_root):
    candidates = [
        msmt_root,
        msmt_root / "MSMT17_V1",
        msmt_root / "MSMT17_V1-001" / "MSMT17_V1",
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate

    for candidate in sorted(msmt_root.rglob("MSMT17_V1")):
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate

    raise RuntimeError("Could not find MSMT17_V1/train and test under {}".format(msmt_root))


def ensure_parent(path, dry_run):
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)


def copy_gt(src_path, output_path, overwrite, dry_run):
    if output_path.exists() and not overwrite:
        return False
    ensure_parent(output_path, dry_run)
    if not dry_run:
        shutil.copy2(str(src_path), str(output_path))
    return True


def write_dark(src_path, output_path, rng, args):
    if output_path.exists() and not args.overwrite:
        return False
    if args.dry_run:
        return True

    with Image.open(src_path) as image:
        image = image.convert("RGB")
        dark_image = synthesize_image(image, rng, args)
        save_image(dark_image, output_path, args.quality)
    return True


def process_market(args, market_root, dark_root, gt_root, rng):
    scanned = 0
    skipped_junk = 0
    written_dark = 0
    written_gt = 0

    for dirname in MARKET_DIRS:
        input_dir = market_root / dirname
        if not input_dir.is_dir():
            raise RuntimeError("Missing Market1501 directory: {}".format(input_dir))

        for src_path in iter_images(input_dir):
            if is_market_junk(src_path):
                skipped_junk += 1
                continue

            rel_path = Path(dirname) / src_path.name
            dark_path = dark_root / rel_path
            gt_path = gt_root / rel_path
            scanned += 1

            if write_dark(src_path, dark_path, rng, args):
                written_dark += 1
            if copy_gt(src_path, gt_path, args.overwrite, args.dry_run):
                written_gt += 1

    return scanned, skipped_junk, written_dark, written_gt


def get_pid_dirs(root):
    return [path for path in sorted(root.iterdir()) if path.is_dir() and path.name.isdigit()]


def parse_msmt_camid(src_path):
    parts = src_path.stem.split("_")
    if len(parts) < 3:
        raise RuntimeError("Unexpected MSMT17 file name: {}".format(src_path.name))
    return int(parts[2])


def process_msmt(args, msmt_v1_root, dark_root, gt_root, rng):
    train_dir = msmt_v1_root / "train"
    test_dir = msmt_v1_root / "test"
    train_pid_dirs = get_pid_dirs(train_dir)
    test_pid_dirs = get_pid_dirs(test_dir)
    num_train_pids = len(train_pid_dirs)

    scanned = 0
    written_dark = 0
    written_gt = 0
    global_index = args.start_index

    split_infos = (("train", train_pid_dirs, 0), ("test", test_pid_dirs, num_train_pids))
    for _, pid_dirs, pid_base in split_infos:
        for pid_dir in pid_dirs:
            old_pid = int(pid_dir.name)
            msmt_global_pid = pid_base + old_pid
            new_pid = args.msmt_pid_offset + msmt_global_pid

            for src_path in iter_images(pid_dir):
                old_camid = parse_msmt_camid(src_path)
                new_camid = old_camid + args.msmt_cam_offset
                new_name = "{:04d}_c{}s1_{:06d}_00.jpg".format(
                    new_pid, new_camid, global_index
                )
                rel_path = Path("bounding_box_train") / new_name
                dark_path = dark_root / rel_path
                gt_path = gt_root / rel_path
                scanned += 1
                global_index += 1

                if write_dark(src_path, dark_path, rng, args):
                    written_dark += 1
                if copy_gt(src_path, gt_path, args.overwrite, args.dry_run):
                    written_gt += 1

    return scanned, written_dark, written_gt, num_train_pids, len(test_pid_dirs)


def main():
    args = parse_args()
    for name in ("brightness", "saturation", "hue"):
        validate_range(name, getattr(args, name))

    market_root = Path(args.market_root).resolve()
    msmt_root = Path(args.msmt_root).resolve()
    dark_root = Path(args.dark_root).resolve()
    gt_root = Path(args.gt_root).resolve()

    if not market_root.is_dir():
        raise RuntimeError("Market1501 root does not exist: {}".format(market_root))
    if not msmt_root.is_dir():
        raise RuntimeError("MSMT17 root does not exist: {}".format(msmt_root))

    msmt_v1_root = resolve_msmt_v1_root(msmt_root)
    rng = random.Random(args.seed)

    market_stats = process_market(args, market_root, dark_root, gt_root, rng)
    msmt_stats = process_msmt(args, msmt_v1_root, dark_root, gt_root, rng)

    action = "would write" if args.dry_run else "wrote"
    print("Market1501 root: {}".format(market_root))
    print("MSMT17_V1 root: {}".format(msmt_v1_root))
    print(
        "Market1501 scanned {}, skipped {} junk images, {} {} low-light, {} {} GT".format(
            market_stats[0], market_stats[1], action, market_stats[2], action, market_stats[3]
        )
    )
    print(
        "MSMT17 scanned {}, train pids {}, test pids {}, {} {} low-light, {} {} GT".format(
            msmt_stats[0], msmt_stats[3], msmt_stats[4], action, msmt_stats[1], action, msmt_stats[2]
        )
    )
    print("Low-light output: {}".format(dark_root))
    print("GT output: {}".format(gt_root))


if __name__ == "__main__":
    main()

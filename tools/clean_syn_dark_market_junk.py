#!/usr/bin/env python
# encoding: utf-8
"""Remove Market1501 junk images whose file names start with -1_."""

from __future__ import print_function

import argparse
from pathlib import Path


DEFAULT_SPLITS = ("bounding_box_train", "query", "bounding_box_test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean -1_ Market1501 junk images from Syn_dark output directories."
    )
    parser.add_argument("--dark-root", required=True, help="Synthetic low-light dataset root.")
    parser.add_argument("--gt-root", required=True, help="Normal-light GT dataset root.")
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated split directories to clean.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag the script only reports counts.",
    )
    return parser.parse_args()


def find_junk_files(root, splits):
    files = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        files.extend(sorted(split_dir.glob("-1_*.jpg")))
        files.extend(sorted(split_dir.glob("-1_*.jpeg")))
        files.extend(sorted(split_dir.glob("-1_*.png")))
        files.extend(sorted(split_dir.glob("-1_*.bmp")))
    return files


def delete_files(files, apply):
    if not apply:
        return
    for path in files:
        path.unlink()


def main():
    args = parse_args()
    splits = tuple(split.strip() for split in args.splits.split(",") if split.strip())
    dark_root = Path(args.dark_root).resolve()
    gt_root = Path(args.gt_root).resolve()

    dark_files = find_junk_files(dark_root, splits)
    gt_files = find_junk_files(gt_root, splits)

    action = "Deleted" if args.apply else "Would delete"
    print("{} {} low-light junk files under {}".format(action, len(dark_files), dark_root))
    print("{} {} GT junk files under {}".format(action, len(gt_files), gt_root))

    for sample in (dark_files + gt_files)[:10]:
        print("  {}".format(sample))

    delete_files(dark_files, args.apply)
    delete_files(gt_files, args.apply)


if __name__ == "__main__":
    main()

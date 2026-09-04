import glob
import os.path as osp
import re

import torch
from torch.utils.data import Dataset

from .bases import read_image


IMG_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp')


def _collect_images(root, split='bounding_box_train'):
    img_dir = osp.join(root, split)
    if not osp.exists(img_dir):
        raise RuntimeError("'{}' is not available".format(img_dir))

    img_paths = []
    for ext in IMG_EXTENSIONS:
        img_paths.extend(glob.glob(osp.join(img_dir, ext)))
    return sorted(img_paths)


def build_srd_pairs(dark_root, gt_root, split='bounding_box_train'):
    dark_paths = _collect_images(dark_root, split)
    gt_paths = _collect_images(gt_root, split)
    gt_by_name = {osp.basename(path): path for path in gt_paths}
    pattern = re.compile(r'([-\d]+)_c(\d+)')

    pid_container = set()
    matched = []
    missing = []
    for dark_path in dark_paths:
        name = osp.basename(dark_path)
        gt_path = gt_by_name.get(name)
        if gt_path is None:
            missing.append(name)
            continue

        match = pattern.search(name)
        if match is None:
            raise RuntimeError("Image filename does not match pid/camid pattern: {}".format(dark_path))

        pid, camid = map(int, match.groups())
        if pid == -1:
            continue
        if camid < 1:
            raise RuntimeError("Camera id should start from 1, but got {} in {}".format(camid, dark_path))

        pid_container.add(pid)
        matched.append((dark_path, gt_path, pid, camid - 1, 1))

    if missing:
        raise RuntimeError(
            "Found {} synthetic dark images without GT pairs. First missing file: {}".format(
                len(missing), missing[0]
            )
        )
    if len(matched) == 0:
        raise RuntimeError("No SRD dark/GT pairs found in '{}' and '{}'".format(dark_root, gt_root))

    pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
    pairs = []
    for dark_path, gt_path, pid, camid, viewid in matched:
        pairs.append((dark_path, gt_path, pid2label[pid], camid, viewid))

    return pairs, len(pid2label)


class SRDPairDataset(Dataset):
    def __init__(
        self,
        pairs,
        dark_transform=None,
        gt_transform=None,
        order_transform=None,
    ):
        self.pairs = pairs
        self.dark_transform = dark_transform
        self.gt_transform = gt_transform
        self.order_transform = order_transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        dark_path, gt_path, pid, camid, viewid = self.pairs[index]
        dark_img = read_image(dark_path)
        gt_img = read_image(gt_path)

        # The ordinary dark view keeps the original ReID augmentation.  The
        # extra order view uses the deterministic GT geometry so patch-wise
        # normal/dark order supervision remains spatially aligned.
        order_dark_img = None
        if self.order_transform is not None:
            order_dark_img = self.order_transform(dark_img)

        if self.dark_transform is not None:
            dark_img = self.dark_transform(dark_img)
        if self.gt_transform is not None:
            gt_img = self.gt_transform(gt_img)

        sample = (dark_img, gt_img, pid, camid, viewid, dark_path, gt_path)
        if order_dark_img is not None:
            sample += (order_dark_img,)
        return sample


def srd_pair_collate_fn(batch):
    has_order_view = len(batch[0]) == 8
    if has_order_view:
        dark_imgs, gt_imgs, pids, camids, viewids, dark_paths, gt_paths, order_dark_imgs = zip(*batch)
    else:
        dark_imgs, gt_imgs, pids, camids, viewids, dark_paths, gt_paths = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    collated = (
        torch.stack(dark_imgs, dim=0),
        torch.stack(gt_imgs, dim=0),
        pids,
        camids,
        viewids,
        dark_paths,
        gt_paths,
    )
    if has_order_view:
        collated += (torch.stack(order_dark_imgs, dim=0),)
    return collated

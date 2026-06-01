# encoding: utf-8

import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class RGBNT201RGB(BaseImageDataset):
    """RGB-only protocol for RGBNT201.

    This loader intentionally uses only the RGB folders:
      - train_171/RGB for training
      - full test/RGB as query
      - full test/RGB as gallery
    """

    dataset_dir = 'RGBNT201'
    train_split = 'train_171'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(RGBNT201RGB, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, self.train_split, 'RGB')
        self.test_dir = osp.join(self.dataset_dir, 'test', 'RGB')

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.test_dir, relabel=False)
        gallery = self._process_dir(self.test_dir, relabel=False)

        if verbose:
            print("=> RGBNT201 RGB-only loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
            self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, relabel=False, allowed_cams=None):
        img_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            img_paths.extend(glob.glob(osp.join(dir_path, ext)))
        img_paths = sorted(img_paths)

        pattern = re.compile(r'^([-\d]+)_cam(\d+)_([-\d]+)_')

        pid_container = set()
        for img_path in img_paths:
            parsed = self._parse_img_name(pattern, img_path)
            if parsed is None:
                continue
            pid, camid, _ = parsed
            if allowed_cams is not None and camid not in allowed_cams:
                continue
            if pid == -1:
                continue
            pid_container.add(pid)

        if not pid_container:
            raise RuntimeError("No valid RGBNT201 RGB images found in '{}'.".format(dir_path))

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in img_paths:
            parsed = self._parse_img_name(pattern, img_path)
            if parsed is None:
                continue
            pid, camid, viewid = parsed
            if allowed_cams is not None and camid not in allowed_cams:
                continue
            if pid == -1:
                continue
            if camid < 1:
                raise RuntimeError("Camera id should start from 1, but got {} in {}".format(camid, img_path))

            camid -= 1
            if relabel:
                pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, viewid))

        return dataset

    @staticmethod
    def _parse_img_name(pattern, img_path):
        match = pattern.search(osp.basename(img_path))
        if match is None:
            return None
        pid, camid, viewid = map(int, match.groups())
        return pid, camid, viewid


class RGBNT201RGB141(RGBNT201RGB):
    """RGB-only RGBNT201 using train_141/RGB for training."""

    train_split = 'train_141'

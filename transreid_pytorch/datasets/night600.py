# encoding: utf-8

import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class Night600(BaseImageDataset):
    """Night600 person ReID dataset in Market1501-style folder layout."""

    dataset_dir = 'night600'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(Night600, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query_3')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Night600 loaded")
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
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            img_paths.extend(glob.glob(osp.join(dir_path, ext)))
        img_paths = sorted(img_paths)

        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            if match is None:
                raise RuntimeError("Image filename does not match pid/camid pattern: {}".format(img_path))

            pid, _ = map(int, match.groups())
            if pid == -1:
                continue
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            pid, camid = map(int, match.groups())
            if pid == -1:
                continue
            if camid < 1:
                raise RuntimeError("Camera id should start from 1, but got {} in {}".format(camid, img_path))

            camid -= 1
            if relabel:
                pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        return dataset

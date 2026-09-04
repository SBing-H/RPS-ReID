# encoding: utf-8

import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class NightReID(BaseImageDataset):
    """Night-ReID dataset using its official train/query/gallery splits."""

    dataset_dir = 'NightReID'
    _filename_pattern = re.compile(
        r'^(\d+)([RL])(\d+)C(\d+)\.(?:jpg|jpeg|png|bmp)$',
        re.IGNORECASE,
    )

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(NightReID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print('=> NightReID loaded')
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
        for required_dir in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.isdir(required_dir):
                raise RuntimeError("'{}' is not available".format(required_dir))

    @staticmethod
    def _collect_images(dir_path):
        img_paths = []
        for extension in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            img_paths.extend(glob.glob(osp.join(dir_path, extension)))
            img_paths.extend(glob.glob(osp.join(dir_path, extension.upper())))
        return sorted(set(img_paths))

    def _parse_filename(self, img_path):
        match = self._filename_pattern.match(osp.basename(img_path))
        if match is None:
            raise RuntimeError('Invalid NightReID filename: {}'.format(img_path))
        pid = int(match.group(1))
        side = match.group(2).upper()
        camera_index = int(match.group(3))
        if camera_index < 1:
            raise RuntimeError('NightReID camera index must start from 1: {}'.format(img_path))
        # R1-R3 and L1-L3 are the six physical camera views.
        camid = camera_index - 1 if side == 'R' else camera_index + 2
        return pid, camid

    def _process_dir(self, dir_path, relabel=False):
        img_paths = self._collect_images(dir_path)
        if not img_paths:
            raise RuntimeError("No supported images found in '{}'".format(dir_path))

        parsed = [(img_path, *self._parse_filename(img_path)) for img_path in img_paths]
        pid_container = sorted({pid for _, pid, _ in parsed if pid != -1})
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path, pid, camid in parsed:
            if pid == -1:
                continue
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

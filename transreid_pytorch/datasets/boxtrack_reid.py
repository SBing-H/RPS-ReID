# encoding: utf-8

import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class BoxTrackReID(BaseImageDataset):
    """BoxTrack-ReID dataset with a Market1501-style split layout.

    Expected directory structure::

        BoxTrack-ReID/
            bounding_box_train/
            query/
            bounding_box_test/

    Image names encode identity and camera information, for example
    ``001_C2_00013.jpg``. Both upper- and lower-case camera prefixes are
    accepted to keep the parser robust.
    """

    dataset_dir = 'BoxTrack-ReID'
    _filename_pattern = re.compile(
        r'^([-\d]+)_[cC](\d+)_(\d+)\.(?:jpg|jpeg|png|bmp)$',
        re.IGNORECASE,
    )

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(BoxTrackReID, self).__init__()
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
            print('=> BoxTrack-ReID loaded')
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
            raise RuntimeError(
                'BoxTrack-ReID filename does not match PID_C<camera>_<index>: {}'.format(
                    img_path
                )
            )
        pid, camid, _ = map(int, match.groups())
        return pid, camid

    def _process_dir(self, dir_path, relabel=False):
        img_paths = self._collect_images(dir_path)
        if not img_paths:
            raise RuntimeError("No supported images found in '{}'".format(dir_path))

        pid_container = set()
        for img_path in img_paths:
            pid, _ = self._parse_filename(img_path)
            if pid != -1:
                pid_container.add(pid)

        pid2label = {
            pid: label for label, pid in enumerate(sorted(pid_container))
        }

        dataset = []
        for img_path in img_paths:
            pid, camid = self._parse_filename(img_path)
            if pid == -1:
                continue
            if camid < 1:
                raise RuntimeError(
                    'Camera id should start from 1, but got {} in {}'.format(
                        camid, img_path
                    )
                )

            camid -= 1
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        return dataset

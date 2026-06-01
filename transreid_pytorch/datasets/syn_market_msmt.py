import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class _MarketStyleAuxDataset(BaseImageDataset):
    dataset_dir = ''

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(_MarketStyleAuxDataset, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.pid_begin = pid_begin

        self._check_before_run()
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> {} loaded".format(self.__class__.__name__))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

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
            if pid != -1:
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
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid - 1, 1))
        return dataset


class SynDarkMarketMSMT(_MarketStyleAuxDataset):
    dataset_dir = 'Syn_dark_market_msmt_v3'


class SynDarkMarketMSMTGT(_MarketStyleAuxDataset):
    dataset_dir = 'Syn_dark_market_msmt_v3_GT'

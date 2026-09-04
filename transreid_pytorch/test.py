import os
import torch
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, _, _ = make_dataloader(cfg, is_train=False)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)
    beta_override = float(cfg.TEST.PHOTOMETRIC_ORDER_BETA_OVERRIDE)
    if beta_override >= 0.0:
        order_module = getattr(model, 'photometric_order', None)
        if order_module is None:
            raise RuntimeError(
                'TEST.PHOTOMETRIC_ORDER_BETA_OVERRIDE requires a photometric-order model.'
            )
        learned_beta = float(order_module.beta.detach().cpu())
        with torch.no_grad():
            order_module.beta.fill_(beta_override)
        logger.info(
            'Photometric-order beta overridden for inference: {:.6f} -> {:.6f}'.format(
                learned_beta,
                beta_override,
            )
        )

    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, _, _ = make_dataloader(cfg, is_train=False)
            rank_1, rank5 = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0))
    else:
       do_inference(cfg,
                 model,
                 val_loader,
                 num_query)


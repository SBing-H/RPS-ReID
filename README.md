# RPS-ReID

Official PyTorch implementation of **RPS-ReID: Learning Relative Photometric
Structure for Nighttime Person Re-Identification**.

RPS-ReID models multi-scale relative luminance relationships between local
image patches and converts them into a structural attention prior. An
asymmetric dual-path Transformer combines appearance semantics with relative
photometric structure, while paired normal-light and synthetic low-light data
provide auxiliary identity supervision during training.

## Framework

The main components are:

- **Multi-scale soft photometric ordinality:** continuous, tolerance-aware
  encoding of relative patch luminance in the log-luminance domain.
- **Ordinality-guided structural attention:** patch-to-patch and
  identity-to-patch biases guide local interaction and global identity
  aggregation.
- **Asymmetric dual paths:** an appearance path preserves color, texture, and
  semantic cues, while a structure path introduces the photometric prior in
  its final global Transformer block.
- **Global-local representation:** global features and four horizontal local
  features from the two paths are fused for retrieval.
- **Paired auxiliary training:** a frozen normal-light teacher transfers
  identity knowledge to synthetic low-light samples through auxiliary ReID
  supervision and decoupled knowledge distillation (DKD). The teacher and
  auxiliary branch are removed at inference time.

## Results

Results are reported without re-ranking.

| Dataset | Modality | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---:|---:|---:|---:|
| Night600 | RGB | 12.5 | 28.1 | 43.7 | 52.2 |
| RGBNT201 | RGB only | 59.1 | 59.8 | 72.5 | 79.5 |

## Environment

The experiments were conducted on Ubuntu 20.04 with Python 3.10, PyTorch
2.6.0, and a single NVIDIA A100 GPU. Other recent CUDA-capable GPUs can be
used, but the batch size may need to be reduced.

```bash
conda create -n rps-reid python=3.10 -y
conda activate rps-reid
pip install -r requirements.txt
```

`faiss-gpu` is only required by the optional FAISS re-ranking utilities and is
not needed for the reported results. Install a CUDA-compatible FAISS build
separately if those utilities are used.

## Data Preparation

Place all datasets under the repository-level `data` directory or override
the relevant paths in the YAML configuration files.

```text
.
|-- data/
|   |-- night600/
|   |   |-- bounding_box_train/
|   |   |-- query_3/
|   |   `-- bounding_box_test/
|   |-- RGBNT201/
|   |   |-- train_171/RGB/
|   |   `-- test/RGB/
|   |-- Syn_dark_market_msmt_v3/
|   |   |-- bounding_box_train/
|   |   |-- query/
|   |   `-- bounding_box_test/
|   `-- Syn_dark_market_msmt_v3_GT/
|       |-- bounding_box_train/
|       |-- query/
|       `-- bounding_box_test/
|-- model/
|   `-- vit_base_ics_cfs_lup.pth
|-- log/
`-- transreid_pytorch/
```

Night600 filenames must contain the pattern `<pid>_c<camera>`. RGBNT201 RGB
filenames must follow `<pid>_cam<camera>_<view>_...`.

Datasets and pretrained weights are not distributed with this repository.
Please obtain them from their official sources and comply with their licenses.

### Build the paired auxiliary dataset

The following command combines Market1501 and MSMT17 and creates spatially
aligned normal-light and synthetic low-light images:

```bash
python tools/build_syn_dark_market_msmt_v3.py \
  --market-root /path/to/Market-1501-v15.09.15 \
  --msmt-root /path/to/MSMT17 \
  --dark-root data/Syn_dark_market_msmt_v3 \
  --gt-root data/Syn_dark_market_msmt_v3_GT
```

## Training

Run all training and evaluation commands from `transreid_pytorch`.

### 1. Train the normal-light teacher

```bash
cd transreid_pytorch
python train.py \
  --config_file configs/night600/vit_base_ics_srd_teacher_gt.yml
```

The default full-model configurations expect the teacher checkpoint at:

```text
../log/transreid/srd_teacher/vit_base_ics_gt/transformer_best.pth
```

Update `DISTILL.TEACHER_WEIGHT` if the checkpoint is stored elsewhere.

### 2. Train RPS-ReID on Night600

```bash
python train.py \
  --config_file configs/night600/vit_base_ics_srd_dual_photometric_order_structure_attention.yml
```

### 3. Train RPS-ReID on RGBNT201

```bash
python train.py \
  --config_file configs/rgbnt201/vit_base_ics_rgb_srd_dual_photometric_order.yml
```

Configuration values can be overridden from the command line. For example:

```bash
python train.py --config_file CONFIG.yml \
  MODEL.DEVICE_ID "('0')" \
  DATASETS.ROOT_DIR /path/to/data \
  MODEL.PRETRAIN_PATH /path/to/pretrained.pth
```

## Evaluation

```bash
cd transreid_pytorch
python test.py \
  --config_file configs/night600/vit_base_ics_srd_dual_photometric_order_structure_attention.yml \
  TEST.WEIGHT /path/to/transformer_best.pth
```

For RGBNT201, replace the configuration file with:

```text
configs/rgbnt201/vit_base_ics_rgb_srd_dual_photometric_order.yml
```

## Key Reproducibility Settings

| Setting | Night600 | RGBNT201 |
|---|---:|---:|
| Input size | 256 x 128 | 256 x 128 |
| Batch size | 64 | 64 |
| Training epochs | 120 | 120 |
| Initial learning rate | 4e-4 | 3e-4 |
| Random seed | 1234 | 1234 |
| Ordinality scales | {1, 2} | {1, 2} |
| Soft-order margin | 0.02 | 0.02 |
| Soft-order temperature | 0.10 | 0.10 |
| Attention-bias scale | 0.20 | 0.15 |
| Local radius | 2 | 2 |

## Repository Structure

```text
.
|-- figures/                    # Framework figures
|-- log
|-- tools/                      # Data synthesis and analysis utilities
|-- transreid_pytorch/
|   |-- config/                 # Default configuration
|   |-- configs/                # Dataset and experiment configurations
|   |-- datasets/               # Dataset loaders and paired data pipeline
|   |-- loss/                   # ReID, DKD, and auxiliary objectives
|   |-- model/                  # Transformer and RPS-ReID modules
|   |-- processor/              # Training and inference loops
|   |-- solver/                 # Optimizers and learning-rate schedules
|   |-- train.py
|   `-- test.py
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Citation

The citation entry will be added after publication. If this repository is
useful in your research, please cite the paper and the upstream TransReID work.

## Acknowledgements

This implementation is built on
[TransReID](https://github.com/damo-cv/TransReID) and retains components from
its related research code. We thank the original authors and contributors.

## License

RPS-ReID is released under the [MIT License](LICENSE). Third-party components
remain subject to their original copyright notices and license terms.

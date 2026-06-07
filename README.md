# NGL-TransReID

NGL-TransReID is a PyTorch implementation for low-light person re-identification. It is built on TransReID and adds norm-guided token structure enhancement, dual raw/structure branches, local part fusion, SRD auxiliary distillation, and checkpoint ensemble evaluation.

This repository keeps source code only. Datasets, checkpoints, logs, generated outputs, cache files, and local archives are ignored by Git.

## Highlights

- **NG-TSE**: norm-guided token structure enhancement for selected ViT blocks. The module uses attention q/k norm cues to gate neighborhood-based token structure updates.
- **Dual-branch transformer**: a raw TransReID branch and a structure-enhanced branch are fused for the final embedding.
- **Dual local features**: optional horizontal part features are extracted from both branches and fused with the global feature.
- **SRD distillation**: paired synthetic dark images and normal-light GT images are used with an auxiliary ReID loss and DKD teacher guidance.
- **Low-light ReID configs**: experiment configs are included for Night600 and RGBNT201 RGB-only protocols.
- **Ensemble testing**: multiple checkpoints can be evaluated by feature-level or distance-level ensemble.

## Repository Layout

```text
.
|-- transreid_pytorch/        # Main NGL-TransReID training and testing code
|-- dino/                     # DINO/self-supervised pretraining code kept from the base project
|-- cluster-contrast-reid/    # Unsupervised ReID code kept from the base project
|-- tools/                    # Low-light synthesis and dataset utility scripts
|-- requirements.txt
`-- README.md
```

## Environment

The codebase was developed around the original TransReID dependency stack:

```bash
pip install -r requirements.txt
```

The pinned requirements include PyTorch 1.7.1, torchvision 0.8.2, timm 0.3.4, yacs, faiss-gpu, and common scientific Python packages. Adjust CUDA/PyTorch versions for your local GPU environment if needed.

## Data And Weights

Datasets and model weights are not included in this repository. The default configs expect paths relative to `transreid_pytorch/`, for example:

```text
data/
|-- night600/
|   |-- bounding_box_train/
|   |-- query_3/
|   `-- bounding_box_test/
|-- RGBNT201/
|   |-- train_171/RGB/
|   |-- train_141/RGB/
|   `-- test/RGB/
|-- Syn_dark_market_msmt_v3/
|   `-- bounding_box_train/
`-- Syn_dark_market_msmt_v3_GT/
    `-- bounding_box_train/

model/
`-- vit_base_ics_cfs_lup.pth

log/
`-- transreid/srd_teacher/vit_base_ics_gt/transformer_best.pth
```

If your paths are different, edit these config fields:

- `DATASETS.ROOT_DIR`
- `MODEL.PRETRAIN_PATH`
- `DISTILL.DARK_ROOT`
- `DISTILL.GT_ROOT`
- `DISTILL.TEACHER_WEIGHT`
- `OUTPUT_DIR`
- `TEST.WEIGHT`

## Quick Start

Run commands from `transreid_pytorch/`.

Train the SRD teacher on normal-light GT auxiliary data:

```bash
cd transreid_pytorch
python train.py --config_file configs/night600/vit_base_ics_srd_teacher_gt.yml
```

Train the full NGL-TransReID model on Night600:

```bash
python train.py --config_file configs/night600/vit_base_ics_srd_dual_ngtse_V13.yml
```

Train the full NGL-TransReID model on RGBNT201 RGB-only:

```bash
python train.py --config_file configs/rgbnt201/vit_base_ics_rgb_srd_dual_ngtse_V28.yml
```

Evaluate one checkpoint:

```bash
python test.py --config_file configs/night600/vit_base_ics_srd_dual_ngtse_V13.yml TEST.WEIGHT "../log/transreid/night600/vit_base_ics_srd_dual_ngtse_V13/transformer_best.pth"
```

Evaluate an ensemble:

```bash
python test_ensemble.py --config_file configs/night600/vit_base_ics_srd_dual_ngtse_V13.yml --weight path/to/checkpoint_a.pth --weight path/to/checkpoint_b.pth --tta-flip --ensemble-mode feature
```

## Synthetic Low-Light Data

The `tools/` directory contains scripts for creating synthetic low-light data and paired GT data:

```bash
python tools/synthesize_lowlight.py --src-root path/to/source_dataset --dark-root data/Syn_dark_market_msmt_v3 --gt-root data/Syn_dark_market_msmt_v3_GT
```

For the merged Market1501+MSMT17 auxiliary set:

```bash
python tools/build_syn_dark_market_msmt_v3.py --market-root path/to/Market-1501-v15.09.15 --msmt-root path/to/MSMT17 --dark-root data/Syn_dark_market_msmt_v3 --gt-root data/Syn_dark_market_msmt_v3_GT
```

## Upstream Projects

This project is based on TransReID and also keeps code from TransReID-SSL, DINO, and cluster-contrast-reid. Please follow the licenses and citation requirements of the upstream projects when using this code.

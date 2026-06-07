# NGL-TransReID Training Code

This directory contains the main training and evaluation code for NGL-TransReID. The implementation starts from TransReID and adds low-light ReID components for Night600 and RGBNT201 RGB-only experiments.

## Main Components

- `model/make_model.py`: model factory, dual-branch transformer, dual local feature fusion, checkpoint loading.
- `model/modules/ng_tse.py`: norm-guided token structure enhancement module.
- `model/backbones/vit_pytorch.py`: ViT/TransReID backbone with optional NG-TSE insertion.
- `processor/processor.py`: training loop with main ReID loss, SRD auxiliary loss, DKD teacher loss, evaluation, and best checkpoint saving.
- `datasets/`: dataset loaders for Market1501, MSMT17, Night600, RGBNT201 RGB-only, and synthetic SRD pairs.
- `train.py`: single-GPU and distributed training entry point.
- `test.py`: single-checkpoint evaluation entry point.
- `test_ensemble.py`: multi-checkpoint feature or distance ensemble evaluation.

## Model Design

The full model is enabled by configs such as:

```text
configs/night600/vit_base_ics_srd_dual_ngtse_V13.yml
configs/rgbnt201/vit_base_ics_rgb_srd_dual_ngtse_V28.yml
```

Important switches:

| Config key | Meaning |
| --- | --- |
| `MODEL.DUAL_BRANCH` | Builds a raw TransReID branch and a structure-enhanced branch. |
| `MODEL.NGTSE` | Enables NG-TSE in the structure branch. |
| `MODEL.NGTSE_MODE` | Chooses token, attention, or combined NG-TSE behavior. |
| `MODEL.NGTSE_LAYERS` | ViT block indices where NG-TSE is inserted. |
| `MODEL.DUAL_LOCAL` | Enables local horizontal part features in both branches. |
| `MODEL.DEVIDE_LENGTH` | Number of local parts, usually 4. |
| `MODEL.RE_ARRANGE` | Controls patch shuffle before local part extraction. |
| `DISTILL.ENABLED` | Enables paired SRD auxiliary training. |
| `DISTILL.DARK_ROOT` | Synthetic dark auxiliary dataset root. |
| `DISTILL.GT_ROOT` | Matched normal-light GT auxiliary dataset root. |
| `DISTILL.TEACHER_WEIGHT` | Teacher checkpoint used for DKD logits. |
| `TEST.WEIGHT` | Checkpoint used by `test.py`. |

When `MODEL.DUAL_BRANCH` is enabled, the student uses:

```text
raw branch feature + NG-TSE branch feature -> fusion -> BNNeck/classifier
```

When `MODEL.DUAL_LOCAL` is also enabled, patch tokens are split into horizontal parts. Local features from the raw and structure branches are fused and concatenated with the global feature for evaluation.

## SRD Distillation

SRD uses paired auxiliary images:

```text
synthetic dark image -> student auxiliary head
normal-light GT pair -> frozen teacher auxiliary head
```

The training loss is:

```text
main ReID loss
+ DISTILL.AUX_LOSS_WEIGHT * auxiliary ReID loss
+ DISTILL.LAMBDA_KD * DKD loss
```

The teacher can be trained with:

```bash
python train.py --config_file configs/night600/vit_base_ics_srd_teacher_gt.yml
```

The resulting teacher path is expected by default at:

```text
../log/transreid/srd_teacher/vit_base_ics_gt/transformer_best.pth
```

## Expected Data Layout

Run training from this directory. Default dataset paths are relative to `transreid_pytorch/`.

Night600:

```text
../data/night600/
|-- bounding_box_train/
|-- query_3/
`-- bounding_box_test/
```

RGBNT201 RGB-only:

```text
../data/RGBNT201/
|-- train_171/RGB/
|-- train_141/RGB/
`-- test/RGB/
```

SRD auxiliary pairs:

```text
../data/Syn_dark_market_msmt_v3/bounding_box_train/
../data/Syn_dark_market_msmt_v3_GT/bounding_box_train/
```

Each dark image must have a GT image with the same filename.

## Training

Train Night600 full model:

```bash
python train.py --config_file configs/night600/vit_base_ics_srd_dual_ngtse_V13.yml
```

Train RGBNT201 RGB-only full model:

```bash
python train.py --config_file configs/rgbnt201/vit_base_ics_rgb_srd_dual_ngtse_V28.yml
```

Train without SRD from a full config:

```bash
python train.py --config_file configs/night600/vit_base_ics_srd_dual_ngtse_V13.yml DISTILL.ENABLED False
```

Change common paths from the command line:

```bash
python train.py --config_file configs/night600/vit_base_ics_srd_dual_ngtse_V13.yml MODEL.PRETRAIN_PATH "../model/vit_base_ics_cfs_lup.pth" DATASETS.ROOT_DIR "../data" OUTPUT_DIR "../log/transreid/night600/custom_run"
```

## Evaluation

Evaluate one checkpoint:

```bash
python test.py --config_file configs/night600/vit_base_ics_srd_dual_ngtse_V13.yml TEST.WEIGHT "../log/transreid/night600/vit_base_ics_srd_dual_ngtse_V13/transformer_best.pth"
```

Evaluate multiple checkpoints by feature ensemble:

```bash
python test_ensemble.py --config_file configs/night600/vit_base_ics_srd_dual_ngtse_V13.yml --weight path/to/checkpoint_a.pth --weight path/to/checkpoint_b.pth --tta-flip --ensemble-mode feature
```

Evaluate multiple checkpoints by distance ensemble:

```bash
python test_ensemble.py --config_file configs/night600/vit_base_ics_srd_dual_ngtse_V13.yml --weight path/to/checkpoint_a.pth --weight path/to/checkpoint_b.pth --ensemble-mode distance
```

## Outputs

Training logs and checkpoints are saved under `OUTPUT_DIR`. The training loop saves periodic checkpoints and updates `transformer_best.pth` when validation mAP/Rank metrics improve.

Generated logs and checkpoints are intentionally ignored by Git.

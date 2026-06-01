# Adapting SAM to Nuclei Instance Segmentation and Classification via Cooperative Fine-Grained Refinement

**Jingze Su<sup>1</sup>, Tianle Zhu<sup>1</sup>, Jiaxin Cai<sup>1</sup>, Zhiyi Wang<sup>1</sup>, Qi Li<sup>1</sup>, Xiao Zhang<sup>1</sup>, Tong Tong<sup>3</sup>, Shu Wang<sup>2</sup><sup>&dagger;</sup>, Wenxi Liu<sup>1</sup><sup>&dagger;</sup>**

<sup>1</sup> College of Computer and Data Science, Fuzhou University, Fuzhou, China  
<sup>2</sup> School of Mechanical Engineering and Automation, Fuzhou University, Fuzhou, China  
<sup>3</sup> College of Physics and Information Engineering, Fuzhou University, Fuzhou, China  
<sup>&dagger;</sup> Corresponding authors

## Requirements

The code was developed with the following core dependencies:

```text
torch 2.0.1
mmcv
mmdet
mmengine
albumentations
opencv-python
scipy
scikit-image
pytorch-toolbelt
prettytable
terminaltables
thop
```

## Datasets

This release is organized for:

- PanNuke
- CPM17
- MoNuSeg

Prepare the data under `datasets/` before training. A typical layout is:

```text
datasets/
  pannuke/
    fold 1/
    fold 2/
    fold 3/
    Images/
    Masks/
  pannuke123_train_files.npy
  pannuke123_val_files.npy
  pannuke123_test_files.npy
  pannuke213_train_files.npy
  pannuke213_val_files.npy
  pannuke213_test_files.npy
  pannuke321_train_files.npy
  pannuke321_val_files.npy
  pannuke321_test_files.npy

  cpm17/
    train/
    test/
  cpm17_train_files.npy
  cpm17_test_files.npy

  monuseg/
    images/
    labels/
  monuseg_train_files.npy
  monuseg_test_files.npy
```

## Training

### 1. Train the Stage 1

PanNuke:

```bash
python main.py --config pannuke123.py --output_dir stage1_pannuke123 --model-ema
# python main.py --config pannuke213.py --output_dir stage1_pannuke213 --model-ema
# python main.py --config pannuke321.py --output_dir stage1_pannuke321 --model-ema
```

CPM17:

```bash
python main.py --config cpm17.py --output_dir stage1_cpm17 --model-ema
```

MoNuSeg:

```bash
python main.py --config monuseg.py --output_dir stage1_monuseg --model-ema
```

### 2. Generate Nucleus Prompts

PanNuke:

```bash
python predict_prompts.py --config pannuke123.py --resume checkpoint/stage1_pannuke123/best.pth
# python predict_prompts.py --config pannuke213.py --resume checkpoint/stage1_pannuke213/best.pth
# python predict_prompts.py --config pannuke321.py --resume checkpoint/stage1_pannuke321/best.pth
```

CPM17:

```bash
python predict_prompts.py --config cpm17.py --resume checkpoint/stage1_cpm17/best.pth
```

MoNuSeg:

```bash
python predict_prompts.py --config monuseg.py --resume checkpoint/stage1_monuseg/best.pth
```

### 3. Train the Phase 2 

Download the SAM pretrained weights from the official Segment Anything release and place them under:

```text
pretrained/
  sam_vit_b_01ec64.pth
  sam_vit_h_4b8939.pth
```

PanNuke:

```bash
python main.py --config pannuke123_h.py --output_dir pannuke123_h
# python main.py --config pannuke213_h.py --output_dir pannuke213_h
# python main.py --config pannuke321_h.py --output_dir pannuke321_h
```

CPM17:

```bash
python main.py --config cpm17_h.py --output_dir cpm17_h
```

MoNuSeg:

```bash
python main.py --config monuseg_h.py --output_dir monuseg_h
```

## Evaluation

Examples:

```bash
python main.py --resume checkpoint/pannuke123_h/best.pth --eval --config pannuke123_h.py
python main.py --resume checkpoint/cpm17_h/best.pth --eval --config cpm17_h.py
python main.py --resume checkpoint/monuseg_h/best.pth --eval --config monuseg_h.py
```

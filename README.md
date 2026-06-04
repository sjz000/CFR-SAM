# Adapting SAM to Nuclei Instance Segmentation and Classification via Cooperative Fine-Grained Refinement

**Jingze Su<sup>1</sup>, Tianle Zhu<sup>1</sup>, Jiaxin Cai<sup>1</sup>, Zhiyi Wang<sup>1</sup>, Qi Li<sup>1</sup>, Xiao Zhang<sup>1</sup>, Tong Tong<sup>3</sup>, Shu Wang<sup>2</sup><sup>&dagger;</sup>, Wenxi Liu<sup>1</sup><sup>&dagger;</sup>**

<sup>1</sup> College of Computer and Data Science, Fuzhou University, Fuzhou, China  
<sup>2</sup> School of Mechanical Engineering and Automation, Fuzhou University, Fuzhou, China  
<sup>3</sup> College of Physics and Information Engineering, Fuzhou University, Fuzhou, China  
<sup>&dagger;</sup> Corresponding authors

🔥🔥[CFR-SAM Paper](https://doi.org/10.1016/j.media.2026.104144) Published in Medical Image Analysis (MIA).  <br>

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

- [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/)
- [CPM17](https://drive.google.com/drive/folders/1sJ4nmkif6j4s2FOGj8j6i_Ye7z9w0TfA?usp=drive_link)
- [MoNuSeg](https://monuseg.grand-challenge.org/Data/)

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

## Model Zoo

### Stage 1 Prompt Learning

| Dataset    | weight |
|:-----------| :------ |
| PanNuke123 | [GoogleDrive](https://drive.google.com/drive/folders/15LuKYCcXOFZa9nyiNWxiURqsIZe6-4B4?usp=drive_link) |
| PanNuke213 | [GoogleDrive](https://drive.google.com/drive/folders/1DgHlSMjNCpnrPY80h5jrv0U6Jt8phhcw?usp=drive_link) |
| PanNuke321 | [GoogleDrive](https://drive.google.com/drive/folders/17ghLhBppLMwPEC4Zq5ueLIZIJ2dDhX65?usp=drive_link) |
| CPM-17     | [GoogleDrive](https://drive.google.com/drive/folders/1vwo5DFORLsYxvJcDtE-sr91Ul3Rt_7y5?usp=drive_link) |
| MoNuSeg    | [GoogleDrive](https://drive.google.com/drive/folders/1lH1d9OMGkBGddH7HdjV5WoRODEd2S-ZP?usp=drive_link) |

### Stage 2 SAM Adaptation

|Dataset     | weight(Ours-H)                                                                                                                                                        |
|:-------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PanNuke123 | [GoogleDrive](https://drive.google.com/drive/folders/1UtO1p6EQ3xN78ixEOu5kj22Q1bEwjrX4?usp=drive_link)                                                                |
| PanNuke213 | [GoogleDrive](https://drive.google.com/drive/folders/14kDPR8Tt821Cjv2IEPUYv1t7Y84JJ4m-?usp=drive_link)                                                                                                                                                       |
| PanNuke321 | [GoogleDrive](https://drive.google.com/drive/folders/1qFjCczm1plsfASr5Q7g_0fFCVcNIw4bf?usp=drive_link)                                                                                                                            |
| CPM-17     | [GoogleDrive](https://drive.google.com/drive/folders/1g-6aH5AxREpyxoZnnj6RVMkFa98v7axA?usp=drive_link)                                                                |
| MoNuSeg    | [GoogleDrive](https://drive.google.com/drive/folders/1QJnZzDDgRwFKxaCti10NfR_EBJaHWjUa?usp=drive_link)                                                                |

|Dataset     | weight(Ours-B)                             |
|:-------------------|:-------------------------------------------|
| PanNuke123 | [GoogleDrive](https://drive.google.com/drive/folders/1WWdat3kNSa-UUfIBhkaZb2Bcw_SfD-45?usp=drive_link) |
| PanNuke213 | [GoogleDrive](https://drive.google.com/drive/folders/11nTFqLyuoomSru80abInamzxzs00GZR3?usp=drive_link) |
| PanNuke321 | [GoogleDrive](https://drive.google.com/drive/folders/1aItkKgcyvc1DQc8uRDIoHsZ1d09PfzzS?usp=drive_link) |

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

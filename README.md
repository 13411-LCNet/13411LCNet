# [LCNet: Learnable Label Correlation Network for Multi-Label ImageClassification]

## Abstract
In computer vision, multi-label image classification is a well known topic which can be applied in many real-world applications. 
In classical multi-label image classification, only the spatial features of images are used as input of classification models. 
In this paper we introduce LCNet: A Multi-Label Classification model, which includes a learnable semantic graph followed by a 
novel decoder to fuse the multiple modalities. Our decoder is stack-able which allows the model to understand deeper relations 
between the modalities. In addition, we design a feature to reduce spatial resolution loss called Crop-Forwarding. Crop-Forwarding 
allows higher resolution images to be forwarded without using higher resolution spatial feature extractors. Extensive experiments 
conducted on several multi-label classification benchmarks, Pascal-VOC and MS-COCO, demonstrate that our solution significantly 
improved the state-of-the-art results. Our proposed method achieves mAP 91.5% on MSCOCO and 97.7% on Pascal-VOC. Especially, our model
outperforms the state-of-the-art models on a small datasets such as the synthetic-fiber rope damage dataset, resulting in
a new top score of 88.2%.

![overview](https://github.com/13411-LCNet/13411LCNet/blob/92449b65c275fb785152f816367da60151e9b686/ModelOverview.jpg)

## Installation

Tested on Ubuntu only.

**Prerequisite:**

- Python 3.11.9+
- PyTorch 2.4+ and corresponding torchvision

Install [```cuda```](https://developer.nvidia.com/cuda-downloads), [```PyTorch``` and ```torchvision```](https://pytorch.org/).

**Clone our repository:**

```bash
git clone https://github.com/13411LCNet/13411-LCNet.git
```

**Install with pip requirements:**

```bash
cd 13411LCNet
pip install -r requirements.txt .
```


**Download the COCO2014 dataset:**
Download [MS-COCO 2014](https://cocodataset.org/#download).

## Quick Start

### Start the COCO Training with:

```bash
python train.py --print-freq 1000 --print_freq_val 1000 --batch-size 8 --dataname coco14 --num_class 80 --isTrain True --pretrained --backbone TResnetL_V2 --CropLevels 2 --imgInpsize 896 --dataset_dir <Insert datasetDIR here> --output <Insert Output folder here> --resume <Insert pretrained model folder here> 
```

### Start the VOC2007 Training with:

```bash
python train.py --print-freq 1000 --print_freq_val 1000 --batch-size 8 --dataname voc --num_class 20 --isTrain True --pretrained --backbone TResnetL_V2 --CropLevels 3 --imgInpsize 1344 --dataset_dir <Insert datasetDIR here> --output <Insert Output folder here> --resume <Insert pretrained model folder here> 
```

### Start the COCO Verification with:

```bash
python train.py --print-freq 1000 --print_freq_val 1000 --batch-size 8 --dataname coco14 --num_class 80 --isTrain False --pretrained --backbone TResnetL_V2 --CropLevels 2 --imgInpsize 896 --dataset_dir <Insert datasetDIR here> --output <Insert Output folder here> --resume <Insert pretrained model folder here> 
```

### Start the VOC2007 Verification with:

```bash
python train.py --print-freq 120 --print_freq_val 120 --batch-size 8 --dataname voc --num_class 20 --isTrain False --pretrained --backbone TResnetL_V2 --CropLevels 3 --imgInpsize 1344 --dataset_dir <Insert datasetDIR here> --output <Insert Output folder here> --resume <Insert pretrained model folder here
```




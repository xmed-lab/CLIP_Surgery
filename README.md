# CLIP Surgery for Better Explainability with Enhancement in Open-Vocabulary Tasks

## Introduction

This work focuses on the explainability of CLIP via its raw predictions. We identify two problems about CLIP's explainability: opposite visualization and noisy activations. Then we propose the CLIP Surgery, which does not require any fine-tuning or additional supervision. It greatly improves the explainability of CLIP, and enhances downstream open-vocabulary tasks such as multi-label recognition, semantic segmentation, interactive segmentation (specifically the Segment Anything Model, SAM), and multimodal visualization. Currently, we offer a simple demo for interpretability analysis, and how to convert text to point prompts for SAM. Rest codes including evaluation and other tasks will be released later.

## Demo

Firstly to install the SAM, and download the model
```
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Then explain CLIP with SAM via jupyter demo "demo.ipynb", or using python file
```
python demo.py
```

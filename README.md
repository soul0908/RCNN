# RCNN

This project contains an end-to-end pipeline for training and evaluating a Region-based Convolutional Neural Network (RCNN) on the PASCAL VOC dataset.

## Overview

The main script `rcnn.py` implements the following stages:

1. **Proposal Generation** - Uses Selective Search to produce region proposals for every image.
2. **Index Creation** - Matches proposals with ground truth boxes to label positive and negative samples.
3. **Model Training** - Trains a custom AlexNet classifier on cropped proposal patches.
4. **Feature Extraction** - Extracts features for all proposals and stores them in an HDF5 file.
5. **SVM Training** - Trains a linear SVM for each object class using the extracted features.
6. **Bounding Box Regression** - Learns regressors to refine proposal boxes.
7. **Inference & Evaluation** - Applies the trained models, performs non-maximum suppression, and computes mAP.

The code expects the VOC2007 dataset under `VOC2007/` and caches intermediate data in `RCNNDataCache/`.

Run the entire pipeline with:

```bash
python rcnn.py
```

The script will automatically generate proposals and train models if necessary. Results such as trained SVMs and extracted features are stored in the cache directory so repeated runs reuse previous outputs.

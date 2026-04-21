---

# Boschnew

A physics-informed learning framework for Bosch DRIE morphology prediction and few-shot transfer.

## Environment

* PyCharm 2025.3
* NVIDIA RTX 5060
* Python 3.12

## Data

* Simulation case table: `case.csv`
* Processed Phys7 table: `case_with_phys7.xlsx`

Simulated physical-parameter data can be downloaded from:

* Baidu Netdisk: [https://pan.baidu.com/s/1XBAgZSYUaX4OD8k81K9GTQ?pwd=e6uw](https://pan.baidu.com/s/1XBAgZSYUaX4OD8k81K9GTQ?pwd=e6uw)
* Extraction code: `e6uw`

## Overview

This repository implements a staged learning pipeline for Bosch deep reactive ion etching (DRIE), aiming to bridge **process recipes**, **IEDF-derived physical descriptors**, and **cycle-resolved morphology prediction**.

The main workflow includes:

* **Stage A**: recipe → Phys7 surrogate learning
* **Stage B**: Phys7-conditioned morphology prediction
* **Stage C**: few-shot transfer to sparse experimental metrology

## Pipeline

### Stage A: recipe → Phys7

`stageA_train_phys_pycharm.py` learns the mapping from the 7-dimensional recipe vector

`[APC, source_RF, LF_RF, SF6, C4F8, DEP time, etch time]`

to the 7-dimensional Phys7 target.

### Stage B: Phys7-conditioned morphology prediction

`stageB_train_morph_on_phys7_pycharm.py` predicts cycle-resolved morphology using:

* recipe-derived static features
* Phys7 descriptors
* cycle / time-step information

### Stage C: few-shot transfer

`stageC_finetune_joint.py` adapts the pretrained morphology model to sparse experimental metrology with masked supervision and transfer learning.

## Repository Structure

```text
Boschnew/
├── case.csv
├── case_with_phys7.xlsx
├── extract_phys7_from_iedf.py
├── phys_model.py
├── physio_util.py
├── stage0_train_iedf_ae.py
├── stageA_train_phys_pycharm.py
├── stageB_train_morph_on_phys7_pycharm.py
├── stageB_util.py
└── stageC_finetune_joint.py
```


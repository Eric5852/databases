# MACNet：Multiscale attention cross-sharing networks for Colorectal polyp segmentation

> **Authors:** 
> [Xinhui Jiang]
> [chunmiao Wei]
> [Xiaolin Li]




## 1. Preface

- This repository provides code for "_**Multiscale attention cross-sharing networks for Colorectal polyp segmentation (MACNet)**_". 
([paper provide later]())([code](https://drive.google.com/drive/folders/1CfMLCVIsGxsnCMcM7iN5q8W-fq8PBwh1?usp=sharing))

- If you have any questions about our paper, feel free to contact me.

### 1.1. :fire: NEWS :fire:

- [2024/11/4] Release training/testing code.

- [2020/11/4] Create repository.


## 2. Proposed Baseline

### 2.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single NVIDIA GeForce RTX 3090 with 24 GB Memory.

> Note that our model also supports low memory GPU, which means you can lower the batch size


1. Configuring your environment (Prerequisites):
   
    Note that MACNetis only tested on Win10 OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n MACNet python=3.9`.
    
    + Installing necessary packages: PyTorch 2.0.0

2. Downloading necessary data:

    + downloading testing dataset and move it into `./data/TestDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1hwirZO201i_08fFgqmeqMuPuhPboHdVH/view?usp=sharing). It contains five sub-datsets: CVC-300 or referred to as Endoscene (60 test samples),CVC612 or referred to as ClinicDB (62 test samples), CVC-ColonDB (380 test samples), ETIS-LaribPolypDB (196 test samples), Kvasir (100 test samples).
    
    + downloading training dataset and move it into `./data/TrainDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1hzS21idjQlXnX9oxAgJI8KZzOBaz-OWj/view?usp=sharing). It contains two sub-datasets: Kvasir-SEG (900 train samples) and CVC-ClinicDB (550 train samples).
    
    + downloading pretrained weights and move it into `weights/MACNet/MACNet.pt`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/drive/folders/1XX2dOM8HLex5T9w2dgIjN4Kgs-U_rQSp?usp=sharing).
    
    + downloading pvt_v2 weights and and move it into `./lib/`, "and modify the encoder path", 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1OEebKrIAXZ1k7WpDLkClVQX4eL4kbel-/view?usp=sharing).
   
3. Training Configuration:

    + Assigning your costumed path, like `--save_model` and `--train_path` in `train.py`.
    
    + Enjoy the process!

4. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `test.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).
    
    + Enjoy the process!

### 2.2 Evaluating your trained model:


1 Evaluating Your Trained Model: You can use the Python program adapted from MATLAB in the repository to evaluate your trained model.
After downloading the predicted results and labels, set your custom paths and other configurations in "Eval.py," "utils/utils.py," and utils/MACNet.yaml. Running Eval.py will then provide an accurate evaluation of the model's predictive performance across various metrics.



or use the evaluation method provided by MATLAB
Matlab: One-key evaluation is written in MATLAB code ([link](https://drive.google.com/file/d/1_h4_CjD5GKEf7B1MRuzye97H0MXf2GE9/view?usp=sharing)), 
please follow this the instructions in `./eval/main.m` and just run it to generate the evaluation results in `./res/`.
Enjoy the process!

### 2.3 Pre-computed maps: 
They can be found in [download link](https://drive.google.com/drive/folders/1BQjwxqWTltNaAPMIT6RRHncgODtQd12H?usp=sharing).




## 3. License

The source code is free for research and education use only. Any comercial use should get formal permission first.

---

**[⬆ back to top](#0-preface)**

# Class-Incremental Learning with Cross-Space Clustering and Controlled Transfer

The official implementation of our ECCV 2022 paper "Class-Incremental Learning with Cross-Space Clustering and Controlled Transfer".
[[Paper]](https://arxiv.org/abs/2208.03767) [[Project Page]](https://cscct.github.io)

## Getting Started

In order to run this repository, we advise you to install python 3.6 and PyTorch 1.2.0 with Anaconda.

You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install PyTorch and torchvision on it:

```bash
conda create --yes --name CSCCT-PyTorch python=3.6
conda activate CSCCT-PyTorch
conda install --yes pytorch=1.2.0 
conda install --yes torchvision -c pytorch
```

Install other requirements:
```bash
pip install tqdm scipy sklearn tensorboardX Pillow==6.2.2
```

## Running Experiments

### Baselines

```bash
python main.py --nb_cl_fg=INITIAL_TASK_SIZE --nb_cl=TASK_SIZE --gpu=GPU --random_seed=1993 --baseline=BASELINE --branch_mode=single --branch_1=free --dataset=DATASET
```

The above script can be used, replacing 

`INITIAL_TASK_SIZE` with the number of classes in the first task (given as $\mathcal{B}$ in the paper),

`TASK_SIZE` with the number of classes in every subsequent task (given as $\mathcal{C}$ in the paper),

`BASELINE` with either `'lucir'` or `'icarl'`,

`DATASET` with either `'cifar100` or `'imagenet_sub'`,

`GPU` with the GPU to run the model in.

### Baselines + CSCCT

To add cross-space clustering and controlled transfer (CSCCT) to the baselines, follow the below directions.

To add **cross-space clustering (CSC)**, add the additional flags 

```bash
--csc --csc_weight SPECIFY_CSC_WEIGHT
```

replacing `SPECIFY_CSC_WEIGHT` with the appropriate weight for the CSC objective. The default value for `csc_weight` is $3$.

To add **controlled transfer(CT)**, add the additional flags

```
--ct --ct_weight SPECIFY_CT_WEIGHT --ct_temperature SPECIFY_CT_TEMP
```

replacing `SPECIFY_CT_WEIGHT` with the appropriate weight for the CT objective, and `SPECIFY_CT_TEMP` with the temperature. 

The default value for `ct_weight` is $1.5$, and default value for `ct_temperature` is $2$. 


### Note on datasets

CIFAR100 is automatically downloaded to `./data`; the directory can be changed using the flag `--data_dir PATH`.

ImageNet-Subset is assumed to be present at `./data/imagenet_sub`; the parent directory (`./data`) can be changed using the flag `--data_dir PATH`.

To download ImageNet-Subset, the full ImageNet dataset from [the official ImageNet website](https://image-net.org/) (note: requires login) must be first downloaded. Then, the 100-class [train](https://github.com/arthurdouillard/incremental_learning.pytorch/blob/master/imagenet_split/train_100.txt) and [val](https://github.com/arthurdouillard/incremental_learning.pytorch/blob/master/imagenet_split/val_100.txt) splits (taken from the [codebase of PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch)) should be used to remove the other 900 classes and preprocess the data.

### Running Experiments on ImageNet

To run the experiments on ImageNet-Subset, you need to change the hyperparameters according to [this file](https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/imagenet-class-incremental/cbf_class_incremental_cosine_imagenet.py). 

## Bibtex

If you find this code useful, please cite our work:
```
@article{ashok2022class, 
title={Class-Incremental Learning with Cross-Space Clustering and Controlled Transfer}, 
author={Ashok, Arjun and Joseph, KJ and Balasubramanian, Vineeth}, 
journal={arXiv preprint arXiv:2208.03767}, year={2022} }
```

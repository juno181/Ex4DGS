<h2 align="center">Fully Explicit Dynamic Gaussian Splatting</h2>
<p align="center">
  <a href="https://leejunoh.com/"><strong>Junoh Lee</strong></a>
  ·  
  <strong>ChangYeon Won</strong>
  ·  
  <strong>Hyunjun Jung</strong>
  ·
  <a href="https://ihbae.com/"><strong>Inhwan Bae</strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=Ei00xroAAAAJ"><strong>Hae-Gon Jeon</strong></a>
  <br>
  NeurIPS 2024
</p>

<p align="center">
  <a href="https://leejunoh.com/Ex4DGS/"><strong><code>Project Page</code></strong></a>
  <a href="https://neurips.cc/virtual/2024/poster/94164"><strong><code>NeurIPS Paper</code></strong></a>
  <a href="http://arxiv.org/abs/2410.15629"><strong><code>Arxiv Paper</code></strong></a>
  <a href="https://github.com/juno181/Ex4DGS"><strong><code>Source Code</code></strong></a>
  <a href="#-citation"><strong><code>Related Works</code></strong></a>
</p>

<div align='center'>
  <br><img src="img/Ex4DGS-thumbnail.gif" width=70%>
  <!--<img src="img/eigentrajectory-model.svg" width=70%>-->
  <br>A novel view synthesis result of Ex4DGS.
</div>

<br>**Summary**: **4D Gaussian Splatting** with **static & dynamic separation** using an incrementally **extensible**, **keyframe**-based model

<br>

## Contents

1. [Setup](#-Setup)
2. [Preprocess Datasets](#-Preprocess-Datasets)
3. [Training](#-Training)
4. [Evaluation](#-Evaluation)
5. [Pretrained models](#-Pretrained-models)


## Setup

### Environment Setup

Clone the source code of this repo.
```shell
git clone https://github.com/juno181/Ex4DGS.git
cd Ex4DGS
```

Installation through pip is recommended. First, set up your Python environment:
```shell
conda create -n Ex4DGS python=3.9
conda activate Ex4DGS
```

Make sure to install CUDA and PyTorch versions that match your CUDA environment. We've tested on RTX 4090 GPU with PyTorch  version 2.12.
Please refer https://pytorch.org/ for further information.

```shell
pip install torch
```

The remaining packages can be installed with:

```shell
pip install --upgrade setuptools cython wheel
pip install -r requirements.txt
```

<!-- Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate Ex4DGS
``` -->

## Preprocess Datasets

For dataset preprocessing, we follow [STG](https://github.com/oppo-us-research/SpacetimeGaussians.git).

### Neural 3D Video Dataset
First, download the dataset from [here](https://github.com/facebookresearch/Neural_3D_Video). You will need colmap environment for preprocess.
To setup dataset preprocessing environment, run scrips:
```shell
./scripts/env_setup.sh
```

To preprocess dataset, run script:
```shell
./scripts/preprocess_all_n3v.sh <path to dataset>
```

### Technicolor dataset
Download the dataset from [here](https://www.interdigital.com/data_sets/light-field-dataset).
To setup dataset preprocessing environment, run scrips:

```shell
./scripts/preprocess_all_techni.sh <path to dataset>
```

Please refer [STG](https://github.com/oppo-us-research/SpacetimeGaussians.git) for further information.

## Training

Run command:
```shell
python train.py --config configs/<some config name>.json --model_path <some output folder>  --source_path <path to dataset>
```

## Evaluation

Run command:
```shell
python render.py --model_path <path to trained model>  --source_path <path to dataset> --skip_train --iteration <trained iter>
```

## Pretrained models

We provide pretrained models in [release](https://github.com/juno181/Ex4DGS/releases/tag/v0.1).

## 📖 Citation
<!-- If you find this code useful for your research, please cite our trajectory prediction papers :) -->

[**`🛍️ Ex4DGS (NeurIPS'24) 🛍️`**](https://github.com/juno181/Ex4DGS) **|**

```bibtex
@inproceedings{lee2024ex4dgs,
  title={Fully Explicit Dynamic Guassian Splatting},
  author={Lee, Junoh and Won, ChangYeon and Jung, Hyunjun and Bae, Inhwan and Jeon, Hae-Gon},
  booktitle={Proceedings of the Neural Information Processing Systems},
  year={2024}
}
```

<br>
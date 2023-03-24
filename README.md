QTClassification
========

**A lightweight and extensible toolbox for image classification**

[![version](https://img.shields.io/badge/Version-0.2.0-brightgreen)](https://github.com/horrible-dong/QTClassification)
&emsp;[![docs](https://img.shields.io/badge/Docs-Latest-orange)](https://github.com/horrible-dong/QTClassification/blob/main/README.md)
&emsp;[![license](https://img.shields.io/badge/License-Apache--2.0-blue)](https://github.com/horrible-dong/QTClassification/blob/main/LICENSE)

> Author: QIU, Tian  
> Affiliate: Zhejiang University  
> <a href="#installation">🛠️ Installation</a> | <a href="#getting_started">📘
> Documentation </a> | <a href="#dataset_zoo">🌱 Dataset Zoo</a> | <a href="#model_zoo">👀 Model Zoo</a>  
> English | [简体中文](README_zh-CN.md)

## <span id="Installation">Installation</span>

Our development environment is `python 3.7 & pytorch 1.11.0+cu113`.

1. Create your conda environment if needed.

```bash
conda create -n qtcls python==3.7 -y
```

2. Enter your conda environment.

```bash
conda activate qtcls
```

3. Install PyTorch.

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

or you can refer to [PyTorch](https://pytorch.org/) to install newer or older versions. We strongly recommend you
use `pytorch >= 1.11.0` for its less GPU memory consumption.

4. Install the necessary dependencies.

```bash
pip install -r requirements.txt
```

## <span id="getting_started">Getting Started</span>

For a quick experience, you can directly run the following commands:

**Training**

```bash
# multi-gpu (recommend, needs pytorch>=1.9.0)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --epochs 50 \
  --output_dir ./runs/__tmp__
  
# multi-gpu (for any pytorch version)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --epochs 50 \
  --output_dir ./runs/__tmp__
  
# single-gpu
python main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --epochs 50 \
  --output_dir ./runs/__tmp__
```

The `cifar10` dataset and `resnet50` pretrained weights will be automatically downloaded. The `cifar10` dataset will be
downloaded to `./data`. During the training, the checkpoints, logs and other outputs will be stored in `./runs/__tmp__`.

**Evaluation**

```bash
# multi-gpu (recommend, needs pytorch>=1.9.0)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --resume ./runs/__tmp__/checkpoint.pth \
  --eval
  
# multi-gpu (for any pytorch version)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --resume ./runs/__tmp__/checkpoint.pth \
  --eval
  
# single-gpu
python main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --resume ./runs/__tmp__/checkpoint.pth \
  --eval
```

### How to use

When using our toolbox for training and evaluation, you can run the commands we provided above *with your own
arguments*.

**Frequently-used command line arguments**

|  Command Line Arguments  |                                                                                             Description                                                                                              |  Default value   |
|:------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------:|
|      `--data_root`       |                                                                               Directory where your datasets is stored.                                                                               |     `./data`     |
|  `--dataset`<br />`-d`   |                                      Dataset name defined in [qtcls/datasets/\_\_init\_\_.py](qtcls/datasets/__init__.py), such as `cifar10` and `imagenet1k`.                                       |        /         |
|      `--model_lib`       |                                  Model library where models come from. Our basic model library is extended from `torchvision` (default), and also supports `timm`.                                   | `torchvision-ex` |
|   `--model`<br />`-m`    | Model name defined in [qtcls/models/\_\_init\_\_.py](qtcls/models/__init__.py), such as `resnet50` and `vit_b_16`. Currently supported model names are listed in <a href="#model_zoo">Model Zoo</a>. |        /         |
|      `--criterion`       |                         Criterion name defined in [qtcls/criterions/\_\_init\_\_.py](qtcls/criterions/__init__.py). The `default` criterion computes the cross entropy loss.                         |    `default`     |
|      `--optimizer`       |                                        Optimizer name defined in [qtcls/optimizers/\_\_init\_\_.py](qtcls/optimizers/__init__.py), such as `sgd` and `adam`.                                         |    `default`     |
|      `--scheduler`       |                                            Scheduler name defined in [qtcls/schedulers/\_\_init\_\_.py](qtcls/schedulers/__init__.py), such as `cosine`.                                             |    `default`     |
|      `--evaluator`       |             Evaluator name defined in [qtcls/evaluators/\_\_init\_\_.py](qtcls/evaluators/__init__.py). The `default` evaluator computes the accuracy, recall, precision, and f1_score.              |    `default`     |
|   `--resume`<br />`-r`   |                                                                                   Checkpoint path to resume from.                                                                                    |        /         |
| `--output_dir`<br />`-o` |                                                                       Path to store your checkpoints, logs, and other outputs.                                                                       | `./runs/__tmp__` |
|          `--lr`          |                                                                                            Learning rate.                                                                                            |      `1e-4`      |
|        `--epochs`        |                                                                                                  /                                                                                                   |       `50`       |
| `--batch_size`<br />`-b` |                                                                                                  /                                                                                                   |       `8`        |
|         `--eval`         |                                                                                    To evaluate without training.                                                                                     |     `False`      |

**How to put your dataset**

Currently, `mnist`, `cifar10` and `cifar100` datasets will be automatically downloaded to the `--data_root` directory.
For other datasets, please refer to [this instruction](data/README.md)。

### How to customize

Our toolbox is flexible enough to be extended. Please follow the instructions below:

[How to register your datasets](qtcls/datasets/README.md)

[How to register your models](qtcls/models/README.md)

[How to register your criterions](qtcls/criterions/README.md)

[How to register your optimizers](qtcls/optimizers/README.md)

[How to register your schedulers](qtcls/schedulers/README.md)

[How to register your evaluators](qtcls/evaluators/README.md)

## <span id="dataset_zoo">Dataset Zoo</span>

Currently supported argument `--dataset`:  
`mnist`, `cifar10`, `cifar100`, `imagenet1k` and all datasets in `folder` format (consistent with `imagenet` storage
format, that is, images of each category are stored in a folder/directory, and the folder/directory name is the category
name).

Scheduled:  
`stl10`, `svhn`, `pets` ...

## <span id="model_zoo">Model Zoo</span>

Our basic model library is extended from `torchvision` (default), and also supports `timm`.

### torchvision (extended)

Set the argument `--model_lib` to `torchvision-ex`.

Currently supported argument `--model`:

**AlexNet**  
`alexnet`

**ConvNext**  
`convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`

**DenseNet**  
`densenet121`, `densenet169`, `densenet201`, `densenet161`

**EfficientNet**  
`efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`, `efficientnet_b4`, `efficientnet_b5`, `efficientnet_b6`, `efficientnet_b7`

**GoogLeNet**  
`googlenet`

**Inception**    
`inception_v3`

**MNASNet**   
`mnasnet0_5`, `mnasnet0_75`, `mnasnet1_0`, `mnasnet1_3`

**MobileNet**  
`mobilenet_v2`, `mobilenetv3`, `mobilenet_v3_large`, `mobilenet_v3_small`

**RegNet**  
`regnet_y_400mf`, `regnet_y_800mf`, `regnet_y_1_6gf`, `regnet_y_3_2gf`, `regnet_y_8gf`, `regnet_y_16gf`, `regnet_y_32gf`, `regnet_y_128gf`, `regnet_x_400mf`, `regnet_x_800mf`, `regnet_x_1_6gf`, `regnet_x_3_2gf`, `regnet_x_8gf`, `regnet_x_16gf`, `regnet_x_32gf`

**ResNet**     
`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnext50_32x4d`, `resnext101_32x8d`, `wide_resnet50_2`, `wide_resnet101_2`

**ShuffleNet**  
`shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0`, `squeezenet1_0`, `squeezenet1_1`

**VGG**  
`vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`

**Vision Transformer**  
`vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32`

### timm

Set the argument `--model_lib` to `timm`.

Currently supported argument `--model`:  
All supported. Please refer to `timm` for the specific model name.

## LICENSE

QTClassification is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

Copyright (c) QIU, Tian. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with
the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "
AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

## Citation

If you find QTClassification Toolbox useful in your research, please consider citing:

```bibtex
@misc{2023QTClassification,
    title={QTClassification},
    author={QTClassification Contributors},
    howpublished = {\url{https://github.com/horrible-dong/QTClassification}},
    year={2023}
}
```

QTClassification<sup>Dev</sup>
========

> **An elegant toolbox for image classification**   
> Author: QIU, Tian   
> Affiliate: Zhejiang University

## Installation

Our development environment is `python 3.7 & pytorch 1.11.0+cu113`.

1. Create your conda environment if needed.

```
conda create -n qtcls python==3.7 -y
```

2. Enter your conda environment.

```
conda activate qtcls
```

3. Install PyTorch.

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

or you can refer to [PyTorch](https://pytorch.org/) to install newer or older versions. We strongly recommend you
use `pytorch >= 1.11.0` for its less GPU memory consumption.

4. Install the necessary dependencies.

```
pip install -r requirements.txt
```

## Getting Started

For quick use, you can directly run the following commands:

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

*To be continued ...*

## Model Zoo

Our basic model library is based on `torchvision` (default), and also supports `timm`.

### torchvision

Set the argument`--model_lib` to `torchvision` (default).

Currently supported argument `--model`:

**AlexNet**

`alexnet`

**ConvNext**

`convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`

**DenseNet**

`densenet`, `DenseNet`, `densenet121`, `densenet169`, `densenet201`, `densenet161`

**EfficientNet**

`efficientnet`, `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`, `efficientnet_b4`, `efficientnet_b5`, `efficientnet_b6`, `efficientnet_b7`

**GoogleNet**

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

`shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0` `squeezenet1_0`, `squeezenet1_1`

**VGG**

`vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`

**Vision Transformer**

`vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32`

### timm

Set the argument`--model_lib` to `timm`.

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
    title={QTClassification: An Elegant Toolbox for Image Classification},
    author={QTClassification Contributors},
    howpublished = {\url{https://github.com/horrible-dong/QTClassification}},
    year={2023}
}
```


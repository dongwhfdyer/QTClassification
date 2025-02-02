QTClassification
========

**一个轻量可扩展的图像分类工具箱**

[![version](https://img.shields.io/badge/Version-0.2.0-brightgreen)](https://github.com/horrible-dong/QTClassification)
&emsp;[![docs](https://img.shields.io/badge/Docs-Latest-orange)](https://github.com/horrible-dong/QTClassification/blob/main/README_zh-CN.md)
&emsp;[![license](https://img.shields.io/badge/License-Apache--2.0-blue)](https://github.com/horrible-dong/QTClassification/blob/main/LICENSE)

> 作者: QIU, Tian  
> 机构: 浙江大学  
> <a href="#安装">🛠️ 安装</a> | <a href="#使用教程">📘 使用教程</a> | <a href="#数据集">🌱 数据集</a> | <a href="#模型库">
> 👀 模型库</a>  
> [English](README.md) | 简体中文

## 安装

我们的开发环境是 `python 3.7 & pytorch 1.11.0+cu113`.

1. 如果需要的话，创建你的conda环境。

```bash
conda create -n qtcls python==3.7 -y
```

2. 进入你的conda虚拟环境。

```bash
conda activate qtcls
```

3. 安装 PyTorch。

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

或者你可以参考[PyTorch官网](https://pytorch.org/)来安装其他版本. 我们强烈推荐你使用`pytorch >= 1.11.0`，因为新版本的显存开销更小。

4. 安装必要的依赖。

```bash
pip install -r requirements.txt
```

## 使用教程

想要快速体验，你可以直接执行以下命令：

**训练**

```bash
# 多gpu (推荐, 需要pytorch版本>=1.9.0)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --epochs 50 \
  --output_dir ./runs/__tmp__
  
# 多gpu (适用于任何pytorch版本)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --epochs 50 \
  --output_dir ./runs/__tmp__
  
# 单gpu
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --epochs 50 \
  --output_dir ./runs/__tmp__
```

`cifar10` 数据集和 `resnet50` 预训练权重会自动下载。`cifar10` 数据集会被下载到 `./data`
目录下。在训练过程中，checkpoint文件（包含模型权重、优化器权重等）、日志文件和其他输出都会被存放在 `./runs/__tmp__` 目录下。

**验证**

```bash
# 多gpu (推荐, 需要pytorch版本>=1.9.0)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --resume ./runs/__tmp__/checkpoint.pth \
  --eval
  
# 多gpu (适用于任何pytorch版本)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --resume ./runs/__tmp__/checkpoint.pth \
  --eval
  
# 单gpu
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --resume ./runs/__tmp__/checkpoint.pth \
  --eval
```

### 如何使用

使用我们的工具箱进行训练和验证时，你可以参照上述命令执行，不过你需要修改命令行参数。

**常用的命令行参数**

|        命令行参数        |                             描述                             |      默认值      |
| :----------------------: | :----------------------------------------------------------: | :--------------: |
|      `--data_root`       |                    你的数据集存放的路径。                    |     `./data`     |
|  `--dataset`<br />`-d`   | 数据集名称，在 [qtcls/datasets/\_\_init\_\_.py](qtcls/datasets/__init__.py) 里定义，如 `cifar10` 和 `imagenet1k`。 |        /         |
|      `--model_lib`       | 模型库，模型都取自模型库。我们的基础模型库由 `torchvision` 扩展而来 (我们的默认模型库)，同时我们也支持 `timm` 模型库。 | `torchvision-ex` |
|   `--model`<br />`-m`    | 模型名称，在 [qtcls/models/\_\_init\_\_.py ](qtcls/models/__init__.py) 里定义，如 `resnet50` 和 `vit_b_16`。目前支持的模型名称在<a href="#模型库">模型库</a>中列出。 |        /         |
|      `--criterion`       | 损失函数名称，在 [qtcls/criterions/\_\_init\_\_.py](qtcls/criterions/__init__.py) 里定义。默认的损失函数会计算交叉熵损失。 |    `default`     |
|      `--optimizer`       | 优化器名称，在 [qtcls/optimizers/\_\_init\_\_.py](qtcls/optimizers/__init__.py)，如 `sgd` 和 `adam`。 |    `default`     |
|      `--scheduler`       | 学习率调整策略名称，在 [qtcls/schedulers/\_\_init\_\_.py](qtcls/schedulers/__init__.py) 中定义，如 `cosine`。 |    `default`     |
|      `--evaluator`       | 验证器名称，在 [qtcls/evaluators/\_\_init\_\_.py](qtcls/evaluators/__init__.py) 中定义。默认的验证器会计算准确率、召回率、精确率和f1分数。 |    `default`     |
|   `--resume`<br />`-r`   |                   要从中恢复的检查点路径。                   |        /         |
| `--output_dir`<br />`-o` | 输出目录，用来存放checkpoint文件（包含模型权重、优化器权重等）、日志文件和其他输出。 | `./runs/__tmp__` |
|          `--lr`          |                           学习率。                           |      `1e-4`      |
|        `--epochs`        |                              /                               |       `50`       |
| `--batch_size`<br />`-b` |                              /                               |       `8`        |
|         `--eval`         |                       只验证，不训练。                       |     `False`      |

**如何放置你的数据集**

目前，`mnist`、`cifar10` 和 `cifar100` 数据集会自动下载到 `--data_root`
目录下。其余数据集请参考[“如何放置你的数据集”](data/README_zh-CN.md)。

### 如何自定义

你可以很轻松地对我们的工具箱进行扩展，请参考以下文档：

[如何注册你的数据集](qtcls/datasets/README_zh-CN.md)

[如何注册你的模型](qtcls/models/README_zh-CN.md)

[如何注册你的损失函数](qtcls/criterions/README_zh-CN.md)

[如何注册你的优化器](qtcls/optimizers/README_zh-CN.md)

[如何注册你的学习率调整策略](qtcls/schedulers/README_zh-CN.md)

[如何注册你的验证器](qtcls/evaluators/README_zh-CN.md)

## <span id="数据集">数据集</span>

目前支持的 `--dataset` 参数:  
`mnist`, `cifar10`, `cifar100`, `imagenet1k` 以及所有 `folder` 格式的数据集（与 `imagenet`
存储格式一致，即每个类别的图片存放在一个文件夹内，文件夹名称是类别名称）。

计划支持的数据集：  
`stl10`, `svhn`, `pets` ...

## <span id="模型库">模型库</span>

我们的基础模型库由 `torchvision` 扩展而来 (我们的默认模型库)，同时我们也支持 `timm` 模型库。

### torchvision（经过我们扩展的）

把 `--model_lib` 赋值为 `torchvision-ex`。

目前支持的 `--model` 参数：

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

把 `--model_lib` 赋值为 `timm`。

目前支持的 `--model` 参数：  
全部支持。 具体模型名称请参考 `timm`。

（注：国内自动下载  `timm` 权重需要翻墙）

## 开源许可证

QTClassification 基于 Apache 2.0 开源许可证. 具体请看[开源许可证](LICENSE)。

版权属于 QIU, Tian，并保留所有权利。

## 引用

如果你觉得我们的 “QTClassification工具箱” 对你有帮助，欢迎引用：

```bibtex
@misc{2023QTClassification,
    title={QTClassification},
    author={QTClassification Contributors},
    howpublished = {\url{https://github.com/horrible-dong/QTClassification}},
    year={2023}
}
```

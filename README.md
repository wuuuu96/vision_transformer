# Vision Transformer 和 MLP-Mixer 架构

在本仓库中，我们发布了这些论文中所使用的模型。

- (ViT) [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- (MLP-Mixer) [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
- [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270)
- [When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations](https://arxiv.org/abs/2106.01548)
- (LiT) [LiT: Zero-Shot Transfer with Locked-image text Tuning](https://arxiv.org/abs/2111.07991)
- [Surrogate Gap Minimization Improves Sharpness-Aware Training](https://arxiv.org/abs/2203.08065)


这些模型在 [ImageNet](http://www.image-net.org/) 和 [ImageNet-21k](http://www.image-net.org/) 数据集上进行了预训练。

使用 [JAX](https://jax.readthedocs.io) 和 [Flax](http://flax.readthedocs.io) 框架编写的源代码，并且在已有预训练模型的基础上继续微调训练，以适配新的任务或数据集。



这些模型最初是在以下代码库中训练的：👉 https://github.com/google-research/big_vision/

在那里，你可以找到更高级的代码（例如 多主机训练（multi-host training）），以及一些最初的训练脚本，例如：

configs/vit_i21k.py
：用于 预训练 ViT（Vision Transformer）模型；

configs/transfer.py
：用于 迁移已有模型（transfer learning）。

目录:

- [视觉Transformer和MLP-Mixer架构](#vision-transformer-and-mlp-mixer-architectures)
	- [Colab在线运行](#Colab)
	- [安装步骤](#installation)
	- [微调模型](#fine-tuning-a-model)
	- [视觉Transformer（ViT）模型](#vision-transformer)
		- [可用的ViT模型](#available-vit-models)
		- [ViT的预期结果](#expected-vit-results)
	- [MLP-Mixer模型](#mlp-mixer)
		- [可用的Mixer模型](#available-mixer-models)
		- [Mixer的预期结果](#expected-mixer-results)
	- [LiT模型](#lit-models)
	- [云端运行](#running-on-cloud)
		- [创建虚拟机](#create-a-vm)
		- [配置虚拟机](#setup-vm)
	- [参考文献BibTeX条目](#bibtex)
	- [免责声明](#disclaimers)
	- [更新日志](#changelog)


## Colab

以下两个 Colab 示例都可以在 GPU 或 TPU（8 核数据并行） 环境下运行。

🔹第一个 Colab

演示了 Vision Transformer (ViT) 和 MLP-Mixer 的 JAX 实现代码。

在这个 Colab 中，你可以：

直接在 Colab 界面中编辑仓库中的文件；

通过带注释的代码单元格（annotated cells）逐步学习代码逻辑；

交互式地操作与可视化数据。

https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax.ipynb


🔹第二个 Colab

该示例用于探索超过 5 万个 Vision Transformer 与混合模型（hybrid）检查点（checkpoints），这些模型是论文
《How to train your ViT? ...》
中生成实验数据所用的模型。

该 Colab 包含以下功能：

提供 检查点浏览与选择 的代码；

支持使用本仓库中的 JAX 代码 或 PyTorch 的 [`timm`] 库进行推理（timm 可直接加载这些模型）；

部分模型也已直接发布在 TensorFlow Hub 上（由 [Sayak Paul] 提供的外部贡献），例如
[sayakpaul/collections/vision_transformer]


此外，该 Colab 还支持：

对这些预训练检查点进行微调（fine-tuning）；

支持任意 tfds 数据集 或 你自己的 JPEG 图像数据集（可直接从 Google Drive 读取）。

https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax_augreg.ipynb



⚠️ 注意事项（截至 2021 年 6 月 20 日）

Google Colab 当前仅支持单个 GPU（NVIDIA Tesla T4）；

TPU（TPUv2-8） 与 Colab 虚拟机是通过网络间接连接的，通信延迟较高，导致训练速度较慢；

若你的微调任务涉及大量数据，建议搭建独立服务器或云端实例；

具体部署方式详见章节[Running on cloud](#running-on-cloud)

[`timm`]: https://github.com/rwightman/pytorch-image-models
[sayakpaul/collections/vision_transformer]: https://tfhub.dev/sayakpaul/collections/vision_transformer
[Sayak Paul]: https://github.com/sayakpaul



## Installation

`Python>=3.10` 

Install JAX and python dependencies by running:

```
# If using GPU:
pip install -r vit_jax/requirements.txt

# If using TPU:
pip install -r vit_jax/requirements-tpu.txt
```

对于新版的 [JAX](https://github.com/google/jax), 请按照该仓库中提供的安装说明进行操作。

需要注意的是，CPU、GPU 和 TPU 的安装步骤略有不同。

安装 [Flaxformer](https://github.com/google/flaxformer), 同样请遵循其对应仓库中的安装说明。

如需了解更多详情，请参考下文的云端运行部分 [Running on cloud](#running-on-cloud)



## Fine-tuning a model

你可以在自己感兴趣的数据集上对下载的模型进行微调（fine-tuning）。所有模型都使用相同的命令行接口。

例如，要**在 CIFAR-10 数据集上微调一个在 ImageNet-21k 上预训练过的 ViT-B/16 模型**
（请注意，我们在配置参数中使用了 b16,cifar10，并通过 --config.pretrained_dir 让代码直接从 GCS 云端存储桶 读取模型，而不是先下载到本地目录）：

**在 CIFAR-10 数据集上微调一个在 ImageNet-21k 上预训练过的 ViT-B/16 模型:使用如下命令👇**
```bash
python -m vit_jax.main --workdir=/tmp/vit-$(date +%s) \
    --config=$(pwd)/vit_jax/configs/vit.py:b16,cifar10 \
    --config.pretrained_dir='gs://vit_models/imagenet21k'
```

python -m vit_jax.main：运行vit_jax文件夹下的main函数的python脚本
--workdir：生成一个工作目录带时间戳文件夹(如/tmp/vit-1730793635/，其中1730793635就表示自1970年1月1日00:00:00 UTC（Unix epoch）起到当前时刻所经过的秒数。从而确保确文件夹的唯一性与可追溯性)，用于保存训练结果（如日志logs与权重checkpoints）。
--config：指定模型与数据集的配置文件路径，$(pwd) 表示当前工作目录路径，有当前工作目录路径/vit_jax/configs/vit.py文件,
          b16：代表 ViT-B/16 模型结构,“B” 表示 Base 模型,“16” 表示图像被划分为 16×16 的 Patch 大小；cifar10：表示使用 CIFAR-10 数据集 进行训练或微调。
--config.pretrained_dir：定义预训练模型权重的路径，这里直接从 Google Cloud Storage 读取，而无需本地下载。

**要在 CIFAR-10 数据集 上微调一个在 ImageNet-21k 上预训练过的 Mixer-B/16 模型:使用如下命令👇**

```bash
python -m vit_jax.main --workdir=/tmp/vit-$(date +%s) \
    --config=$(pwd)/vit_jax/configs/mixer_base16_cifar10.py \
    --config.pretrained_dir='gs://mixer_models/imagenet21k'
```

论文《How to train your ViT? ...》中新增了超过 5 万个模型权重（checkpoints）的预训练模型，
你可以使用 [`configs/augreg.py`] 配置文件对这些模型进行微调（fine-tuning）。
当你仅指定模型名称 ( 即 [`configs/model.py`] 中的 `config.name`参数值)时, 
系统会自动选择在上游验证集上精度最高的 ImageNet-21k 最优权重， 也就是论文第 4.5 节中提到的“推荐（recommended）”模型。
如果你想了解哪种模型更适合使用，可以参考论文中的 图 3（Figure 3）。
当然，你也可以手动指定其他预训练权重文件， (参考 Colab 示例 [`vit_jax_augreg.ipynb`]) 
具体方法是：到[`gs://vit_models/augreg`] 目录查找想要的模型文件名（去掉 .npz 后缀），然后在命令中通过 --config.pretrained_dir 参数告诉程序加载它。

**运行 ViT-JAX 主训练脚本，模型结构是 R_Ti_16，数据集是Oxford-IIIT Pet 👇**
```bash
python -m vit_jax.main --workdir=/tmp/vit-$(date +%s) \
    --config=$(pwd)/vit_jax/configs/augreg.py:R_Ti_16 \
    --config.dataset=oxford_iiit_pet \
    --config.base_lr=0.01
```
如果还要加指令可以加 

自己指定的预训练权重 --config.pretrained_dir='gs://vit_models/augreg/B_16_i21k_ft1k'（来自 gs://vit_models/augreg/ 目录）

批量大小：--config.batch_size=256

训练步数：--config.total_steps=20000

权重衰减（L2 正则化）：--config.weight_decay=0.0001

输出间隔：--config.log_every_steps=100

如果要训练直接数据集，可以加上项目两行

--config.dataset=my_dataset #用 ImageNet 格式加载我的自定义数据集

--config.dataset_dir=/home/ws/datasets/my_dataset #自己数据集的目录


目前，代码会自动下载 CIFAR-10 和 CIFAR-100 数据集。
他公共数据集或自定义数据集也可以很容易地集成，只需使用 [tensorflow
datasets library](https://github.com/tensorflow/datasets/). 
请注意，如果你添加了新的数据集，还需要修改 `vit_jax/input_pipeline.py` 文件，以指定该数据集的一些相关参数（如图像大小、通道数、类别数等）。

代码在微调（fine-tuning）时会自动使用所有可用的 GPU 或 TPU。

要查看所有可用的命令行参数（flags），可以运行： `python3 -m vit_jax.train
--help`.

内存使用说明：

- 不同模型对内存的需求不同。实际可用内存还取决于加速器（GPU/TPU）的类型和数量。如果遇到 显存不足（out-of-memory, OOM） 错误，可以：
  增大（梯度累积步数）`--config.accum_steps=8`，以降低单步显存占用或减小`--config.batch=512` （批量大小）(同时相应地降低 `--config.base_lr` 学习率).
- 主机（host）在内存中会维护一个数据打乱缓冲区（shuffle buffer）。
  如果出现 主机内存不足（host OOM），而不是显卡显存不足，可以适当减小默认的`--config.shuffle_buffer=50000`的值


## Vision Transformer

作者：Alexey Dosovitskiy*†、Lucas Beyer*、Alexander Kolesnikov*、Dirk Weissenborn*、Xiaohua Zhai*、Thomas Unterthiner、Mostafa Dehghani、Matthias Minderer、Georg Heigold、Sylvain Gelly、Jakob Uszkoreit 和 Neil Houlsby*†。

（*）表示技术贡献相同；（†）表示共同指导。

![Figure 1 from paper](vit_figure.png)

模型概述：
我们将一张图像划分为**固定大小的图像块（patches）**，
对每个图像块进行**线性嵌入（linear embedding）**，
再**加入位置嵌入（position embeddings）**，
然后将得到的向量序列输入到一个**标准的 Transformer 编码器**中。

为了实现图像分类，我们采用标准做法 —— 在输入序列前**添加一个可学习的“分类标记（classification token）”**，
Transformer 最终通过这个标记来输出整张图像的分类结果。


### Available ViT models

我们在不同的 GCS（Google Cloud Storage）存储桶 中提供了多种 ViT 模型。
这些模型可以通过如下命令下载，例如：

```
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

模型文件名 (去掉 `.npz` 后缀) 对应于[`vit_jax/configs/models.py`]文件中的 `config.model_name`参数。

模型存储路径与说明：
- [`gs://vit_models/imagenet21k`] - 在 ImageNet-21k 数据集上预训练的模型。.
- [`gs://vit_models/imagenet21k+imagenet2012`] - 在 ImageNet-21k 上预训练，并在 ImageNet-2012（即标准 ImageNet）上微调的模型。
- [`gs://vit_models/augreg`] - 在 ImageNet-21k 上预训练，并使用 [AugReg]（数据增强与正则化） 技术的模型，性能相比基础版本有明显提升。
- [`gs://vit_models/sam`] - 在 ImageNet 上使用 [SAM]（Sharpness-Aware Minimization，锐度感知最小化） 优化方法训练的模型。
- [`gs://vit_models/gsam`] - 在 ImageNet 上使用 [GSAM]（Generalized SAM） 方法训练的模型。

我们推荐使用以下采用 [AugReg]（数据增强与正则化） 方法训练的模型权重，这些模型在预训练阶段取得了最优的性能指标。
**以第一行为例**

**L/16 Vision Transformer（ViT）模型，预训练(在 ImageNet-21k 数据集上训练了 300 个 epoch 的 L/16 模型，并应用了 强数据增强（aug_strong1）、L2 权重衰减（wd=0.1） 等技术,模型大小：1243 MiB（约 1.24 GB）)和微调(在 ImageNet-21k 上预训练 的模型，后续在 ImageNet2012 数据集 上微调了 20,000 步，分辨率为 384x384，并使用了 更小的学习率（lr=0.01） 进行微调),模型在 ImageNet 上的分类性能（准确率为 85.59%），该模型处理图像的速度为 每秒 50 张图像。**

|  Model   |                                   Pre-trained checkpoint                                   |   Size   |                                                       Fine-tuned checkpoint                                                        | Resolution | Img/sec | Imagenet accuracy |
| :------- | :----------------------------------------------------------------------------------------- | -------: | :--------------------------------------------------------------------------------------------------------------------------------- | ---------: | ------: | ----------------: |
| L/16     | `gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz`     | 1243 MiB | `gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz`     |        384 |      50 |            85.59% |
| B/16     | `gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz`     |  391 MiB | `gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz`     |        384 |     138 |            85.49% |
| S/16     | `gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz`     |  115 MiB | `gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz`     |        384 |     300 |            83.73% |
| R50+L/32 | `gs://vit_models/augreg/R50_L_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz` | 1337 MiB | `gs://vit_models/augreg/R50_L_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz` |        384 |     327 |            85.99% |
| R26+S/32 | `gs://vit_models/augreg/R26_S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz`  |  170 MiB | `gs://vit_models/augreg/R26_S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz`  |        384 |     560 |            83.85% |
| Ti/16    | `gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz`      |   37 MiB | `gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz`      |        384 |     610 |            78.22% |
| B/32     | `gs://vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz`      |  398 MiB | `gs://vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz`      |        384 |     955 |            83.59% |
| S/32     | `gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz`        |  118 MiB | `gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz`        |        384 |    2154 |            79.58% |
| R+Ti/16  | `gs://vit_models/augreg/R_Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz`    |   40 MiB | `gs://vit_models/augreg/R_Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz`    |        384 |    2426 |            75.40% |

使用 [`gs://vit_models/imagenet21k`] 存储桶中的模型，已经复现了原始ViT论文 (https://arxiv.org/abs/2010.11929) 结果如下:

**这张表格显示了 R50+ViT-B/16 模型在不同数据集（CIFAR-10、CIFAR-100 和 ImageNet2012）上的训练效果，具体包括 dropout 的不同设置（0.0 和 0.1）对模型准确率和训练时间的影响。**

| model        | dataset      | dropout=0.0                                                                                                                                                         | dropout=0.1                                                                                                                                                          |
|:-------------|:-------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| R50+ViT-B_16 | cifar10      | 98.72%, 3.9h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5ER50.ViT-B_16/cifar10/do_0.0&_smoothingWeight=0)      | 98.94%, 10.1h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5ER50.ViT-B_16/cifar10/do_0.1&_smoothingWeight=0)      |
| R50+ViT-B_16 | cifar100     | 90.88%, 4.1h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5ER50.ViT-B_16/cifar100/do_0.0&_smoothingWeight=0)     | 92.30%, 10.1h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5ER50.ViT-B_16/cifar100/do_0.1&_smoothingWeight=0)     |
| R50+ViT-B_16 | imagenet2012 | 83.72%, 9.9h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5ER50.ViT-B_16/imagenet2012/do_0.0&_smoothingWeight=0) | 85.08%, 24.2h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5ER50.ViT-B_16/imagenet2012/do_0.1&_smoothingWeight=0) |
| ViT-B_16     | cifar10      | 99.02%, 2.2h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_16/cifar10/do_0.0&_smoothingWeight=0)          | 98.76%, 7.8h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_16/cifar10/do_0.1&_smoothingWeight=0)           |
| ViT-B_16     | cifar100     | 92.06%, 2.2h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_16/cifar100/do_0.0&_smoothingWeight=0)         | 91.92%, 7.8h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_16/cifar100/do_0.1&_smoothingWeight=0)          |
| ViT-B_16     | imagenet2012 | 84.53%, 6.5h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_16/imagenet2012/do_0.0&_smoothingWeight=0)     | 84.12%, 19.3h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_16/imagenet2012/do_0.1&_smoothingWeight=0)     |
| ViT-B_32     | cifar10      | 98.88%, 0.8h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_32/cifar10/do_0.0&_smoothingWeight=0)          | 98.75%, 1.8h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_32/cifar10/do_0.1&_smoothingWeight=0)           |
| ViT-B_32     | cifar100     | 92.31%, 0.8h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_32/cifar100/do_0.0&_smoothingWeight=0)         | 92.05%, 1.8h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_32/cifar100/do_0.1&_smoothingWeight=0)          |
| ViT-B_32     | imagenet2012 | 81.66%, 3.3h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_32/imagenet2012/do_0.0&_smoothingWeight=0)     | 81.31%, 4.9h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_32/imagenet2012/do_0.1&_smoothingWeight=0)      |
| ViT-L_16     | cifar10      | 99.13%, 6.9h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_16/cifar10/do_0.0&_smoothingWeight=0)          | 99.14%, 24.7h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_16/cifar10/do_0.1&_smoothingWeight=0)          |
| ViT-L_16     | cifar100     | 92.91%, 7.1h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_16/cifar100/do_0.0&_smoothingWeight=0)         | 93.22%, 24.4h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_16/cifar100/do_0.1&_smoothingWeight=0)         |
| ViT-L_16     | imagenet2012 | 84.47%, 16.8h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_16/imagenet2012/do_0.0&_smoothingWeight=0)    | 85.05%, 59.7h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_16/imagenet2012/do_0.1&_smoothingWeight=0)     |
| ViT-L_32     | cifar10      | 99.06%, 1.9h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_32/cifar10/do_0.0&_smoothingWeight=0)          | 99.09%, 6.1h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_32/cifar10/do_0.1&_smoothingWeight=0)           |
| ViT-L_32     | cifar100     | 93.29%, 1.9h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_32/cifar100/do_0.0&_smoothingWeight=0)         | 93.34%, 6.2h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_32/cifar100/do_0.1&_smoothingWeight=0)          |
| ViT-L_32     | imagenet2012 | 81.89%, 7.5h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_32/imagenet2012/do_0.0&_smoothingWeight=0)     | 81.13%, 15.0h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_32/imagenet2012/do_0.1&_smoothingWeight=0)     |

我们还希望强调，通过较短的训练周期也可以获得高质量的结果，并鼓励使用我们代码的用户调整超参数，以在**准确度**和**计算预算**之间找到平衡。下面的表格展示了在 CIFAR-10 和 CIFAR-100 数据集上的一些示例。

| upstream(预训练数据集的来源)    | model(模型名称)    | dataset(数据集)      | total_steps / warmup_steps(训练的总步数/热身步数)  | accuracy(分类准确率) | wall-clock time(训练的总时间) |                                                                         link(TensorBoard可视化结果) |
| ----------- | -------- | ------------ | --------------------------- | -------- | --------------- | ---------------------------------------------------------------------------- |
| imagenet21k | ViT-B_16 | cifar10      | 500 / 50                    |   98.59% |             17m | [tensorboard.dev](https://tensorboard.dev/experiment/QgkpiW53RPmjkabe1ME31g/) |
| imagenet21k | ViT-B_16 | cifar10      | 1000 / 100                  |   98.86% |             39m | [tensorboard.dev](https://tensorboard.dev/experiment/w8DQkDeJTOqJW5js80gOQg/) |
| imagenet21k | ViT-B_16 | cifar100     | 500 / 50                    |   89.17% |             17m | [tensorboard.dev](https://tensorboard.dev/experiment/5hM4GrnAR0KEZg725Ewnqg/) |
| imagenet21k | ViT-B_16 | cifar100     | 1000 / 100                  |   91.15% |             39m | [tensorboard.dev](https://tensorboard.dev/experiment/QLQTaaIoT9uEcAjtA0eRwg/) |


## MLP-Mixer

作者：Ilya Tolstikhin*、Neil Houlsby*、Alexander Kolesnikov*、Lucas Beyer*、
Xiaohua Zhai、Thomas Unterthiner、Jessica Yung、Andreas Steiner、Daniel Keysers、
Jakob Uszkoreit、Mario Lucic、Alexey Dosovitskiy。

（*）表示技术贡献相同。

![Figure 1 from paper](mixer_figure.png)

MLP-Mixer（简称 Mixer）由每个图像块的线性嵌入（per-patch linear embeddings）、Mixer 层和分类头（classifier head）组成。
Mixer 层包含一个 token-mixing MLP 和一个 channel-mixing MLP，每个 MLP 由两层全连接层和一个 GELU 非线性激活函数组成。
其他组成部分包括：跳跃连接（skip-connections）、dropout 和 线性分类头（linear classifier head）。

安装步骤请参考上面的 [the same steps](#installation)

### Available Mixer models

我们提供了在 ImageNet 和 ImageNet-21k 数据集上预训练的 Mixer-B/16 和 Mixer-L/16 模型。
详细信息可以在 Mixer 论文的第 3 表 中找到。
所有模型可以在以下链接下载：

https://console.cloud.google.com/storage/mixer_models/

请注意，这些模型也可以直接从 TF-Hub 获取:
[sayakpaul/collections/mlp-mixer] (由 [Sayak
Paul]提供的外部贡献).

[sayakpaul/collections/mlp-mixer]: https://tfhub.dev/sayakpaul/collections/mlp-mixer

### Expected Mixer results

我们在 Google Cloud 的四个 V100 GPU 机器上运行了微调代码，使用了该仓库中的默认适配参数。以下是结果：

upstream     | model      | dataset | accuracy | wall_clock_time | link
:----------- | :--------- | :------ | -------: | :-------------- | :---
ImageNet     | Mixer-B/16 | cifar10 | 96.72%   | 3.0h            | [tensorboard.dev](https://tensorboard.dev/experiment/j9zCYt9yQVm93nqnsDZayA/)
ImageNet     | Mixer-L/16 | cifar10 | 96.59%   | 3.0h            | [tensorboard.dev](https://tensorboard.dev/experiment/Q4feeErzRGGop5XzAvYj2g/)
ImageNet-21k | Mixer-B/16 | cifar10 | 96.82%   | 9.6h            | [tensorboard.dev](https://tensorboard.dev/experiment/mvP4McV2SEGFeIww20ie5Q/)
ImageNet-21k | Mixer-L/16 | cifar10 | 98.34%   | 10.0h           | [tensorboard.dev](https://tensorboard.dev/experiment/dolAJyQYTYmudytjalF6Jg/)


## LiT models

有关详细信息，请参考 Google AI 博客文章
[LiT: adding language understanding to image models](http://ai.googleblog.com/2022/04/locked-image-tuning-adding-language.html),
或阅读 CVPR 论文 "LiT: Zero-Shot Transfer with Locked-image text Tuning"
(https://arxiv.org/abs/2111.07991).

我们发布了一个 Transformer B/16-base 模型，具有 72.1% 的 ImageNet 零-shot 准确率，
以及一个 L/16-large 模型，具有 75.7% 的 ImageNet 零-shot 准确率。
有关这些模型的更多详情，请参阅[LiT model card](model_cards/lit.md).

我们提供了一个浏览器内的演示，使用了小型文本编码器，供交互式使用（最小的模型甚至可以在现代手机上运行）:

https://google-research.github.io/vision_transformer/lit/

最后，我们提供了一个 Colab 示例，展示如何使用 JAX 模型，结合图像和文本编码器：

https://colab.research.google.com/github/google-research/vision_transformer/blob/main/lit.ipynb

请注意，以上模型尚不支持多语言输入，但我们正在努力发布此类模型，并将在它们可用时更新本仓库。

本仓库仅包含 LiT 模型的评估代码。训练代码可以在 `big_vision` 仓库中找到：

https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/image_text

预计的零-shot 结果可以在 [`model_cards/lit.md`] 中找到（请注意，零-shot 评估与 Colab 中简化的评估略有不同）：

**零-shot 准确率：该模型在不同任务（如分类、检索）上不需要额外的训练就能达到的准确率。**

| Model | B16B_2 | L16L |
| :--- | ---: | ---: |
| ImageNet zero-shot | 73.9% | 75.7% |
| ImageNet v2 zero-shot | 65.1% | 66.6% |
| CIFAR100 zero-shot | 79.0% | 80.5% |
| Pets37 zero-shot | 83.3% | 83.3% |
| Resisc45 zero-shot | 25.3% | 25.6% |
| MS-COCO Captions image-to-text retrieval | 51.6% | 48.5% |
| MS-COCO Captions text-to-image retrieval | 31.8% | 31.1% |

## Running on cloud

虽然上面的 [colabs](#colab) 非常适合入门，但通常你可能希望在更强大的机器上进行训练，使用更多的加速器（如 GPU 或 TPU）。

### Create a VM

你可以使用以下命令在 Google Cloud 上设置一个带 GPU 的虚拟机（VM）：

**提供了如何在 Google Cloud 上设置并管理一个 GPU 虚拟机的详细步骤，命令如下:**

```bash
# 设置所有命令使用的变量。
# 请注意，项目必须已启用账单。
# 有关带 GPU 的区域列表，请参考
# https://cloud.google.com/compute/docs/gpus/gpu-regions-zones
PROJECT=my-awesome-gcp-project  # Project must have billing enabled. # 项目必须启用账单。
VM_NAME=vit-jax-vm-gpu
ZONE=europe-west4-b

# 以下设置已通过该仓库进行测试。你可以选择其他
# 镜像和机器类型的组合（例如），参考以下 gcloud 命令：
# gcloud compute images list --project ml-images
# gcloud compute machine-types list
# 等等。
gcloud compute instances create $VM_NAME \
    --project=$PROJECT --zone=$ZONE \
    --image=c1-deeplearning-tf-2-5-cu110-v20210527-debian-10 \
    --image-project=ml-images --machine-type=n1-standard-96 \
    --scopes=cloud-platform,storage-full --boot-disk-size=256GB \
    --boot-disk-type=pd-ssd --metadata=install-nvidia-driver=True \
    --maintenance-policy=TERMINATE \
    --accelerator=type=nvidia-tesla-v100,count=8

# 在设置并启动虚拟机几分钟后，连接到虚拟机。
gcloud compute ssh --project $PROJECT --zone $ZONE $VM_NAME

# 使用后停止虚拟机（停止的虚拟机只会产生存储费用）。
gcloud compute instances stop --project $PROJECT --zone $ZONE $VM_NAME

# 使用后删除虚拟机（这将删除虚拟机上存储的所有数据）。
gcloud compute instances delete --project $PROJECT --zone $ZONE $VM_NAME
```

可以使用以下类似的命令来创建一个带 TPU 的云虚拟机（VM）。下面的命令来自TPU教程 [TPU tutorial]):

[TPU tutorial]: https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm

```bash
PROJECT=my-awesome-gcp-project  # Project must have billing enabled.
VM_NAME=vit-jax-vm-tpu
ZONE=europe-west4-a

# 初始设置时需要创建服务身份。
gcloud beta services identity create --service tpu.googleapis.com

# 创建一个直接连接 TPU 的虚拟机。
gcloud alpha compute tpus tpu-vm create $VM_NAME \
    --project=$PROJECT --zone=$ZONE \
    --accelerator-type v3-8 \
    --version tpu-vm-base

# 连接到虚拟机（设置和启动机器需要一些时间）。
gcloud alpha compute tpus tpu-vm ssh --project $PROJECT --zone $ZONE $VM_NAME

# 使用后停止虚拟机（停止后的虚拟机只会产生存储费用）。
gcloud alpha compute tpus tpu-vm stop --project $PROJECT --zone $ZONE $VM_NAME

# 使用后删除虚拟机（这将删除虚拟机上存储的所有数据）。
gcloud alpha compute tpus tpu-vm delete --project $PROJECT --zone $ZONE $VM_NAME
```

### Setup VM

然后，你可以像往常一样获取仓库并安装依赖 (包括带有 TPU 支持的 `jaxlib`) :

```bash
git clone --depth=1 --branch=master https://github.com/google-research/vision_transformer
cd vision_transformer

# optional: install virtualenv
pip3 install virtualenv
python3 -m virtualenv env
. env/bin/activate
```

如果你连接到带有 GPU 的虚拟机，使用以下命令安装 JAX 和其他依赖：

```bash
pip install -r vit_jax/requirements.txt
```

如果你连接到带有 TPU 的虚拟机，使用以下命令安装 JAX 和其他依赖：

```bash
pip install -r vit_jax/requirements-tpu.txt
```

安装 [Flaxformer](https://github.com/google/flaxformer), 并按照相应仓库中的安装说明进行操作。

对于 GPU 和 TPU，可以通过以下命令检查 JAX 是否能连接到已附加的加速器：
```bash
python -c 'import jax; print(jax.devices())'
```

最后，执行[fine-tuning a model](#fine-tuning-a-model)部分提到的命令


## Bibtex

**引用论文**

```
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={ICLR},
  year={2021}
}

@article{tolstikhin2021mixer,
  title={MLP-Mixer: An all-MLP Architecture for Vision},
  author={Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Jessica and Steiner, Andreas and Keysers, Daniel and Uszkoreit, Jakob and Lucic, Mario and Dosovitskiy, Alexey},
  journal={arXiv preprint arXiv:2105.01601},
  year={2021}
}

@article{steiner2021augreg,
  title={How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers},
  author={Steiner, Andreas and Kolesnikov, Alexander and and Zhai, Xiaohua and Wightman, Ross and Uszkoreit, Jakob and Beyer, Lucas},
  journal={arXiv preprint arXiv:2106.10270},
  year={2021}
}

@article{chen2021outperform,
  title={When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations},
  author={Chen, Xiangning and Hsieh, Cho-Jui and Gong, Boqing},
  journal={arXiv preprint arXiv:2106.01548},
  year={2021},
}

@article{zhuang2022gsam,
  title={Surrogate Gap Minimization Improves Sharpness-Aware Training},
  author={Zhuang, Juntang and Gong, Boqing and Yuan, Liangzhe and Cui, Yin and Adam, Hartwig and Dvornek, Nicha and Tatikonda, Sekhar and Duncan, James and Liu, Ting},
  journal={ICLR},
  year={2022},
}

@article{zhai2022lit,
  title={LiT: Zero-Shot Transfer with Locked-image Text Tuning},
  author={Zhai, Xiaohua and Wang, Xiao and Mustafa, Basil and Steiner, Andreas and Keysers, Daniel and Kolesnikov, Alexander and Beyer, Lucas},
  journal={CVPR},
  year={2022}
}
```


## Changelog

In reverse chronological order:

- 2022-08-18: Added LiT-B16B_2 model that was trained for 60k steps
  (LiT_B16B: 30k) without linear head on the image side (LiT_B16B: 768) and has
  better performance.

- 2022-06-09: Added the ViT and Mixer models trained from scratch using
  [GSAM] on ImageNet without strong data augmentations. The resultant ViTs
  outperform those of similar sizes trained using AdamW optimizer or the
  original [SAM] algorithm, or with strong data augmentations.

- 2022-04-14: Added models and Colab for [LiT models](#lit-models).

- 2021-07-29: Added ViT-B/8 AugReg models (3 upstream checkpoints and adaptations
  with resolution=224).

- 2021-07-02: Added the "When Vision Transformers Outperform
  ResNets..." paper

- 2021-07-02: Added [SAM](https://arxiv.org/abs/2010.01412)
  (Sharpness-Aware Minimization) optimized ViT and MLP-Mixer checkpoints.

- 2021-06-20: Added the "How to train your ViT? ..." paper, and a new
  Colab to explore the >50k pre-trained and fine-tuned checkpoints mentioned in
  the paper.

- 2021-06-18: This repository was rewritten to use Flax Linen API and
  `ml_collections.ConfigDict` for configuration.

- 2021-05-19: With publication of the "How to train your ViT? ..."
  paper, we added more than 50k ViT and hybrid models pre-trained on ImageNet and
  ImageNet-21k with various degrees of data augmentation and model regularization,
  and fine-tuned on ImageNet, Pets37, Kitti-distance, CIFAR-100, and Resisc45.
  Check out [`vit_jax_augreg.ipynb`] to navigate this treasure trove of models!
  For example, you can use that Colab to fetch the filenames of recommended
  pre-trained and fine-tuned checkpoints from the `i21k_300` column of Table 3 in
  the paper.

- 2020-12-01: Added the R50+ViT-B/16 hybrid model (ViT-B/16 on
  top of a Resnet-50 backbone). When pretrained on imagenet21k, this model
  achieves almost the performance of the L/16 model with less than half the
  computational finetuning cost. Note that "R50" is somewhat modified for the
  B/16 variant: The original ResNet-50 has [3,4,6,3] blocks, each reducing the
  resolution of the image by a factor of two. In combination with the ResNet
  stem this would result in a reduction of 32x so even with a patch size of
  (1,1) the ViT-B/16 variant cannot be realized anymore. For this reason we
  instead use [3,4,9] blocks for the R50+B/16 variant.

- 2020-11-09: Added the ViT-L/16 model.

- 2020-10-29: Added ViT-B/16 and ViT-L/16 models pretrained
  on ImageNet-21k and then fine-tuned on ImageNet at 224x224 resolution (instead
  of default 384x384). These models have the suffix "-224" in their name.
  They are expected to achieve 81.2% and 82.7% top-1 accuracies respectively.


## Disclaimers

Open source release prepared by Andreas Steiner.

注意：本仓库是从
[google-research/big_transfer](https://github.com/google-research/big_transfer)分叉并修改而来的。

**这不是一个官方的 Google 产品。**


[GSAM]: https://arxiv.org/abs/2203.08065
[SAM]: https://arxiv.org/abs/2010.01412
[AugReg]: https://arxiv.org/abs/2106.10270

[`vit_jax/configs/models.py`]: https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py
[`model_cards/lit.md`]: https://github.com/google-research/vision_transformer/blob/main/model_cards/lit.md

[`configs/augreg.py`]: https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/augreg.py
[`configs/model.py`]: https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py
[`vit_jax_augreg.ipynb`]: https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax_augreg.ipynb
[`vit_jax.ipynb`]: https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax.ipynb

[`gs://vit_models/imagenet21k`]: https://console.cloud.google.com/storage/browser/vit_models/imagenet21k/
[`gs://vit_models/imagenet21k+imagenet2012`]: https://console.cloud.google.com/storage/browser/vit_models/imagenet21k+imagenet2012/
[`gs://vit_models/augreg`]: https://console.cloud.google.com/storage/browser/vit_models/augreg/
[`gs://vit_models/sam`]: https://console.cloud.google.com/storage/browser/vit_models/sam/
[`gs://mixer_models/sam`]: https://console.cloud.google.com/storage/mixer_models/sam/
[`gs://vit_models/gsam`]: https://console.cloud.google.com/storage/browser/vit_models/gsam/
[`gs://mixer_models/gsam`]: https://console.cloud.google.com/storage/mixer_models/gsam/

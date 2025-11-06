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

by Alexey Dosovitskiy\*†, Lucas Beyer\*, Alexander Kolesnikov\*, Dirk
Weissenborn\*, Xiaohua Zhai\*, Thomas Unterthiner, Mostafa Dehghani, Matthias
Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit and Neil Houlsby\*†.

(\*) equal technical contribution, (†) equal advising.

![Figure 1 from paper](vit_figure.png)

Overview of the model: we split an image into fixed-size patches, linearly embed
each of them, add position embeddings, and feed the resulting sequence of
vectors to a standard Transformer encoder. In order to perform classification,
we use the standard approach of adding an extra learnable "classification token"
to the sequence.

### Available ViT models

We provide a variety of ViT models in different GCS buckets. The models can be
downloaded with e.g.:

```
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

The model filenames (without the `.npz` extension) correspond to the
`config.model_name` in [`vit_jax/configs/models.py`]

- [`gs://vit_models/imagenet21k`] - Models pre-trained on ImageNet-21k.
- [`gs://vit_models/imagenet21k+imagenet2012`] - Models pre-trained on
  ImageNet-21k and fine-tuned on ImageNet.
- [`gs://vit_models/augreg`] - Models pre-trained on ImageNet-21k,
  applying varying amounts of [AugReg]. Improved performance.
- [`gs://vit_models/sam`] - Models pre-trained on ImageNet with [SAM].
- [`gs://vit_models/gsam`] - Models pre-trained on ImageNet with [GSAM].

We recommend using the following checkpoints, trained with [AugReg] that have
the best pre-training metrics:

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

The results from the original ViT paper (https://arxiv.org/abs/2010.11929) have
been replicated using the models from [`gs://vit_models/imagenet21k`]:

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

We also would like to emphasize that high-quality results can be achieved with
shorter training schedules and encourage users of our code to play with
hyper-parameters to trade-off accuracy and computational budget.
Some examples for CIFAR-10/100 datasets are presented in the table below.

| upstream    | model    | dataset      | total_steps / warmup_steps  | accuracy | wall-clock time |                                                                         link |
| ----------- | -------- | ------------ | --------------------------- | -------- | --------------- | ---------------------------------------------------------------------------- |
| imagenet21k | ViT-B_16 | cifar10      | 500 / 50                    |   98.59% |             17m | [tensorboard.dev](https://tensorboard.dev/experiment/QgkpiW53RPmjkabe1ME31g/) |
| imagenet21k | ViT-B_16 | cifar10      | 1000 / 100                  |   98.86% |             39m | [tensorboard.dev](https://tensorboard.dev/experiment/w8DQkDeJTOqJW5js80gOQg/) |
| imagenet21k | ViT-B_16 | cifar100     | 500 / 50                    |   89.17% |             17m | [tensorboard.dev](https://tensorboard.dev/experiment/5hM4GrnAR0KEZg725Ewnqg/) |
| imagenet21k | ViT-B_16 | cifar100     | 1000 / 100                  |   91.15% |             39m | [tensorboard.dev](https://tensorboard.dev/experiment/QLQTaaIoT9uEcAjtA0eRwg/) |


## MLP-Mixer

by Ilya Tolstikhin\*, Neil Houlsby\*, Alexander Kolesnikov\*, Lucas Beyer\*,
Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers,
Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy.

(\*) equal contribution.

![Figure 1 from paper](mixer_figure.png)

MLP-Mixer (*Mixer* for short) consists of per-patch linear embeddings, Mixer
layers, and a classifier head. Mixer layers contain one token-mixing MLP and one
channel-mixing MLP, each consisting of two fully-connected layers and a GELU
nonlinearity. Other components include: skip-connections, dropout, and linear
classifier head.

For installation follow [the same steps](#installation) as above.

### Available Mixer models

We provide the Mixer-B/16 and Mixer-L/16 models pre-trained on the ImageNet and
ImageNet-21k datasets. Details can be found in Table 3 of the Mixer paper. All
the models can be found at:

https://console.cloud.google.com/storage/mixer_models/

Note that these models are also available directly from TF-Hub:
[sayakpaul/collections/mlp-mixer] (external contribution by [Sayak
Paul]).

[sayakpaul/collections/mlp-mixer]: https://tfhub.dev/sayakpaul/collections/mlp-mixer

### Expected Mixer results

We ran the fine-tuning code on Google Cloud machine with four V100 GPUs with the
default adaption parameters from this repository. Here are the results:

upstream     | model      | dataset | accuracy | wall_clock_time | link
:----------- | :--------- | :------ | -------: | :-------------- | :---
ImageNet     | Mixer-B/16 | cifar10 | 96.72%   | 3.0h            | [tensorboard.dev](https://tensorboard.dev/experiment/j9zCYt9yQVm93nqnsDZayA/)
ImageNet     | Mixer-L/16 | cifar10 | 96.59%   | 3.0h            | [tensorboard.dev](https://tensorboard.dev/experiment/Q4feeErzRGGop5XzAvYj2g/)
ImageNet-21k | Mixer-B/16 | cifar10 | 96.82%   | 9.6h            | [tensorboard.dev](https://tensorboard.dev/experiment/mvP4McV2SEGFeIww20ie5Q/)
ImageNet-21k | Mixer-L/16 | cifar10 | 98.34%   | 10.0h           | [tensorboard.dev](https://tensorboard.dev/experiment/dolAJyQYTYmudytjalF6Jg/)


## LiT models

For details, refer to the Google AI blog post
[LiT: adding language understanding to image models](http://ai.googleblog.com/2022/04/locked-image-tuning-adding-language.html),
or read the CVPR paper "LiT: Zero-Shot Transfer with Locked-image text Tuning"
(https://arxiv.org/abs/2111.07991).

We published a Transformer B/16-base model with an ImageNet zeroshot accuracy of
72.1%, and a L/16-large model with an ImageNet zeroshot accuracy of 75.7%. For
more details about these models, please refer to the
[LiT model card](model_cards/lit.md).

We provide a in-browser demo with small text encoders for interactive use (the
smallest models should even run on a modern cell phone):

https://google-research.github.io/vision_transformer/lit/

And finally a Colab to use the JAX models with both image and text encoders:

https://colab.research.google.com/github/google-research/vision_transformer/blob/main/lit.ipynb

Note that none of above models support multi-lingual inputs yet, but we're
working on publishing such models and will update this repository once they
become available.

This repository only contains evaluation code for LiT models. You can find the
training code in the `big_vision` repository:

https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/image_text

Expected zeroshot results from [`model_cards/lit.md`] (note that the zeroshot
evaluation is slightly different from the simplified evaluation in the Colab):

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

While above [colabs](#colab) are pretty useful to get started, you would usually
want to train on a larger machine with more powerful accelerators.

### Create a VM

You can use the following commands to setup a VM with GPUs on Google Cloud:

```bash
# Set variables used by all commands below.
# Note that project must have accounting set up.
# For a list of zones with GPUs refer to
# https://cloud.google.com/compute/docs/gpus/gpu-regions-zones
PROJECT=my-awesome-gcp-project  # Project must have billing enabled.
VM_NAME=vit-jax-vm-gpu
ZONE=europe-west4-b

# Below settings have been tested with this repository. You can choose other
# combinations of images & machines (e.g.), refer to the corresponding gcloud commands:
# gcloud compute images list --project ml-images
# gcloud compute machine-types list
# etc.
gcloud compute instances create $VM_NAME \
    --project=$PROJECT --zone=$ZONE \
    --image=c1-deeplearning-tf-2-5-cu110-v20210527-debian-10 \
    --image-project=ml-images --machine-type=n1-standard-96 \
    --scopes=cloud-platform,storage-full --boot-disk-size=256GB \
    --boot-disk-type=pd-ssd --metadata=install-nvidia-driver=True \
    --maintenance-policy=TERMINATE \
    --accelerator=type=nvidia-tesla-v100,count=8

# Connect to VM (after some minutes needed to setup & start the machine).
gcloud compute ssh --project $PROJECT --zone $ZONE $VM_NAME

# Stop the VM after use (only storage is billed for a stopped VM).
gcloud compute instances stop --project $PROJECT --zone $ZONE $VM_NAME

# Delete VM after use (this will also remove all data stored on VM).
gcloud compute instances delete --project $PROJECT --zone $ZONE $VM_NAME
```

Alternatively, you can use the following similar commands to set up a Cloud VM
with TPUs attached to them (below commands copied from the [TPU tutorial]):

[TPU tutorial]: https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm

```bash
PROJECT=my-awesome-gcp-project  # Project must have billing enabled.
VM_NAME=vit-jax-vm-tpu
ZONE=europe-west4-a

# Required to set up service identity initially.
gcloud beta services identity create --service tpu.googleapis.com

# Create a VM with TPUs directly attached to it.
gcloud alpha compute tpus tpu-vm create $VM_NAME \
    --project=$PROJECT --zone=$ZONE \
    --accelerator-type v3-8 \
    --version tpu-vm-base

# Connect to VM (after some minutes needed to setup & start the machine).
gcloud alpha compute tpus tpu-vm ssh --project $PROJECT --zone $ZONE $VM_NAME

# Stop the VM after use (only storage is billed for a stopped VM).
gcloud alpha compute tpus tpu-vm stop --project $PROJECT --zone $ZONE $VM_NAME

# Delete VM after use (this will also remove all data stored on VM).
gcloud alpha compute tpus tpu-vm delete --project $PROJECT --zone $ZONE $VM_NAME
```

### Setup VM

And then fetch the repository and the install dependencies (including `jaxlib`
with TPU support) as usual:

```bash
git clone --depth=1 --branch=master https://github.com/google-research/vision_transformer
cd vision_transformer

# optional: install virtualenv
pip3 install virtualenv
python3 -m virtualenv env
. env/bin/activate
```

If you're connected to a VM with GPUs attached, install JAX and other dependencies with the following
command:

```bash
pip install -r vit_jax/requirements.txt
```

If you're connected to a VM with TPUs attached, install JAX and other dependencies with the following
command:

```bash
pip install -r vit_jax/requirements-tpu.txt
```

Install [Flaxformer](https://github.com/google/flaxformer), follow the instructions
provided in the corresponding repository linked here.

For both GPUs and TPUs, Check that JAX can connect to attached accelerators with the command:
```bash
python -c 'import jax; print(jax.devices())'
```

And finally execute one of the commands mentioned in the section
[fine-tuning a model](#fine-tuning-a-model).


## Bibtex

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

Note: This repository was forked and modified from
[google-research/big_transfer](https://github.com/google-research/big_transfer).

**This is not an official Google product.**


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

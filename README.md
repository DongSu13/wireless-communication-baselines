# wireless-communication-baselines

[中文](README.md) | [English](README.en.md)

面向 RGB 预测、GPS 轨迹、CSI 预测和波束分类的无线通信模型集合，以及可独立使用的层级模型模块。仓库包含核心算法、通用 Ridge 回归、模型推理入口和合成输入测试。每项注明原工作与本地实现范围，用于学习和复现练习。

当前为私有审阅版本，尚未公开。

## 内容与运行

| 文件 | 用途 |
|---|---|
| [paper_baselines.py](paper_baselines.py) | 14 种任务基线，统一返回 `BaselineOutput` |
| [model.py](model.py) | 多模态状态模型、条件潜空间扩散、CSI 解码与波束头 |
| [vae.py](vae.py) | 卷积 KL 自编码器及损失 |
| [ridge_baseline.py](ridge_baseline.py) | 按组隔离的特征回归，仅在训练集拟合标准化 |
| [wifo_inference.py](wifo_inference.py) | 信道预测模型推理入口，调用可信的外部官方源码 |
| [tests/](tests/) | 合成张量的输出、梯度、划分和加载安全检查 |

使用 Python 3.9+；依赖见 [requirements.txt](requirements.txt)。

```powershell
py -3.9 -m pip install -r requirements.txt
py -3.9 -m unittest discover -s tests
```

`build_baseline(name, shapes, num_beams)` 的 `shapes` 描述 image/CSI 张量形状、历史长度和预测步数。RGB、GPS、CSI 与波束输出独立；测试展示最小合成输入。Ridge 接收普通数值数组组成的 NPZ：`features[N,F]`、`targets[N,T]`、`groups[N]`，拒绝对象/pickle 数组。

```powershell
py -3.9 ridge_baseline.py --arrays ./data/features.npz --output ./outputs/ridge.json
```


WiFo 与其他模型一样收录在下方引用表；运行该模型前，按 [模型说明](docs/WIFO.md) 安装 `requirements-wifo.txt` 中的可选依赖，并单独取得可信官方源码、公开数据和权重。其他 baseline 使用根依赖即可。

## 模型引用与实现范围

方法根据相关任务和算法构建。本表逐项提供论文或 GitHub 引用及实际实现。部分历史 ID 与参考架构不同：例如 `convlstm`/`predrnn_style`/`strictsimvp_style` 使用 2D CNN，`convgru`/`temporal3d` 使用 MLP。任务适配不等于原论文完整复现。

| Model / method | Task | Actual implementation | Paper / GitHub | Relation |
|---|---|---|---|---|
| [`convlstm`](paper_baselines.py) | RGB prediction | 1 hidden 2D CNN block; history folded into channels; no recurrent cell / 单层 CNN | [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214) | architecture context only; local proxy, not ConvLSTM |
| [`predrnn_style`](paper_baselines.py) | RGB prediction | 2 hidden 2D CNN blocks; no recurrence/memory / 双层 CNN | [PredRNN official repository and paper citations](https://github.com/thuml/predrnn-pytorch) | architecture context only; local proxy, not PredRNN |
| [`strictsimvp_style`](paper_baselines.py) | RGB prediction | 3 hidden 2D CNN blocks; no SimVP encoder/translator/decoder / 三层 CNN | [SimVP: Simpler yet Better Video Prediction](https://arxiv.org/abs/2206.05099) | architecture context only; local proxy, not strict SimVP reproduction |
| [`gru`](paper_baselines.py) | GPS trajectory | torch.nn.GRU(2,64) plus future trajectory linear head / 标准 GRU + 线性头 | [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) | standard recurrent primitive adapted to local GPS task |
| [`lstm`](paper_baselines.py) | GPS trajectory | torch.nn.LSTM(2,64) plus future trajectory linear head / 标准 LSTM + 线性头 | [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) | standard recurrent primitive adapted to local GPS task |
| [`tcn`](paper_baselines.py) | GPS trajectory | 2 Conv1d blocks, symmetric padding and mean pooling / 非因果、无膨胀卷积 | [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) | architecture context only; local temporal CNN, not published causal dilated TCN |
| [`wayformer_style`](paper_baselines.py) | GPS trajectory | GPS-only projection/time embedding, 2-layer Transformer encoder, linear future head | [Wayformer: Motion Forecasting via Simple & Efficient Attention Networks](https://arxiv.org/abs/2207.05844)<br>[Attention Is All You Need](https://arxiv.org/abs/1706.03762) | local Transformer task adapter; no Wayformer map/agent multimodal architecture |
| [`convgru`](paper_baselines.py) | Fine CSI prediction | 1 hidden dense layer on flattened CSI history / 单层 MLP，无卷积/GRU | [Delving Deeper into Convolutional Networks for Learning Video Representations](https://arxiv.org/abs/1511.06432) | architecture context only; local MLP proxy, not ConvGRU |
| [`temporal3d`](paper_baselines.py) | Fine CSI prediction | 2 hidden dense layers on flattened CSI history / 双层 MLP，无 3D 卷积 | [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767) | C3D is background only; original specific upstream untraced |
| [`channeltransformer10_style`](paper_baselines.py) | Fine CSI prediction | CSI projection/time embedding, 2 generic Transformer encoder layers, future linear head | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | generic Transformer algorithm reference; exact ChannelTransformer10 upstream unresolved / 具体来源待确认 |
| [`gmu`](paper_baselines.py) | Beam classification | RGB/CSI/GPS linear encoders, 3-way softmax gate, weighted fusion and beam head | [Gated Multimodal Units for Information Fusion](https://arxiv.org/abs/1702.01992) | local gated fusion adaptation, not exact original GMU equation |
| [`crossattn`](paper_baselines.py) | Beam classification | GPS query cross-attends to RGB/CSI tokens, then beam head | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | attention primitive adapted to local beam task |
| [`decisionfusion`](paper_baselines.py) | Beam classification | Average of RGB/CSI/GPS logits; identical logits repeated for each future step | [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406) | local late fusion baseline; survey is background only |
| [`rgb_csi_mmbp_style`](paper_baselines.py) | Beam classification | RGB/CSI flattened linear encoders, feature concatenation, 2-layer beam MLP | [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406) | local beam adapter; exact MM-BP upstream unresolved / 具体来源待确认; survey is background only |
| [`csi_rgb_ridge`](ridge_baseline.py) | CSI/pose → RGB features | Generic scikit-learn Ridge with disjoint groups and train-only standardization | [scikit-learn Ridge implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py)<br>[Scikit-learn: Machine Learning in Python](https://jmlr.org/papers/v12/pedregosa11a.html) | Library attribution; generic feature regression, no dataset-specific experiment claimed |
| [WiFo](wifo_inference.py) | Channel prediction / 信道预测 | Official model through an external trusted checkout; two-sample inference adapter / 官方模型外部调用及两样本推理入口 | [WiFo: Wireless Foundation Model for Channel Prediction](https://doi.org/10.1007/s11432-025-4349-0)<br>[Official GitHub](https://github.com/PKU-PCNI/WiFo) | local inference adapter; no redistributed upstream model source or full-paper reproduction / 具体运行见 [模型说明](docs/WIFO.md) |
| [`CoarseWorldModel`](model.py) | Multimodal state/condition prediction | Custom causal RGB-latent/CSI-stat/GPS Transformer including CSIStatEncoder | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)<br>[Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability](https://github.com/OpenDriveLab/Vista) | Transformer algorithm reference; Vista is task background only, not traced official code |
| [`LatentDiffusion`](model.py) | Conditional latent prediction | Local U-Net/attention, DDPM noise objective and deterministic DDIM-like sampling | [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)<br>[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)<br>[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) | architecture/algorithm context; not official LDM architecture/weights reproduction |
| [`FineCSIHead`](model.py) | Future CSI decoding | Local patched CSI memory and learned future-query Transformer decoder | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer primitive reference; not attributed to WiFo or a specific channel paper |
| [`BeamHead`](model.py) | Beam classification | Local LayerNorm + linear classifier on CSI-stat state | [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406) | own-source GitHub is provenance; survey is task background only |
| [`AutoencoderKL`](vae.py) | RGB latent encoding/decoding | Local f4 convolutional Gaussian KL autoencoder with residual blocks and attention | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)<br>[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | VAE/LDM architecture context; not official pretrained Stable Diffusion autoencoder |

`ChannelTransformer10` 与 `MM-BP` 的特定上游尚未确认，相应引用仅为通用算法背景。支持模型为本地实现；合成测试只验证代码行为，不证明论文精度、训练效果或真实数据泛化。

## 验证与使用范围

此前 11 项模块与加载安全测试通过，覆盖任务方法的合成输出/梯度、层级组件、Ridge 划分和模型权重加载。WiFo 的实际推理检查只覆盖两个公开 D1 样本与三种掩码，没有完成训练或全量论文评测。语法与合成检查不代表论文精度或真实任务泛化；依赖范围并未穷尽验证。

本项目是非官方学习与复现练习。使用相关方法或成果时，请引用各项原论文和官方代码；完整引用汇总见 [来源说明](docs/REFERENCES.md)。

## 许可

本集合不另行声明 MIT 授权。学习用途和引用不等于源码再分发许可；外部项目、依赖、数据和权重遵守各自条款。许可未确认的上游模型源码不随集合分发。数据、权重、私密资料和历史实验档案均不打包。

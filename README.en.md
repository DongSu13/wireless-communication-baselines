# wireless-communication-baselines

[中文](README.md) | [English](README.en.md)

A wireless communication model collection for RGB, GPS trajectory, CSI and beam prediction, plus independently usable hierarchical components. Contents are core algorithms, generic Ridge regression, model inference interfaces and synthetic tests. Original works and local adaptations are documented for learning and reproduction exercises.

## Contents and usage

- [paper_baselines.py](paper_baselines.py): 14 task methods returning `BaselineOutput`.
- [model.py](model.py): multimodal state model, conditional latent diffusion, CSI and beam heads.
- [vae.py](vae.py): convolutional KL autoencoder and loss.
- [ridge_baseline.py](ridge_baseline.py): disjoint-group regression with train-only scaling.
- [wifo_inference.py](wifo_inference.py): channel-prediction inference using a trusted external official model checkout.
- [tests/](tests/): synthetic output/gradient/split and loading checks.

Use Python 3.9+ and [requirements.txt](requirements.txt).

```powershell
py -3.9 -m pip install -r requirements.txt
py -3.9 -m unittest discover -s tests
```

`build_baseline(name, shapes, num_beams)` takes image/CSI shapes and history/future lengths. Synthetic tests show minimal inputs. Ridge consumes a numeric NPZ with `features[N,F]`, `targets[N,T]` and integer/string `groups[N]`; object/pickle arrays are rejected.

```powershell
py -3.9 ridge_baseline.py --arrays ./data/features.npz --output ./outputs/ridge.json
```


WiFo is one model entry in the reference table below. Its [model instructions](docs/WIFO.md) cover optional dependencies in `requirements-wifo.txt` and separately obtained trusted official source, public data and weights. Other baselines use the root requirements.

## Model references and implementation scope

Each row links a relevant paper or GitHub source and describes the actual implementation. Some legacy IDs describe comparison families rather than matching architectures: RGB proxies are 2D CNNs and `convgru`/`temporal3d` are MLPs. Task adaptation is distinct from full-paper reproduction.

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

Exact historical ChannelTransformer10 and MM-BP upstream sources are unresolved; their generic references are explicitly background. Support models are local implementations. Synthetic tests establish software behavior, not published accuracy or real-data generalization.

## Validation and use

The preceding 11 module/load-safety tests passed for synthetic outputs/gradients, support components, Ridge splits and checkpoint loading. WiFo real-asset inference was limited to two public D1 samples and three masks, without training or full-paper evaluation. Syntax and synthetic contracts do not establish paper accuracy or trained generalization; dependency combinations are not exhaustively validated.

This is an unofficial learning/reproduction collection. Cite each original paper and official implementation when using its methods or results. See the [reference summary](docs/REFERENCES.md).

## Source terms

No additional MIT grant is asserted. Learning use and citations do not supply redistribution permission; external projects, dependencies, data and weights retain their own terms. Upstream model source without confirmed terms is obtained separately. No data, weights, private records or experiment archives are bundled.

# Wireless Communication Baselines + WiFo

[中文](README.md) | [English](README.en.md)

用于无线通信相关模型的学习、方法理解和复现练习。集合由本地轻量 baseline 与 WiFo 小样本推理入口组成，各项注明论文/GitHub 来源及实际实现范围。属于非官方学习项目；原作者贡献与本地适配代码分别说明。

当前是待审阅版本，尚未公开。

## 内容与出处

| 模块 | 内容 | 原工作与实现说明 |
|---|---|---|
| 任务 baseline | 14 种 RGB、轨迹、CSI 与波束预测方法 | [逐项引用与架构差异](docs/REFERENCES.md) |
| 层级模型与 VAE | 多模态状态、条件扩散、CSI/波束头、KL 自编码器 | [算法与架构背景](docs/REFERENCES.md) |
| Ridge | 按组隔离、训练集标准化的数值回归 | [scikit-learn 原实现](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py) |
| WiFo | 官方模型的资源校验和两样本推理入口 | [论文](https://doi.org/10.1007/s11432-025-4349-0) · [官方 GitHub](https://github.com/PKU-PCNI/WiFo) · [范围与运行](docs/WIFO.md) |

部分 baseline ID 保留历史名字，但实际是简化代理：`convlstm`/`predrnn_style`/`strictsimvp_style` 为 2D CNN，`convgru`/`temporal3d` 为 MLP。其他方法也有任务适配；引用不表示与原论文架构或指标一致。`ChannelTransformer10` 与 `MM-BP` 的特定上游未确认，只引用可确认的通用算法背景，详见逐项说明。

## 代码结构

| 文件 | 用途 |
|---|---|
| [paper_baselines.py](paper_baselines.py) | 14 种任务 baseline，统一 `BaselineOutput` 接口 |
| [model.py](model.py) / [vae.py](vae.py) | 本地层级模型组件与 KL 自编码器 |
| [ridge_baseline.py](ridge_baseline.py) | 通用 Ridge 数值数组入口 |
| [wifo_inference.py](wifo_inference.py) | WiFo 独立复现入口，调用外部可信官方源码 |
| [tests/](tests/) | 合成输入、梯度、划分及权重加载安全测试 |
| [docs/REFERENCES.md](docs/REFERENCES.md) / [docs/WIFO.md](docs/WIFO.md) | 模型引用、学习范围和 WiFo 资源说明 |

仅包含核心代码、测试和介绍。数据、权重、实验档案、内部协议及私有资料不打包。

## 快速开始

Python 3.9+，从仓库根目录执行：

```powershell
py -3.9 -m pip install -r requirements-wifo.txt
py -3.9 -m unittest discover -s tests -v
```

只使用 baseline 时安装 `requirements.txt`，分别运行 `tests/test_core_models.py` 与 `tests/test_ridge.py`。依赖采用最低版本范围，未锁定全部传递依赖或验证每种安装组合。

`build_baseline(name, shapes, num_beams)` 的最小合成输入见测试。Ridge 接收通用数值 NPZ：`features[N,F]`、`targets[N,T]`、`groups[N]`，拒绝对象/pickle 数组。

```powershell
py -3.9 ridge_baseline.py --arrays ./data/features.npz --output ./outputs/ridge.json
```

WiFo 需另行取得可信官方源码、公开 D1 数据和官方权重，按 [WiFo 说明](docs/WIFO.md) 运行。入口校验权重 SHA-256，并以 `weights_only=True` 加载。

## 验证与使用范围

11 项测试通过，覆盖全部 14 个任务方法的合成输出/梯度、层级组件、Ridge 划分及 WiFo 加载安全；这不证明论文指标或真实数据泛化。WiFo 实际推理验证仅覆盖两个公开 D1 样本和三种掩码；没有完成完整训练或全部数据集评测。

本项目用于学习、研究理解与复现练习，不是原作者官方发布。使用相关方法或成果时请引用原论文与官方代码，而非仅引用本集合。

## 许可与引用

本集合不另行声明 MIT 授权。公开可见性、学习用途和引用说明不等于授予源码再分发许可；外部项目、依赖、数据和权重分别遵守其原始条款。未确认许可的 WiFo 模型源码不随本集合分发。所有方法的原工作及本地实现差异见 [引用表](docs/REFERENCES.md)。

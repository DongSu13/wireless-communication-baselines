# WiFo 学习与小样本复现 / WiFo learning and small-sample inference

[中文项目介绍](../README.md) · [English overview](../README.en.md)

## 原工作与本地实现

WiFo 是用于无线信道预测的预训练基础模型。原工作作者为 Boxun Liu、Shijian Gao、Xuanyu Liu、Xiang Cheng、Liuqing Yang。

- 论文：[WiFo: Wireless Foundation Model for Channel Prediction](https://doi.org/10.1007/s11432-025-4349-0)，SCIENCE CHINA Information Sciences, 68, 162302 (2025)。
- 官方源码：[PKU-PCNI/WiFo](https://github.com/PKU-PCNI/WiFo)，使用修订 `a0889e124aeb9dcc423fc37e6796f77c7c4f7f24`。
- 公开数据：[RF-only channel dataset for WiFo](https://huggingface.co/datasets/pku-pcni-lab/RF-only_channel_dataset_for_WiFo)，这里使用 D1。
- 官方权重：[liuboxun/WiFo](https://huggingface.co/liuboxun/WiFo)，这里使用 `wifo_base.pkl`。

[wifo_inference.py](../wifo_inference.py)是本地复现入口：读入 D1 前两个样本，运行时间、频率和随机掩码，输出有限性检查及 NMSE。它调用可信官方 checkout 的 `WiFo_model`，没有在本仓库重新分发原模型文件。没有训练 WiFo，也没有声称重跑全部数据集或达到原论文指标。三个掩码的任务与比例不同，数值不能混为同一任务排名。

The official work supplies the architecture and pretrained model. This collection contributes an inference adapter for two D1 samples, with temporal, frequency and random masking. It is an unofficial learning/reproduction exercise, not a full training or paper-accuracy reproduction. Different masking tasks are not directly comparable.

## 运行 / Run

在仓库根目录执行。仅使用可信官方源码；`--upstream-source` 会执行该目录中的 Python 代码。首次获取官方资源需自行遵守其资源条款。 / Run from the repository root with a trusted official checkout. The source path is executable Python, not inert data.

```powershell
py -3.9 -m pip install -r requirements-wifo.txt
git clone https://github.com/PKU-PCNI/WiFo ./upstream/WiFo
git -C ./upstream/WiFo checkout a0889e124aeb9dcc423fc37e6796f77c7c4f7f24
py -3.9 wifo_inference.py --upstream-source ./upstream/WiFo --data-file ./data/D1/X_test.mat --checkpoint ./weights/wifo_base.pkl --output ./outputs/wifo.json
```

从上述公开资源页面单独取得 `D1/X_test.mat` 和 `wifo_base.pkl`，按命令所示放置；它们不随仓库打包。 / Download those assets separately from the public pages above; no data or weights are bundled.

入口在导入外部源码和反序列化权重前核对官方 base 权重 SHA-256：`5681e6dfbe3ebd358b01510f476b18a25a6616dcb5b88a31595b288aeba22fce`，随后使用 `weights_only=True`，没有不安全 pickle 回退。 / Official checkpoint bytes are verified before loading with restricted tensor deserialization.

## 验证范围 / Validation scope

baseline 使用合成输入验证输出、梯度及数据划分；WiFo 安全测试验证拒绝未知权重、仅接受 tensor state dictionary、缺少源码时失败和无资源 `--help`。实际推理限于两个公开 D1 样本和官方 base 权重。训练、全量数据集评测、论文精度及新依赖环境的完整兼容性均未验证。

## 来源条款 / Source terms

本项目用于学习与复现练习，并与原作者保持明确归属。WiFo GitHub 源码在上述修订没有明确项目许可证，所以只提供官方来源与调用入口。数据卡和权重卡的 Apache-2.0 声明只适用于各自资源，不能据此为 GitHub 模型源码添加 MIT 或 Apache 许可。 / Citation and a learning-use notice do not grant redistribution permission for the separately hosted model source.

## 官方论文引用 / Official paper citation

```bibtex
@article{liu2025wifo,
  author  = {Liu, Boxun and Gao, Shijian and Liu, Xuanyu and Cheng, Xiang and Yang, Liuqing},
  title   = {WiFo: Wireless Foundation Model for Channel Prediction},
  journal = {Sci. China Inf. Sci.},
  volume  = {68},
  pages   = {162302},
  year    = {2025},
  month   = {May},
  doi     = {10.1007/s11432-025-4349-0}
}
```

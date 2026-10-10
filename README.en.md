# Wireless Communication Baselines + WiFo

[中文](README.md) | [English](README.en.md)

An unofficial collection for learning and reproduction exercises: local lightweight baselines plus a WiFo inference adapter. Every method has a paper/GitHub reference and an explicit implementation scope. This review version has not been made public.

## Contents and attribution

- 14 task baselines for RGB, GPS trajectory, CSI and beam prediction, plus local state/diffusion/CSI/beam components and a KL autoencoder: [method-by-method references](docs/REFERENCES.md).
- Group-isolated Ridge regression with train-only standardization: [scikit-learn implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py).
- WiFo checkpoint verification and two-sample inference: [original paper](https://doi.org/10.1007/s11432-025-4349-0), [official repository](https://github.com/PKU-PCNI/WiFo), and [instructions/citation](docs/WIFO.md).

Legacy names are not exact architecture claims: the ConvLSTM/PredRNN/SimVP-style RGB entries are 2D CNN proxies, and ConvGRU/temporal3d CSI entries are MLPs. Exact ChannelTransformer10 and MM-BP sources remain untraced; their references are explicitly generic background. Consult the reference table before interpreting any method as paper reproduction.

## Code and usage

Core files: [paper_baselines.py](paper_baselines.py), [model.py](model.py), [vae.py](vae.py), [ridge_baseline.py](ridge_baseline.py), [wifo_inference.py](wifo_inference.py), and [tests/](tests/). No data, weights, internal protocols or experiment archives are bundled.

Use Python 3.9+ from the repository root:

```powershell
py -3.9 -m pip install -r requirements-wifo.txt
py -3.9 -m unittest discover -s tests -v
```

For baseline-only use, install `requirements.txt` and run `tests/test_core_models.py` and `tests/test_ridge.py` separately. Requirements use lower bounds; all possible resolutions and transitive dependencies have not been audited.

Ridge accepts numeric NPZ arrays `features[N,F]`, `targets[N,T]`, and `groups[N]`; object/pickle arrays are rejected. WiFo uses a separately obtained trusted official checkout, public D1 data and official base checkpoint, with SHA-256 validation before restricted tensor loading. See [WiFo setup](docs/WIFO.md) for pinned source and exact commands.

## Validation and learning use

11 tests pass for synthetic output/gradient contracts, group isolation and WiFo load safety. WiFo real-asset inference is limited to two public D1 samples and three masks. No full training, complete dataset evaluation, paper-accuracy reproduction or trained generalization is claimed. Current checks do not establish every dependency/environment combination.

This is an unofficial learning and reproduction exercise. Cite the original papers and official code when using their methods or results; the collection does not replace those sources or imply author endorsement.

## Source terms

No additional MIT license is asserted for this collection. Public visibility, citation and a learning-use notice do not grant redistribution rights. External source, dependencies, data and weights retain their own terms. WiFo model source without an explicit project license is obtained separately and is not redistributed here. [References and implementation scope](docs/REFERENCES.md) distinguishes local adaptations from original works.

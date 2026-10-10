"""Two-sample CPU inference using explicitly supplied official WiFo assets."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace


BASE_CHECKPOINT_SHA256 = "5681e6dfbe3ebd358b01510f476b18a25a6616dcb5b88a31595b288aeba22fce"
KNOWN_EXTRA_KEYS = {"decoder_pred_dual.0.weight", "decoder_pred_dual.0.bias",
                    "decoder_pred_dual.2.weight", "decoder_pred_dual.2.bias"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_base_state(checkpoint):
    """Reject nonofficial bytes before deserialization; never allow pickle fallback."""
    if file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise ValueError("checkpoint does not match the referenced official WiFo base asset")
    import torch
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                                               for key, value in state.items()):
        raise ValueError("expected a tensor-only state dictionary")
    return state


def load_upstream_factory(source):
    """Execute only the local checkout explicitly selected by the caller."""
    source = Path(source).resolve(strict=True)
    if (source / "src").is_dir():
        source = source / "src"
    for filename in ("model.py", "Embed.py", "mask_strategy.py"):
        if not (source / filename).is_file():
            raise ValueError("upstream checkout is missing a required source module")
    spec = importlib.util.spec_from_file_location("external_wifo_model", source / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module.WiFo_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-source", type=Path, required=True, help="Trusted upstream checkout or its src directory")
    parser.add_argument("--data-file", type=Path, required=True, help="Public WiFo D1/X_test.mat")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Official wifo_base.pkl")
    parser.add_argument("--output", type=Path, required=True, help="External JSON output; reports contain no input paths")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    data = args.data_file.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    upstream = args.upstream_source.resolve(strict=True)
    output = args.output.resolve()
    if output in (data, checkpoint) or upstream in output.parents or output == upstream:
        parser.error("output must be outside the upstream source and must not overwrite input assets")

    # Check checkpoint integrity before importing external source or loading its tensors.
    state = load_official_base_state(checkpoint)
    import h5py
    import numpy as np
    import torch
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    factory = load_upstream_factory(upstream)
    model = factory(SimpleNamespace(size="base", t_patch_size=4, patch_size=4,
                                    pos_emb="SinCos_3D", no_qkv_bias=0)).eval()
    loaded = model.load_state_dict(state, strict=False)
    if loaded.missing_keys or set(loaded.unexpected_keys) - KNOWN_EXTRA_KEYS:
        raise ValueError("incompatible upstream architecture or checkpoint keys")

    with h5py.File(data, "r") as dataset:
        values = dataset["X_val"]
        if values.ndim != 4 or values.shape[-1] < 2:
            raise ValueError("expected official D1 MATLAB HDF5 input with at least two samples")
        raw = values[..., :2].transpose(3, 2, 1, 0)
    if raw.dtype.names is None or not {"real", "imag"}.issubset(raw.dtype.names):
        raise ValueError("expected complex MATLAB real/imag fields")
    channels = raw["real"] + 1j * raw["imag"]
    x = torch.from_numpy(np.stack((channels.real, channels.imag), axis=1)).float()
    if tuple(x.shape) != (2, 2, 24, 4, 128) or not torch.isfinite(x).all():
        raise ValueError("expected finite D1 tensor with shape [2,2,24,4,128]")

    results = {}
    with torch.no_grad():
        for strategy, ratio in (("temporal", 0.5), ("fre", 0.5), ("random", 0.85)):
            torch.manual_seed(args.seed)
            loss, _, prediction, target, mask = model([sample.unsqueeze(0) for sample in x],
                                                      mask_ratio=ratio, mask_strategy=strategy, data="D1")
            if not all(torch.isfinite(v).all() for v in (loss, prediction, target, mask)):
                raise ValueError("nonfinite inference output")
            predicted = prediction[mask == 1].reshape(len(x), -1).numpy()
            actual = target[mask == 1].reshape(len(x), -1).numpy()
            power = np.mean(np.abs(actual) ** 2, axis=1)
            if not np.all(power > 0):
                raise ValueError("masked target must have nonzero power")
            nmse = float(np.mean(np.mean(np.abs(predicted - actual) ** 2, axis=1) / power))
            if not np.isfinite(nmse):
                raise ValueError("nonfinite NMSE")
            results[strategy] = {"mask_ratio": ratio, "nmse": nmse, "loss": float(loss)}
    record = {"scope": "two public D1 samples; CPU base-model inference only", "seed": args.seed,
              "checkpoint_sha256": BASE_CHECKPOINT_SHA256, "input_shape": list(x.shape),
              "checkpoint_missing_keys": loaded.missing_keys, "checkpoint_extra_keys": loaded.unexpected_keys,
              "results": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

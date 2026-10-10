"""Numerical contracts for public core algorithms using synthetic tensors only."""
import unittest

import torch

from paper_baselines import build_baseline
from model import BeamHead, CoarseWorldModel, FineCSIHead, LatentDiffusion
from vae import AutoencoderKL, vae_loss


TASKS = {
    "convlstm": "rgb", "predrnn_style": "rgb", "strictsimvp_style": "rgb",
    "gru": "gps", "lstm": "gps", "tcn": "gps", "wayformer_style": "gps",
    "convgru": "fine_csi", "temporal3d": "fine_csi", "channeltransformer10_style": "fine_csi",
    "gmu": "beam_logits", "crossattn": "beam_logits", "decisionfusion": "beam_logits", "rgb_csi_mmbp_style": "beam_logits",
}


class CoreContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_methods_have_one_correct_output_and_finite_gradients(self):
        shapes = {"image": (3, 8, 8), "csi": (2, 2, 3), "history": 3,
                  "pred_steps": 5, "fine_history": 8, "fine_future": 8}
        batch = {"rgb_history": torch.rand(2, 3, 3, 8, 8), "gps_history": torch.rand(2, 3, 2),
                 "csi_history": torch.rand(2, 3, 2, 2, 3), "fine_csi_context": torch.rand(2, 8, 2, 2, 3)}
        expected = {"rgb": (2, 5, 3, 8, 8), "gps": (2, 5, 2),
                    "fine_csi": (2, 8, 2, 2, 3), "beam_logits": (2, 5, 16)}
        for method, task in TASKS.items():
            with self.subTest(method=method):
                torch.manual_seed(7)
                model = build_baseline(method, shapes, num_beams=16)
                output = model(batch)
                values = [getattr(output, key) for key in ("rgb", "gps", "coarse_csi", "fine_csi", "beam_logits")]
                self.assertEqual(sum(value is not None for value in values), 1)
                value = getattr(output, task)
                self.assertEqual(tuple(value.shape), expected[task])
                self.assertTrue(torch.isfinite(value).all())
                value.square().mean().backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            build_baseline("unknown", {}, num_beams=16)

    def test_hierarchical_support_heads_and_diffusion_are_finite(self):
        world = CoarseWorldModel(latent_channels=4, d_model=32, depth=1, heads=4).eval()
        latent = torch.rand(2, 3, 4, 8, 8)
        csi = torch.rand(2, 3, 2, 4, 8)
        gps = torch.rand(2, 3, 2)
        state, condition, stat, position = world(latent, csi, gps)
        self.assertEqual(tuple(position.shape), (2, 2))
        fine = FineCSIHead((2, 4, 8), d_model=32, heads=4, history=3, future=2)
        predicted = fine(csi, stat)
        self.assertEqual(tuple(predicted.shape), (2, 2, 2, 4, 8))
        self.assertEqual(tuple(BeamHead(32, 16)(stat).shape), (2, 16))
        diffusion = LatentDiffusion(channels=4, base=32, cond_dim=32, diffusion_steps=4)
        loss = diffusion.loss(latent[:, 0], condition)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        sample = diffusion.sample(latent[:, 0].shape, condition.detach(), sample_steps=2)
        self.assertEqual(tuple(sample.shape), tuple(latent[:, 0].shape))
        self.assertTrue(torch.isfinite(sample).all())

    def test_autoencoder_reconstruction_and_loss(self):
        model = AutoencoderKL(base=8)
        images = torch.rand(2, 3, 16, 16) * 2 - 1
        prediction, mean, logvar = model(images)
        self.assertEqual(tuple(prediction.shape), tuple(images.shape))
        loss, metrics = vae_loss(prediction, images, mean, logvar)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        loss.backward()


if __name__ == "__main__":
    unittest.main()

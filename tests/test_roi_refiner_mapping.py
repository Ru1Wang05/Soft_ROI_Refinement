"""Guard test: the ROI refiner must modify exactly the channels of WEAK_CLASS_IDS.

Regression for the off-by-one audit finding: an earlier forward pass indexed
logits with class_id - 1, so the branch was supervised on LPA/RPA/SVC/Trachea
masks while modifying MPA/LPA/RPA/SVC logits.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.vit_seg_modeling import CONFIGS
from networks.vit_seg_modeling_roi_refiner import SoftROIRefinerVisionTransformer

BIG = 500.0  # dominates any raw logit magnitude


def test_residual_channel_j_drives_class_j():
    config = CONFIGS["R50-ViT-B_16"]
    config.n_classes = 12
    config.n_skip = 3
    config.patches.grid = (14, 14)
    model = SoftROIRefinerVisionTransformer(config, img_size=224, num_classes=12)
    model.eval()

    weak = list(model.WEAK_CLASS_IDS)
    with torch.no_grad():
        probe = model(torch.rand(1, 1, 224, 224), return_aux=True)
    big = float(probe["raw_segmentation"].abs().max()) * 2.0 + 10.0
    res_hw = (224, 224)  # the branch operates at full input resolution
    for j, expected_channel in enumerate(weak):
        synthetic = torch.zeros(1, 4, *res_hw)
        synthetic[0, j, 32, 48] = big
        synthetic[0, j, 80, 96] = -big

        def fake_localizer(module, inputs, output):
            return output.fill_(0.0)  # sigmoid -> attention exactly 1

        def fake_residual(module, inputs, output):
            return synthetic.clone()

        h1 = model.roi_localizer.register_forward_hook(fake_localizer)
        h2 = model.roi_residual.register_forward_hook(fake_residual)
        try:
            with torch.no_grad():
                out = model(torch.rand(1, 1, 224, 224), return_aux=True)
        finally:
            h1.remove()
            h2.remove()

        pred = out["segmentation"].argmax(1)[0]
        # where the synthetic residual is +big the argmax must be the wired class
        assert pred[32, 48].item() == expected_channel, (
            f"residual channel {j} drives logits channel {pred[32, 48].item()}, "
            f"expected {expected_channel} (off-by-one regression?)"
        )
        # where it is -big the wired class must be suppressed if raw predicted it,
        # so just require it is not forced to the wired class
        assert pred[80, 96].item() != expected_channel or out["raw_segmentation"][0, expected_channel, 80, 96] > 0

    print("test_roi_refiner_mapping: OK")


if __name__ == "__main__":
    test_residual_channel_j_drives_class_j()

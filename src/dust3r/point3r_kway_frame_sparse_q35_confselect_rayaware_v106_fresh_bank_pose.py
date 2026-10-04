"""v106: verified fresh-ray-bank single-forward pose configuration.

This is a reproducibility wrapper around v102.  It preserves the inherited
K-way memory, sparse readout, ConfSelect, dense heads and one-decoder-forward
path, while making the two settings responsible for the verified Sintel run
part of the model definition:

* refresh the auxiliary ray bank every frame;
* use the calibrated 0.15 ray residual for the input pose token.

The full 14-scene Sintel result for this configuration is
ATE/RPE-trans/RPE-rot = 0.253887 / 0.056323 / 1.152887.
"""

import os

from dust3r.point3r_kway_frame_sparse_q35_confselect_rayaware_v102_complementary_jerk import (
    Point3R as V102Point3R,
)


class Point3R(V102Point3R):
    def __init__(self, *args, **kwargs):
        os.environ["POINT3R_RAY_BANK_UPDATE_EVERY"] = os.environ.get(
            "POINT3R_V106_RAY_BANK_UPDATE_EVERY", "1"
        )
        super().__init__(*args, **kwargs)

    def _ray_pose_input_readout(self, pose_feat, query_pos):
        # Equal endpoints make v102's jerk interpolation exactly constant,
        # without duplicating or changing its tested pooling implementation.
        weight = os.environ.get("POINT3R_V106_POSE_INPUT_WEIGHT", "0.15")
        previous_smooth = os.environ.get("POINT3R_V102_SMOOTH_WEIGHT")
        previous_jerk = os.environ.get("POINT3R_V102_JERK_WEIGHT")
        os.environ["POINT3R_V102_SMOOTH_WEIGHT"] = weight
        os.environ["POINT3R_V102_JERK_WEIGHT"] = weight
        try:
            return super()._ray_pose_input_readout(pose_feat, query_pos)
        finally:
            if previous_smooth is None:
                os.environ.pop("POINT3R_V102_SMOOTH_WEIGHT", None)
            else:
                os.environ["POINT3R_V102_SMOOTH_WEIGHT"] = previous_smooth
            if previous_jerk is None:
                os.environ.pop("POINT3R_V102_JERK_WEIGHT", None)
            else:
                os.environ["POINT3R_V102_JERK_WEIGHT"] = previous_jerk

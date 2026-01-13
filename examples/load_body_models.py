import numpy as np

from aitviewer.models.smpl import MANOLayer, SMPLLayer
from aitviewer.renderables.smpl import MANOSequence, SMPLSequence
from aitviewer.viewer import Viewer


def main():
    # Layers
    mano_layer = MANOLayer(
        is_rhand=True,
        flat_hand_mean=False,
    )
    smplh_layer = SMPLLayer(
        model_type="smplh",
        gender="neutral",
        flat_hand_mean=False,
    )
    smplx_layer = SMPLLayer(
        model_type="smplx",
        gender="neutral",
        flat_hand_mean=False,
    )

    # Sequences
    num_frames = 90

    mano_betas = np.zeros(10)
    mano_poses_hand = np.zeros((num_frames, 15 * 3))
    mano_poses_root = np.zeros((num_frames, 3))
    mano_trans = np.zeros((num_frames, 3))
    mano_sequence = MANOSequence(
        poses_hand=mano_poses_hand,
        mano_layer=mano_layer,
        poses_root=mano_poses_root,
        betas=mano_betas,
        trans=mano_trans,
        name="mano",
    )

    smplh_betas = np.zeros(10)
    smplh_poses_body = np.zeros((num_frames, 21 * 3))
    smplh_poses_left_hand = np.zeros((num_frames, 15 * 3))
    smplh_poses_right_hand = np.zeros((num_frames, 15 * 3))
    smplh_poses_root = np.zeros((num_frames, 3))
    smplh_trans = np.zeros((num_frames, 3))
    smplh_sequence = SMPLSequence(
        poses_body=smplh_poses_body,
        smpl_layer=smplh_layer,
        poses_root=smplh_poses_root,
        betas=smplh_betas,
        trans=smplh_trans,
        poses_left_hand=smplh_poses_left_hand,
        poses_right_hand=smplh_poses_right_hand,
        name="smplh",
    )

    smplx_betas = np.zeros(10)
    smplx_poses_body = np.zeros((num_frames, 21 * 3))
    smplx_poses_left_hand = np.zeros((num_frames, 15 * 3))
    smplx_poses_right_hand = np.zeros((num_frames, 15 * 3))
    smplx_poses_root = np.zeros((num_frames, 3))
    smplx_trans = np.zeros((num_frames, 3))
    smplx_sequence = SMPLSequence(
        poses_body=smplx_poses_body,
        smpl_layer=smplx_layer,
        poses_root=smplx_poses_root,
        betas=smplx_betas,
        trans=smplx_trans,
        poses_left_hand=smplx_poses_left_hand,
        poses_right_hand=smplx_poses_right_hand,
        name="smplx",
    )

    # Visualize
    v = Viewer()
    assert v.scene is not None
    v.scene.light_mode = "diffuse"
    v.scene.add(mano_sequence)
    v.scene.add(smplh_sequence)
    v.scene.add(smplx_sequence)
    v.run()


if __name__ == "__main__":
    main()

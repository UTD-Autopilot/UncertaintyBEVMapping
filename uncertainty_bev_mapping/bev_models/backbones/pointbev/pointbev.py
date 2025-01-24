import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import einops

from .models.img_encoder import EfficientNet
from .models.img_encoder import AGPNeck
from .models.img_encoder.neck import AlignRes, PrepareChannel
from .models.heads import BEVConvHead
from .models.sampled import PointBeV
from .models.projector import CamProjector
from .models.view_transform import GridSampleVT
from .models.autoencoder import SparseUNet
from .models.layers.attention import PositionalEncodingMap

from scipy.spatial.transform import Rotation as R

def get_random_ref_matrix(trans_rot):
    """
    Use scipy to create a random reference transformation matrix.
    """
    trans_coeff, rot_coeff = trans_rot[:3], trans_rot[3:]

    # Initialize in homogeneous coordinates.
    mat = np.eye(4, dtype=np.float64)

    # Translate
    mat[:3, 3] = (np.random.random((3)).astype(np.float32) * 2 - 1) * np.array(
        trans_coeff
    )

    # Rotate
    random_zyx = (np.random.random((3)).astype(np.float32) * 2 - 1) * np.array(
        rot_coeff
    )
    mat[:3, :3] = R.from_euler("zyx", random_zyx, degrees=True).as_matrix()

    return mat

class PointBEV(nn.Module):
    def __init__(
        self,
        x_bound=(-50.0, 50.0, 0.5), # unused, check spatial_bounds variable instead
        y_bound=(-50.0, 50.0, 0.5),
        z_bound=(-10.0, 10.0, 20.0),
        n_classes=4,
    ):
        super().__init__()
        self.n_classes = n_classes
        
        self.trans_rot = [30.,20.,0.,20.,0.,0.]

        keep_input_centr_offs = False
        keep_input_binimg = True
        keep_input_hdmap = False
        hdmap_names = ['vehicle', 'background']
        
        projector_shape = [200, 200, 8]
        out_c_base_neck = 128
        out_c_N_group = 1
        
        backbone_checkpoint_path = "data/"
        out_c_backbone = [56, 160]
        self.backbone = EfficientNet(backbone_checkpoint_path, version='b4', downsample=8)

        in_c_neck = out_c_backbone
        neck_interm_c = out_c_base_neck
        out_c_neck = out_c_base_neck * out_c_N_group
        
        in_c_vt = out_c_neck
        out_c_vt = out_c_base_neck
        
        out_c_autoencoder = out_c_vt

        self.neck = AGPNeck(
            align_res_layer=AlignRes(
                mode="upsample",
                scale_factors=[1, 2],
                in_channels=in_c_neck,
            ),
            prepare_c_layer=PrepareChannel(
                in_channels=in_c_neck,
                interm_c=neck_interm_c,
                out_c=out_c_neck,
                mode="doubleconv",
                tail_mode="conv2d",
            ),
        )
        
        self.heads = BEVConvHead(
            shared_out_c=out_c_autoencoder,
            num_classes=n_classes,
            with_centr_offs=keep_input_centr_offs,
            with_binimg=keep_input_binimg,
            with_hdmap=keep_input_hdmap,
            hdmap_names=hdmap_names,
            dense_input=False,
        )
        
        spatial_bounds = [-49.75, 49.75, -49.75, 49.75, -3.375, 5.375]
        voxel_ref = "spatial"
        self.projector = CamProjector(
            spatial_bounds=spatial_bounds,
            voxel_ref=voxel_ref,
            z_value_mode="zero",
        )

        self.view_transform = GridSampleVT(
            voxel_shape=projector_shape,
            in_c=in_c_vt,
            out_c=out_c_vt,
            N_group=out_c_N_group,
            grid_sample_mode="sparse_optim",
            coordembd=PositionalEncodingMap(
                m=8,
                with_mlp=True,
                in_c=3,
                out_c=out_c_vt,
                num_hidden_layers=2,
                mid_c=out_c_vt*2,
            ),
            heightcomp=dict(
                comp=dict(
                    classname="uncertainty_bev_mapping.bev_models.backbones.pointbev.models.layers.MLP",
                    mode="mlp",
                    in_c=in_c_vt,
                    mid_c=out_c_vt,
                    out_c=out_c_vt,
                    num_layers=4,
                    as_conv=True,
                ),
            ),
            input_sparse=True,
            return_sparse=True,
        )
        
        self.autoencoder = SparseUNet(
            in_c=out_c_vt,
            with_tail_conv=False,
            with_large_kernels=False,
            with_decoder_bias=False,
        )

        self.model = PointBeV(
            backbone=self.backbone,
            neck=self.neck,
            heads=self.heads,
            projector=self.projector,
            view_transform=self.view_transform,
            autoencoder=self.autoencoder,
            temporal=None,
            in_shape=dict(
                projector=projector_shape,
                spatial_bounds=spatial_bounds,
            ),
            voxel_ref="spatial",
            in_c=dict(
                neck=in_c_neck,
                vt=out_c_vt,
            ),
            out_c=dict(
                base_neck=out_c_base_neck,
                N_group=out_c_N_group,
                neck=out_c_neck,
                vt=out_c_vt,
                autoencoder=out_c_autoencoder,
            ),
            sampled_kwargs=dict(
                N_coarse=2500,
                # mode="dense",
                mode="rnd_pillars",
                val_mode="dense",
                patch_size=1,
                compress_height=False,
                with_fine=True,
                valid_fine=False,
                N_fine=2500,
                N_anchor=100,
                fine_patch_size=7,
                fine_thresh=0.2,
                temp_thresh=-5,
                reinject_pts=True,
            )
        )
        self.bev_aug = nn.Parameter(torch.eye(4, dtype=torch.float32))
        self.output_conv = nn.Conv2d(1, n_classes, 1)


    def forward(self, images, intrinsics, extrinsics):
        device = images.device
        B, S, C, H, W = images.shape
        
        imgs = einops.repeat(images, 'B S C H W -> B 1 S C H W')
        
        rots = einops.repeat(extrinsics[:, :, :3, :3], 'B S X Y -> B 1 S X Y')
        trans = einops.repeat(extrinsics[:, :, :3, 3], 'B S X -> B 1 S X 1')

        intrins = einops.repeat(intrinsics, 'B S X Y -> B 1 S X Y')
        
        egoTin_to_seq = torch.eye(4, device=device)
        egoTin_to_seq = einops.repeat(egoTin_to_seq, 'X Y -> B 1 X Y', B=B)

        # bev_aug = torch.eye(4, device=device)
        bev_aug = einops.repeat(self.bev_aug, 'X Y -> B 1 X Y', B=B)

        dict_out = self.model(imgs, rots, trans, intrins, bev_aug, egoTin_to_seq)
        
        # # binimg
        # bev_semantic = einops.rearrange(dict_out['bev']['binimg'], 'B 1 C H W -> B C H W')
        # masks = dict_out['masks']['bev']['binimg']
        # masks = einops.rearrange(masks, 'B 1 1 H W -> B H W')
        # # masks = einops.repeat(masks, 'B H W -> B C H W', C=C)
        # # Force masked region to be background, pre-sigmoid value
        # bev_semantic[:, 0][masks == 0] = -5.0
        # bev_semantic[:, 1][masks == 0] = 5.0
        
        # # hdmap
        # bev_semantic = einops.rearrange(dict_out['bev']['hdmap'], 'B 1 C H W -> B C H W')
        # masks = dict_out['masks']['bev']['hdmap']
        # masks = einops.rearrange(masks, 'B 1 1 H W -> B H W')
        # # masks = einops.repeat(masks, 'B H W -> B C H W', C=C)
        # # Force masked region to be background, pre-sigmoid value
        # bev_semantic[:, 0][masks == 0] = -5.0
        # bev_semantic[:, 1][masks == 0] = 5.0

        bev_semantic = einops.rearrange(dict_out['bev']['binimg'], 'B 1 1 H W -> B 1 H W')
        bev_semantic = self.output_conv(bev_semantic)
        return bev_semantic

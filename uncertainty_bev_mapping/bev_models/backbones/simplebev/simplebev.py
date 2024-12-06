import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .nets.segnet import Segnet
from . import utils


# the scene centroid is defined wrt a reference camera,
# which is usually random
scene_centroid_x = 0.0
scene_centroid_y = 1.0 # down 1 meter
scene_centroid_z = 0.0

Z, Y, X = 200, 8, 200

scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()

class SimpleBEV(nn.Module):
    def __init__(
        self,
        x_bound=(-50.0, 50.0, 0.5),
        y_bound=(-50.0, 50.0, 0.5),
        z_bound=(-10.0, 10.0, 20.0),
        n_classes=4,
        encoder_type='res101',
        rand_flip=True,
        do_rgbcompress=True,
    ):
        super().__init__()

        ncams = 6
        nsweeps = 3
        res_scale = 2
        
        bounds = (x_bound[0], x_bound[1], y_bound[0], y_bound[1], z_bound[0], z_bound[1])
        
        self.vox_util = utils.vox.Vox_util(
            Z, Y, X,
            scene_centroid=scene_centroid,#.to(device),
            bounds=bounds,
            assert_cube=False
        )
        
        self.model = Segnet(
            Z, Y, X,
            self.vox_util,
            use_radar=False,
            use_lidar=False,
            use_metaradar=False,
            do_rgbcompress=do_rgbcompress,
            encoder_type=encoder_type,
            rand_flip=rand_flip,
        )
    
    def forward(self, images, intrinsics, extrinsics):
        device = images.device
        B, S, C, H, W = images.shape

        images = images[:, [1, 0, 2, 3, 4, 5], ...]
        intrinsics = intrinsics[:, [1, 0, 2, 3, 4, 5], ...]
        extrinsics = extrinsics[:, [1, 0, 2, 3, 4, 5], ...]

        rgb_camXs = images

        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
    
        intrins_ = __p(intrinsics)
        pix_T_cams_ = utils.geom.merge_intrinsics(*utils.geom.split_intrinsics(intrins_)).to(device)
        pix_T_cams = __u(pix_T_cams_)
        
        rots = extrinsics[:, :, :3, :3]
        trans = extrinsics[:, :, :3, 3]

        velo_T_cams = utils.geom.merge_rtlist(rots, trans).to(device)
        cams_T_velo = __u(utils.geom.safe_inverse(__p(velo_T_cams)))
        
        cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)
        camXs_T_cam0 = __u(utils.geom.safe_inverse(__p(cam0_T_camXs)))
        cam0_T_camXs_ = __p(cam0_T_camXs)

        _, feat_bev_e, seg_bev_e, center_bev_e, offset_bev_e = self.model(
            rgb_camXs=rgb_camXs,
            pix_T_cams=pix_T_cams,
            cam0_T_camXs=cam0_T_camXs,
            vox_util=self.vox_util,
            rad_occ_mem0=None,
        )
        return seg_bev_e

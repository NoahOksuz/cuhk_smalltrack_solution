"""Input views of the cached Depth_Color RGB + IR clips, shared by training and inference.

combined  depth JET colours to the pretrained RGB filters, IR in the extra channel
ir        IR replicated to every channel (best single view)
irdepth   IR to the RGB filters, scalar depth (inverted JET) in the extra channel
depthz    scalar depth replicated to every channel
depth     depth JET colours plus their mean
irnorm    IR with a per-clip 1-99 percent contrast stretch
"""
import numpy as np
import torch


_JET_INVERSE = None


def depth_scalar(rgb):
    """Invert the OpenCV JET colormap of Depth_Color (exact on raw frames) to near=1 .. far=0.

    rgb: (3, T, H, W) in [0, 1]. Bilinear resizing blends colours at edges, so each
    channel is binned to 32 levels and mapped through a nearest-colour table; pixels
    with no channel above 0.4 are invalid depth (0).
    """
    global _JET_INVERSE
    if _JET_INVERSE is None:
        import cv2
        lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[None], cv2.COLORMAP_JET)[0][:, ::-1].astype(np.float32)
        centres = np.arange(32, dtype=np.float32) * 8 + 3.5
        grid = np.stack(np.meshgrid(centres, centres, centres, indexing='ij'), -1).reshape(-1, 3)
        nearest = np.concatenate([((part[:, None] - lut[None]) ** 2).sum(-1).argmin(1) for part in np.array_split(grid, 16)])
        valid = grid.max(1) >= .4 * 255
        _JET_INVERSE = np.where(valid, 1 - nearest / 255., 0.).astype(np.float32)
    u = (rgb.numpy() * 255 + .5).astype(np.uint8) >> 3
    index = (u[0].astype(np.int32) << 10) | (u[1].astype(np.int32) << 5) | u[2]
    return torch.from_numpy(_JET_INVERSE[index])


def input_channels(x, view):
    if view == 'ir':
        return x[3:4].expand(4, -1, -1, -1).clone()
    if view == 'depth':
        return torch.cat([x[:3], x[:3].mean(0, keepdim=True)], dim=0)
    if view == 'irdepth':
        # Pretrained RGB filters see IR as a grey image; scalar depth (inverted JET)
        # enters through the extra channel. 'combined' did the opposite and trails IR-only.
        return torch.cat([x[3:4].expand(3, -1, -1, -1), depth_scalar(x[:3])[None]], dim=0)
    if view == 'depthz':
        return depth_scalar(x[:3])[None].expand(4, -1, -1, -1).clone()
    if view == 'irnorm':
        # Per-clip contrast stretch. Screened on fold 0 and stopped: it tracked plain IR.
        ir = x[3:4]
        lo, hi = torch.quantile(ir.reshape(-1)[::7], torch.tensor([.01, .99]))
        return ((ir - lo) / (hi - lo).clamp_min(.05)).clamp(0, 1).expand(4, -1, -1, -1).clone()
    return x

"""Signed low-bit packing with group-wise scales.

A single scale per output channel spans thousands of weights, which is why 3-bit
collapses and 4-bit costs about three points. Scaling every `group` consecutive
weights instead costs 16 bits per group - 0.25 bits per weight at group 64 - and
recovers most of that loss.
"""
import math

import numpy as np
import torch


def pack_codes(q, bits):
    a = q.detach().cpu().numpy().astype(np.int16).reshape(-1)
    codes = (a & ((1 << bits) - 1)).astype(np.uint8)
    binary = ((codes[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).reshape(-1)
    return torch.from_numpy(np.packbits(binary, bitorder='little'))


def unpack_codes(packed, count, bits):
    binary = np.unpackbits(packed.cpu().numpy(), bitorder='little', count=count * bits).reshape(count, bits)
    code = (binary.astype(np.int32) * (1 << np.arange(bits))).sum(1)
    code = np.where(code >= (1 << (bits - 1)), code - (1 << bits), code)
    return torch.from_numpy(code.astype(np.int8))


def pack(state, bits, group=64):
    assert bits in (3, 4, 5, 6, 8)
    result = {}
    limit = (1 << (bits - 1)) - 1
    for key, value in state.items():
        value = value.detach().cpu()
        if not (value.is_floating_point() and value.ndim >= 2):
            result[key] = value.half() if value.is_floating_point() else value
            continue
        flat = value.float().reshape(-1)
        n = flat.numel()
        pad = (-n) % group
        padded = torch.cat([flat, flat.new_zeros(pad)]) if pad else flat
        blocks = padded.reshape(-1, group)
        scale = (blocks.abs().amax(1, keepdim=True) / limit).clamp_min(1e-8).half()
        q = (blocks / scale.float()).round().clamp(-limit, limit).to(torch.int8)
        packed = pack_codes(q, bits)
        assert torch.equal(unpack_codes(packed, q.numel(), bits).reshape(q.shape), q)
        result[key] = dict(packed=packed, shape=tuple(value.shape), bits=bits,
                           group=group, pad=pad, scale=scale)
    return result


def unpack(state):
    out = {}
    for key, value in state.items():
        if not isinstance(value, dict):
            out[key] = value
            continue
        group, pad = value['group'], value['pad']
        n = math.prod(value['shape'])
        q = unpack_codes(value['packed'], n + pad, value['bits']).reshape(-1, group).float()
        flat = (q * value['scale'].float()).reshape(-1)
        out[key] = flat[:n].reshape(value['shape'])
    return out


def nbytes(packed):
    return sum((v['packed'].numel() + v['scale'].numel() * 2) if isinstance(v, dict)
               else v.numel() * v.element_size() for v in packed.values())

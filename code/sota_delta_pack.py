"""Lossless delta coding of independently group-quantized checkpoints.

The base and each member have their own quantization scales. Byte deltas and
scale XORs only change storage; decoded tensors equal sota_pack2.unpack.
`codec='lzma'` spends more encode time for about 8 MB less per ensemble at int6
(context modelling beats zlib on quantized codes); decoded tensors are identical.
"""
import lzma
import math
import zlib
import numpy as np
import torch

CODECS = {
    'zlib': (lambda b: zlib.compress(b, 6), zlib.decompress),
    'lzma': (lambda b: lzma.compress(b, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME), lzma.decompress),
}


def quantize(value, bits, group):
    flat = value.detach().cpu().float().reshape(-1)
    n = flat.numel()
    pad = (-n) % group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, group)
    limit = (1 << (bits - 1)) - 1
    scale = (blocks.abs().amax(1, keepdim=True) / limit).clamp_min(1e-8).half()
    q = (blocks / scale.float()).round().clamp(-limit, limit).to(torch.int8)
    return q.numpy(), scale.numpy().view(np.uint16)


def requantize(ref, bits, group):
    """Deterministically re-express a decoded base at another bit width.

    Used only as the delta reference for a member stored at `bits`; the member's
    own codes still come from its independent quantization, so decoding stays exact.
    """
    q, scale = ref
    flat = (torch.from_numpy(q.astype(np.int8)).float() * torch.from_numpy(scale.view(np.float16)).float()).reshape(-1, group)
    limit = (1 << (bits - 1)) - 1
    s = (flat.abs().amax(1, keepdim=True) / limit).clamp_min(1e-8).half()
    return (flat / s.float()).round().clamp(-limit, limit).to(torch.int8).numpy(), s.numpy().view(np.uint16)


def encode(state, bits=6, group=128, base=None, codec='zlib', base_bits=None):
    compress = CODECS[codec][0]
    result, reference = {}, {}
    for key, value in state.items():
        value = value.detach().cpu()
        if not (value.is_floating_point() and value.ndim >= 2):
            result[key] = value.half() if value.is_floating_point() else value
            continue
        q, scale = quantize(value, bits, group)
        reference[key] = (q, scale)
        mode = 'direct'
        qc, sc = compress(q.tobytes()), compress(scale.tobytes())
        if base is not None:
            bq, bs = base[key] if base_bits in (None, bits) else requantize(base[key], bits, group)
            assert q.shape == bq.shape and scale.shape == bs.shape
            dq = compress((q.view(np.uint8) - bq.view(np.uint8)).tobytes())
            ds = compress(np.bitwise_xor(scale, bs).tobytes())
            if len(dq) + len(ds) < len(qc) + len(sc):
                qc, sc, mode = dq, ds, 'delta'
        # Tensor storage keeps compressed bytes binary. Pickle protocol 2
        # expands a Python bytes payload into a larger Unicode representation.
        result[key] = dict(q=torch.from_numpy(np.frombuffer(qc, dtype=np.uint8).copy()),
                           scale=torch.from_numpy(np.frombuffer(sc, dtype=np.uint8).copy()),
                           mode=mode, shape=tuple(value.shape), bits=bits, group=group, codec=codec)
        if mode == 'delta' and base_bits not in (None, bits):
            result[key]['base_bits'] = base_bits
    return result, reference


def decode(packed, base=None):
    state, reference = {}, {}
    for key, value in packed.items():
        if not isinstance(value, dict):
            state[key] = value
            continue
        n, group = math.prod(value['shape']), value['group']
        count = n + (-n) % group
        def payload(v):
            return v.cpu().numpy().tobytes() if isinstance(v, torch.Tensor) else v
        decompress = CODECS[value.get('codec', 'zlib')][1]
        qb, sb = decompress(payload(value['q'])), decompress(payload(value['scale']))
        assert len(qb) == count and len(sb) == count // group * 2
        q = np.frombuffer(qb, dtype=np.uint8).reshape(-1, group).copy()
        scale = np.frombuffer(sb, dtype=np.uint16).reshape(-1, 1).copy()
        if value['mode'] == 'delta':
            assert base is not None, 'Delta member requires its exact base'
            bq, bs = base[key] if 'base_bits' not in value else requantize(base[key], value['bits'], group)
            q += bq.view(np.uint8)
            scale ^= bs
        else:
            assert value['mode'] == 'direct'
        q = q.view(np.int8)
        reference[key] = (q, scale)
        flat = (torch.from_numpy(q).float() * torch.from_numpy(scale.view(np.float16)).float()).reshape(-1)
        state[key] = flat[:n].reshape(value['shape'])
    return state, reference

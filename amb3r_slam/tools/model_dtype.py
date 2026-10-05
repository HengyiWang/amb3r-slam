"""Run a model's transformer encoder in bf16 while its prediction heads stay in fp32."""

import torch


def _to_fp32(_module, _inputs, out):
    """Cast bf16 tensors in a module's output back to fp32, preserving structure."""
    def cv(x):
        if torch.is_tensor(x):
            return x.float() if x.dtype == torch.bfloat16 else x
        if isinstance(x, (list, tuple)):
            return type(x)(cv(y) for y in x)
        if isinstance(x, dict):
            return {k: cv(v) for k, v in x.items()}
        return x
    return cv(out)


def cast_bf16(modules, verbose=True):
    """Hold the parameters of each ``(name, module)`` in bf16 and return its outputs in
    fp32. Buffers keep their dtype, so fixed tables such as rotary frequencies stay exact."""
    done = []
    for name, mod in modules:
        for p in mod.parameters():
            p.data = p.data.to(torch.bfloat16)
        mod.register_forward_hook(_to_fp32)
        done.append((name, sum(p.numel() for p in mod.parameters())))
    if verbose:
        print(f'[bf16] {sum(n for _, n in done) * 2 / 1024 ** 3:.2f} GB of encoder weights in '
              f'bf16 ({", ".join(n for n, _ in done) or "NONE"})', flush=True)
    return done

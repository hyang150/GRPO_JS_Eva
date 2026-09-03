"""Phase 0 gate: does this box actually run bf16 on sm_120?

Nothing downstream is worth writing until this passes.  The current base
install (torch 2.6.0+cu124) fails here with
"no kernel image is available for execution on the device".
"""

import sys

import torch


def main() -> int:
    print(f"torch      {torch.__version__}")
    print(f"built for  CUDA {torch.version.cuda}")
    print(f"arch list  {torch.cuda.get_arch_list()}")

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible")
        return 1

    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    print(f"device     {name}  ({sm})")

    if sm not in torch.cuda.get_arch_list():
        print(f"FAIL: this torch has no {sm} kernels -- reinstall from the cu129 index")
        return 1

    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        b = (a @ a).float()
        torch.cuda.synchronize()
        loss = b.mean()
        print(f"bf16 matmul OK   mean={loss.item():+.4f}")

        x = torch.randn(512, 512, device="cuda", requires_grad=True)
        (x * x).sum().backward()
        torch.cuda.synchronize()
        print(f"autograd OK      grad_norm={x.grad.norm().item():.2f}")
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return 1

    free, total = torch.cuda.mem_get_info()
    print(f"vram       {free / 2**30:.1f} GiB free / {total / 2**30:.1f} GiB total")
    print("\nPASS -- Phase 0 gate cleared")
    return 0


if __name__ == "__main__":
    sys.exit(main())

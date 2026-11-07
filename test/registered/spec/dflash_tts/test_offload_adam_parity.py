"""Bitwise-parity test for the offloaded Adam path in dflash_lora_tts.

With offload, a request's optimizer state lives in pinned CPU memory between
updates. For an update it is copied to GPU workspaces, stepped with torch's
functional adam (with the default implementation selector, foreach=None) and
copied back. This test asserts that the result is bitwise identical to a
resident torch.optim.Adam (parameters, both moments and the step counter)
over several steps. It runs on CPU and, when available, on CUDA, where a
default-constructed Adam uses the foreach multi-tensor implementation.

If a torch upgrade changes the sequence of operations in the functional or
the resident optimizer, this test fails, instead of the two paths drifting
apart unnoticed.

Run: python test/registered/spec/dflash_tts/test_offload_adam_parity.py
"""

import torch
from torch.optim.adam import adam as functional_adam

LR = 1e-4
STEPS = 5
N = 4096


def run(device: str) -> None:
    torch.manual_seed(0)
    p_ref = torch.nn.Parameter(torch.randn(N, device=device))
    opt = torch.optim.Adam([p_ref], lr=LR)  # resident reference (defaults)

    # Offload-side state: "CPU" master + moment blocks (pinned when the
    # device is cuda, mirroring the real path).
    pin = device == "cuda"
    master = p_ref.detach().to("cpu").pin_memory() if pin else p_ref.detach().clone()
    m_cpu = torch.empty(N, pin_memory=pin)
    v_cpu = torch.empty(N, pin_memory=pin)
    moments_valid = False
    step_ct = 0

    for it in range(STEPS):
        g = torch.randn(N, device=device)
        p_ref.grad = g.clone()
        opt.step()

        # materialize -> functional adam on workspaces -> writeback
        leaf = torch.empty(N, device=device)
        leaf.copy_(master, non_blocking=pin)
        leaf.requires_grad_(True)
        leaf.grad = g.clone()
        m = torch.empty_like(leaf)
        v = torch.empty_like(leaf)
        if moments_valid:
            m.copy_(m_cpu, non_blocking=pin)
            v.copy_(v_cpu, non_blocking=pin)
        else:
            m.zero_()
            v.zero_()
        with torch.no_grad():
            functional_adam(
                [leaf],
                [leaf.grad],
                [m],
                [v],
                [],
                [torch.tensor(float(step_ct))],
                # None = the same implementation selector as a
                # default-constructed resident Adam (foreach on CUDA,
                # single-tensor on CPU). foreach=False diverges bitwise on
                # CUDA at step 0 (lerp_ vs mul_+addcmul_ forms).
                foreach=None,
                capturable=False,
                differentiable=False,
                fused=None,
                grad_scale=None,
                found_inf=None,
                has_complex=False,
                decoupled_weight_decay=False,
                amsgrad=False,
                beta1=0.9,
                beta2=0.999,
                lr=LR,
                weight_decay=0.0,
                eps=1e-8,
                maximize=False,
            )
        master.copy_(leaf.detach())
        m_cpu.copy_(m)
        v_cpu.copy_(v)
        moments_valid = True
        step_ct += 1

        assert torch.equal(
            p_ref.detach().cpu(), master.cpu()
        ), f"[{device}] param diverged at step {it}"

    sd = opt.state_dict()["state"][0]
    assert torch.equal(sd["exp_avg"].cpu(), m_cpu.cpu()), f"[{device}] exp_avg"
    assert torch.equal(
        sd["exp_avg_sq"].cpu(), v_cpu.cpu()
    ), f"[{device}] exp_avg_sq"
    assert float(sd["step"]) == step_ct, f"[{device}] step count"
    print(
        f"[{device}] offloaded functional Adam BITWISE-identical to resident "
        f"torch.optim.Adam over {STEPS} steps (params + moments + step)"
    )


if __name__ == "__main__":
    run("cpu")
    if torch.cuda.is_available():
        run("cuda")
    else:
        print("[cuda] skipped (no device)")

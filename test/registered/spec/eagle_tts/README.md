# EAGLE-3 Test-Time Speculation (TTS) tests

Tests for `python/sglang/srt/speculative/eagle_tts.py` and
`eagle_lora_tts.py`. None of them needs the target model or a running server.
Six run on toy modules; three load a real EAGLE-3 draft checkpoint from the
Hugging Face cache.

| test | what it checks | GPU | checkpoint |
|---|---|---|---|
| `test_eagle_tts.py` | `EagleTTSState` on a toy draft: update stride, a synchronous update changes the weights, fp32 master weights, async and sync updates give identical weights, `hot_token_id` vocabulary mapping, logits without grad are skipped, `reset()` restores the weights, mismatched row counts raise, the KL gate, and `tts_loss_rows` (the draft/target row pairing, with hand-written expected rows and a synthetic end-to-end agreement check) | optional | no |
| `test_tts_grad_mode.py` | `tts_grad_mode` restores autograd through a forward that inherits `@torch.no_grad()` and runs under the scheduler's `DynamicGradMode`; all overrides are removed on exit; the differentiable all-reduce and all-gather have the expected backward | no | no |
| `test_tts_target_safety.py` | "full" mode: tensors the draft shares with the target are replaced by private copies, no trainable tensor shares storage with the target, and the target is bit-identical after 25 updates and after `reset()` | no | no |
| `test_pool_adam.py` | `PoolAdam`: only the slots that fired are stepped, slots are independent, `reset_slots` clears moments and step count, duplicate slots raise, a step without gradients or with a zero gradient leaves the slot untouched. Also the per-request loss reduction and the per-slot gradient clip | no | no |
| `test_bf16_optimizer.py` | The bfloat16 optimizer dtype on the per-request LoRA, shared LoRA and `full` paths: a step equals an fp32 Adam step on the widened state, rounded; the state is bf16 by dtype and byte count; updates below the bf16 spacing are lost and accumulate with float32; slot and tensor isolation; reset; tensor-parallel completion and clip equal those of float32; non-finite skip; the second moment does not decay; async equals sync | optional | no |
| `test_pool_gather.py` | The per-request path of `LoRAAdapter.delta` and `_RowSelector`: each row uses only its own slot, gradients do not cross slots, an in-place remapping of the static buffer takes effect, behavior before any mapping is published (CUDA graph initialization), a stale mapping is rejected, the `lm_head` selector with several rows per request, and the `pool_size=1` plain path | visible device, CPU tensors | no |
| `test_tts_param_coverage.py` | "full" mode on a real EAGLE-3 draft: the optimizer and the snapshot cover every parameter, every parameter receives a finite non-zero gradient, every parameter changes after one update, and `reset()` restores all of them | visible device; tensors can stay on CPU | yes |
| `test_lora_adapters.py` | "lora" mode on a real draft: adapters are the identity at init, the frozen base stays bit-identical through updates (apart from the two post-attention norms trained in unpooled mode), every adapter tensor is updated, `reset()` restores the pretrained function without a snapshot, the fused `qkv_proj` is wrapped once across q/k/v, `uninstall_adapters` removes the wrappers, and the attention-patch gating reports correctly | yes | yes |
| `test_lora_grad_tp.py` | "lora" mode under tensor parallelism: after two updates, gradients and values of replicated tensors are identical on every rank. Optional negative control and comparison of gradient magnitudes against a tp=1 run | one per rank | yes |

## Running

Each script is run directly and exits non-zero on failure. Commands are given
from the repository root, with `python` being the interpreter of the
environment sglang is installed in.

```bash
python test/registered/spec/eagle_tts/test_eagle_tts.py
python test/registered/spec/eagle_tts/test_bf16_optimizer.py
python test/registered/spec/eagle_tts/test_tts_grad_mode.py
python test/registered/spec/eagle_tts/test_tts_target_safety.py
python test/registered/spec/eagle_tts/test_pool_gather.py
```

`test_pool_adam.py` is pytest-style and has no `__main__` block:

```bash
python -m pytest test/registered/spec/eagle_tts/test_pool_adam.py -q
```

### Device selection: `TTS_TEST_DEV`

`test_eagle_tts.py`, `test_tts_param_coverage.py` and `test_lora_adapters.py`
read `TTS_TEST_DEV` to choose the device for their tensors (default `cuda`).
`test_bf16_optimizer.py` reads it too, and defaults to CUDA when a device is
available. The other tests do not read it.

- `test_tts_grad_mode.py`, `test_tts_target_safety.py` and `test_pool_adam.py`
  always run on CPU and need no GPU.

- `test_eagle_tts.py` runs fully on CPU with `TTS_TEST_DEV=cpu`; the
  async-vs-sync comparison is then skipped, because it needs a CUDA stream:

  ```bash
  CUDA_VISIBLE_DEVICES="" TTS_TEST_DEV=cpu \
    python test/registered/spec/eagle_tts/test_eagle_tts.py
  ```

- `test_pool_gather.py` keeps its tensors on CPU but needs a visible CUDA
  device: under grad, `_RowSelector.rows_for` calls
  `torch.cuda.is_current_stream_capturing()`, which fails when no device is
  available.

- `test_tts_param_coverage.py` accepts `TTS_TEST_DEV=cpu`, but a CUDA device
  must still be visible because `ServerArgs` requires an accelerator:

  ```bash
  CUDA_VISIBLE_DEVICES=0 TTS_TEST_DEV=cpu HF_HUB_OFFLINE=1 \
    python test/registered/spec/eagle_tts/test_tts_param_coverage.py
  ```

- `test_lora_adapters.py` needs its tensors on the GPU. Part of its surrogate
  forward runs outside `tts_grad_mode()`, where sglang's fused RMSNorm
  dispatches to its CUDA kernel regardless of the tensor device, so
  `TTS_TEST_DEV=cpu` does not work for it.

### Tests that load a checkpoint

These read `AngelSlim/Qwen3-8B_eagle3` from `~/.cache/huggingface/hub`.

`test_lora_adapters.py` runs three target sets (the default set,
`o,gate_up,down,lm_head`, and `q,k,v,o`).

```bash
HF_HUB_OFFLINE=1 python test/registered/spec/eagle_tts/test_lora_adapters.py
```

Its surrogate forward runs inside `tts_grad_mode()` whenever gradients are
needed. Outside it the fused RMSNorm / SiluAndMul kernels have no autograd
registration and end the tape, so only adapters downstream of the last fused
op would receive a gradient.

`test_lora_grad_tp.py` runs one process per TP rank under
`torchrun`, each on its own GPU:

```bash
# Rank consistency at tp=2
HF_HUB_OFFLINE=1 torchrun --nproc_per_node=2 \
  test/registered/spec/eagle_tts/test_lora_grad_tp.py

# Negative control: disables the TP gradient corrections and passes only if
# the ranks are observed to diverge
HF_HUB_OFFLINE=1 torchrun --nproc_per_node=2 \
  test/registered/spec/eagle_tts/test_lora_grad_tp.py --negative-control

# Gradient magnitudes against a tp=1 run
HF_HUB_OFFLINE=1 torchrun --nproc_per_node=1 \
  test/registered/spec/eagle_tts/test_lora_grad_tp.py --save-ref /tmp/tp1_grads.pt
HF_HUB_OFFLINE=1 torchrun --nproc_per_node=2 \
  test/registered/spec/eagle_tts/test_lora_grad_tp.py --compare-ref /tmp/tp1_grads.pt
```

## Output

The scripts print one `PASS` / `FAIL` line per check and a final summary
(`ALL PASS` or `all checks passed`, otherwise the list of failed checks).

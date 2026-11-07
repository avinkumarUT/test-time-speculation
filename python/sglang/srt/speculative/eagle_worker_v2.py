import contextlib
import logging
import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.accept_log import make_accept_log
from sglang.srt.speculative.eagle_tts import (
    EagleTTSConfig,
    EagleTTSState,
    privatize_target_aliases,
    tts_grad_mode,
    tts_loss_rows,
)
from sglang.srt.speculative.eagle_info_v2 import (
    assign_extend_cache_locs,
    fill_accepted_out_cache_loc,
    fill_bonus_tokens,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,
    generate_token_bitmask,
    load_token_map,
    maybe_detect_nan,
    maybe_detect_oob,
    select_top_k_tokens,
)
from sglang.srt.utils.common import (
    MultiprocessingSerializer,
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    get_bool_env_var,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    log_info_on_rank0,
    next_power_of_2,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_npu = is_npu()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_hip = is_hip()

logger = logging.getLogger(__name__)


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class EagleDraftWorker(BaseDraftWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank

        # Args for easy access
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Do not capture cuda graph in `TpModelWorker` init,
        # will capture later with init_cuda_graphs()
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Init draft worker
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            # Init draft worker
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            eagle_config = getattr(
                self.draft_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        # TTS state. Declared before CUDA graph capture, because capture calls
        # `draft_forward`, which reads these attributes.
        self._tts_state = None
        self._tts_patch_attention = True
        self._tts_target_logits = None
        self._tts_draft_logits = None

        self.init_token_map()
        self.init_lm_head()

        # Both of the following must run before the CUDA graph capture below.
        #
        # full mode: `init_lm_head` binds the draft's embed_tokens (and, for
        # drafts with load_lm_head_from_target, its lm_head) to the target's
        # tensors. Privatizing rebinds the attribute to a new address, so a
        # graph captured first would keep reading the target's tensor and the
        # updated copy would not be used for drafting.
        #
        # lora mode: installing an adapter rebinds `module.forward`, and the
        # delta has to be inside the captured graph. The rollout replays that
        # graph on every non-firing round, and the optimizer writes A/B in
        # place, so the replay reads the new adapter values without recapture.
        # A graph captured before installation would not contain the adapters.
        self._lora_adapters = None
        _tts_cfg_early = EagleTTSConfig.from_server_args(server_args)
        if _tts_cfg_early.enable:
            if _tts_cfg_early.per_request and not _tts_cfg_early.is_lora:
                raise ValueError(
                    "--speculative-eagle-tts-per-request requires "
                    "--speculative-eagle-tts-mode lora: a per-request copy of "
                    "the full draft does not fit for any useful batch size."
                )
            if _tts_cfg_early.is_lora:
                from sglang.srt.speculative.eagle_lora_tts import (
                    DEFAULT_LORA_TARGETS,
                    EagleLoRAAdapterSet,
                    parse_lora_targets,
                )

                # One adapter slot per concurrent request. `req_pool_idx` is a
                # dense index into `req_to_token_pool`, so it doubles as the
                # slot id and no side table is needed on the decode path. The
                # adapter pool must therefore cover every index the request
                # table can hand out.
                pool_size = 1
                if _tts_cfg_early.per_request:
                    # Size the pool from `_alloc_size`, the number of rows the
                    # request table allocates. A plain ReqToTokenPool has
                    # size + 1 rows: req_pool_idx runs 1..size inclusive and
                    # row 0 is a padding row. CUDA-graph padded batches default
                    # req_pool_indices to 0, so their dummy adapter reads land
                    # on that unused row. The PD-disaggregation
                    # DecodeReqToTokenPool hands out size + pre_alloc_size + 1
                    # indices, so `.size` alone is not a valid bound.
                    req_pool = self.target_worker.model_runner.req_to_token_pool
                    pool_size = int(
                        getattr(req_pool, "_alloc_size", req_pool.size + 1)
                    )
                    if server_args.max_running_requests is None:
                        # If max_running_requests is still unset here, the
                        # request table was sized from KV capacity
                        # (`_resolve_max_num_reqs`), which can be in the
                        # thousands. That is sized for per-request bookkeeping,
                        # not for a slot of adapter and optimizer state each,
                        # and would allocate far more memory than intended.
                        # Require an explicit value instead.
                        raise ValueError(
                            "--speculative-eagle-tts-per-request requires an "
                            "explicit --max-running-requests: the adapter pool "
                            f"is sized to it ({pool_size} slots here, inferred "
                            "from KV capacity), and each slot holds adapters "
                            "plus optimizer state. Set it to the concurrency "
                            "you intend to serve."
                        )
                self._lora_adapters = EagleLoRAAdapterSet(
                    draft_model=self.draft_runner.model,
                    targets=parse_lora_targets(
                        _tts_cfg_early.lora_targets or DEFAULT_LORA_TARGETS
                    ),
                    rank=_tts_cfg_early.lora_rank,
                    alpha=_tts_cfg_early.lora_alpha,
                    tp_size=server_args.tp_size,
                    tp_rank=tp_rank,
                    pool_size=pool_size,
                    # Widest draft forward the row buffer must hold: the
                    # draft-extend writes num_draft_tokens rows per request,
                    # and a prefill writes a whole chunk of prompt at once.
                    # `chunked_prefill_size` is -1 when chunking is disabled
                    # (and `-1 or 0` evaluates to -1), so it is clamped at 0
                    # and the context length, the widest a single prefill can
                    # be, is used instead.
                    max_rows=max(
                        pool_size * (server_args.speculative_num_draft_tokens or 1),
                        max(0, int(server_args.chunked_prefill_size or 0))
                        or int(self.draft_runner.model_config.context_len),
                    ),
                    # Widest lm_head forward: num_tokens_for_logprob rows per
                    # request in the decode extend.
                    head_rows_per_req=(
                        server_args.speculative_num_draft_tokens or 1
                    ),
                    fp32_master=_tts_cfg_early.fp32_master,
                )
            else:
                privatize_target_aliases(
                    self.draft_runner.model, self.target_worker.model_runner.model
                )

        # Init attention backend and cuda graphs
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.init_attention_backend()
            self.init_cuda_graphs()

        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

        # Test-time speculation (TTS): online update of the draft model
        # (optional).
        tts_cfg = EagleTTSConfig.from_server_args(server_args)
        if tts_cfg.enable:
            if get_bool_env_var("SGLANG_ENABLE_TORCH_INFERENCE_MODE", "false"):
                raise ValueError(
                    "EAGLE3 TTS needs gradients through the draft forward, which "
                    "SGLANG_ENABLE_TORCH_INFERENCE_MODE=1 makes impossible."
                )
            # `init_lm_head` binds the draft's embed_tokens (and, for drafts
            # with load_lm_head_from_target, its lm_head) to the target's own
            # tensors, which must not be modified. The target is passed so that
            # `EagleTTSState` can verify that no tensor it updates shares
            # storage with it.
            self._tts_state = EagleTTSState(
                draft_model=self.draft_runner.model,
                config=tts_cfg,
                device=self.device,
                tp_size=server_args.tp_size,
                target_model=self.target_worker.model_runner.model,
                adapters=self._lora_adapters,
            )
            # Patch attention for grad only when some target's gradient has to
            # cross it. With a single-layer draft, an adapter set confined to
            # o / gate_up / down / lm_head needs no path through attention, so
            # the fused kernel stays in place and the full-prefix K/V gather
            # and SDPA rebuild that `_sdpa_attention` performs once per rollout
            # step are skipped.
            self._tts_patch_attention = (
                self._lora_adapters is None
                or self._lora_adapters.needs_differentiable_attention()
            )

    def init_token_map(self):
        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if self.server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif self.server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(self.server_args.speculative_token_map)
            self.server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_runner.model.set_embed_and_head(embed, head)

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners

        self.has_prefill_wrapper_verify = False
        self.draft_extend_attn_backend = None

        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        if self.server_args.model_impl == "mindspore":
            return

        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB.",
            )

        Device2ExtendCudaGraphRunner = {
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        supports_hip_aiter_draft_extend_graph = False
        if _is_hip:
            # Keep import local so non-HIP environments do not require aiter.
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )

            supports_hip_aiter_draft_extend_graph = isinstance(
                self.draft_attn_backend, AiterMultiStepDraftBackend
            )

        supports_cuda_draft_extend_graph = (_is_cuda or _is_musa) and (
            isinstance(self.draft_extend_attn_backend, TritonAttnBackend)
            or isinstance(self.draft_extend_attn_backend, TRTLLMMLABackend)
        )
        # Capture extend
        # TODO: support draft extend cuda graph for more attention backends
        if self.draft_extend_attn_backend and (
            _is_npu
            or supports_cuda_draft_extend_graph
            or supports_hip_aiter_draft_extend_graph
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB.",
            )

    def _bind_adapter_rows(self, forward_batch):
        """Publish row -> adapter-slot for a draft forward; return the slots.

        Every forward that runs the draft model has to do this, not only the
        rollout. The adapter wrappers read the mapping from a static buffer
        and cannot see the ForwardBatch, so a forward that skips the publish
        reuses the previous forward's mapping and applies one request's adapter
        to another request's rows. The row-count check in `rows_for` catches
        the case where the widths differ but not the case where they coincide,
        so each call site binds explicitly.

        Extend forwards write several tokens per request, so the per-row
        mapping is `req_pool_indices` repeated by each request's token count.
        `output_size` is passed so `repeat_interleave` does not sync the device
        to the host to determine its output length.
        """
        if self._lora_adapters is None or self._lora_adapters.pool_size <= 1:
            # No pool (full mode, or LoRA with a single shared adapter): every
            # row uses the same weights, so there is no mapping to publish.
            # Return before the row-count check below, which applies only when
            # rows must be attributed to requests: some draft forwards have
            # more rows than requests and carry no extend_seq_lens.
            return None
        idx = forward_batch.req_pool_indices
        n_rows = forward_batch.input_ids.shape[0]
        row_slots = None
        if n_rows != idx.numel():
            lens = forward_batch.extend_seq_lens
            if lens is None:
                raise RuntimeError(
                    f"EAGLE3 LoRA-TTS: draft forward has {n_rows} rows for "
                    f"{idx.numel()} request(s) but carries no extend_seq_lens, "
                    "so rows cannot be attributed to requests."
                )
            row_slots = torch.repeat_interleave(idx, lens, output_size=n_rows)
        # lm_head sees the rows left after LogitsProcessor's position
        # selection, not the token rows: a contiguous block of
        # `num_tokens_for_logprob_per_req` rows per request. That is 1 at
        # prefill and num_steps + 1 in the decode extend; it is read from the
        # forward's own spec_info.
        head_slots = None
        per_req = int(
            getattr(
                forward_batch.spec_info, "num_tokens_for_logprob_per_req", 1
            )
            or 1
        )
        if per_req > 1:
            head_slots = idx.repeat_interleave(per_req)
        return self._lora_adapters.bind_rows(idx, row_slots, head_slots)

    def draft(self, model_worker_batch: ModelWorkerBatch):
        draft_input: EagleDraftInput = model_worker_batch.spec_info
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            model_worker_batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # An asynchronous TTS step writes the draft weights from a side stream,
        # so every draft forward must wait for it first. The wait is placed
        # here rather than in `draft_forward` so that it also covers the
        # CUDA-graph replay below, which runs on the non-firing rounds. No-op
        # when nothing is pending.
        if self._tts_state is not None:
            self._tts_state.wait_for_pending()

        # Publish this batch's row -> adapter-slot mapping before any draft
        # forward. The wrappers only see `input_`, not the ForwardBatch, so the
        # mapping travels through a static buffer that the captured graph reads
        # at replay time.
        if self._lora_adapters is not None:
            # Bind req_pool_indices directly rather than through
            # `_bind_adapter_rows`: the rollout's per-step inputs are built
            # inside draft_forward from spec_info (select_top_k_tokens), so the
            # adapters see exactly one row per request at topk=1, whereas
            # `forward_batch.input_ids` here is a different tensor whose width
            # is unrelated to the batch size.
            self._tts_state.active_slots = self._lora_adapters.bind_rows(
                forward_batch.req_pool_indices
            )

        # TTS fits the draft's proposal distributions, which are produced in
        # the rollout below, to the target's verification distributions. A
        # replayed CUDA graph builds no autograd tape, so firing rounds run the
        # rollout eagerly.
        if self._tts_state is not None and self._tts_state.should_fire():
            can_cuda_graph = False

        # Run draft
        if can_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch,
            )
        else:
            if (
                not forward_batch.forward_mode.is_idle()
                and self.speculative_num_steps > 1
            ):
                # Skip attention backend init for 1-step draft,
                # `draft_forward` only does sample in this case.
                self.draft_attn_backend.init_forward_metadata(forward_batch)
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        if model_worker_batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )

        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            model_worker_batch.seq_lens,
            model_worker_batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # The TTS loss uses the per-step logits produced in this rollout. They
        # are the draft's proposal distributions: their argmax becomes the
        # draft tokens, so they determine acceptance. The draft-extend pass
        # that runs after verification (`_draft_extend_for_decode`) is
        # teacher-forced, only fills the draft KV cache and produces no draft
        # token, so its logits are not used for the loss.
        collect_tts = self._tts_state is not None and self._tts_state.should_fire()
        self._tts_draft_logits = [] if collect_tts else None
        saved_logits_buffer = forward_batch.next_token_logits_buffer
        saved_graph_runner = self.draft_runner.graph_runner
        saved_piecewise_runner = self.draft_runner.piecewise_cuda_graph_runner
        try:
            if collect_tts:
                # LogitsProcessor would copy the logits into this preallocated
                # buffer and return it, which has no grad_fn.
                forward_batch.next_token_logits_buffer = None
                # `can_cuda_graph=False` in draft() only disables EAGLE's own
                # draft graph runner. The draft ModelRunner has graph runners
                # of its own: `_forward_raw` replays `graph_runner` (and, when
                # enabled, `piecewise_cuda_graph_runner`) for eligible batches,
                # which executes a captured graph instead of the Python forward
                # and builds no autograd tape. Disable both for the rollout;
                # the `finally` below restores them.
                self.draft_runner.graph_runner = None
                self.draft_runner.piecewise_cuda_graph_runner = None

            # Forward multiple steps
            scores = None
            for i in range(self.speculative_num_steps):
                input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                    i, topk_p, topk_index, hidden_states, scores, self.topk
                )
                score_list.append(tree_info[0])
                token_list.append(tree_info[1])
                parents_list.append(tree_info[2])

                # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
                if i == self.speculative_num_steps - 1:
                    break

                # Set inputs
                forward_batch.input_ids = input_ids
                forward_batch.out_cache_loc = out_cache_loc[i]
                forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
                spec_info.hidden_states = hidden_states

                # Run forward
                with (
                    # A fresh instance per step: `tts_grad_mode` is a generator
                    # context manager and cannot be re-entered.
                    tts_grad_mode(
                        self.draft_runner.model,
                        patch_attention=self._tts_patch_attention,
                    )
                    if collect_tts
                    else contextlib.nullcontext()
                ), (torch.enable_grad() if collect_tts else contextlib.nullcontext()):
                    logits_output = self.draft_runner.forward(
                        forward_batch, skip_attn_backend_init=True
                    ).logits_output
                if collect_tts:
                    # Rollout step i produces the distribution over tree node i+2.
                    self._tts_draft_logits.append(logits_output.next_token_logits)
                    logits_output.next_token_logits = (
                        logits_output.next_token_logits.detach()
                    )
                maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
                probs = torch.softmax(logits_output.next_token_logits, dim=-1)
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
                maybe_detect_oob(
                    topk_index,
                    0,
                    logits_output.next_token_logits.shape[-1],
                    f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
                )
                if self.hot_token_id is not None:
                    topk_index = self.hot_token_id[topk_index]
                hidden_states = logits_output.hidden_states
                forward_batch.positions.add_(1)

        finally:
            # Restored in a `finally` so that an exception in the rollout (a
            # NaN check, an attention-patch mismatch) cannot leave both
            # ModelRunner graph runners set to None, which would make every
            # later draft forward run eagerly for the life of the process.
            forward_batch.next_token_logits_buffer = saved_logits_buffer
            self.draft_runner.graph_runner = saved_graph_runner
            self.draft_runner.piecewise_cuda_graph_runner = saved_piecewise_runner

        # Organize the results
        score_list = torch.cat(score_list, dim=1).flatten(
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        maybe_detect_oob(
            top_scores_index,
            0,
            ss_token_list.shape[1],
            "draft_forward: top_scores_index OOB for gather on ss_token_list",
        )
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def draft_extend(self):
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # Construct input_ids
        if not batch.forward_mode.is_idle():
            pt = 0
            for i, extend_len in enumerate(batch.extend_seq_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                batch.input_ids[pt : pt + extend_len] = torch.cat(
                    (input_ids[1:], next_token_ids[i].reshape(1))
                )
                pt += extend_len

        # Construct spec_info
        next_draft_input = EagleDraftInput(
            hidden_states=target_hidden_states,
            bonus_tokens=next_token_ids,
            new_seq_lens=batch.seq_lens,
            # draft mode is same with decode mode, only 1 token per req
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        batch.spec_info = next_draft_input

        # Run forward
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
        forward_batch.return_logprob = False
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds
        # Prefill runs the adapted draft too, so it needs its own row mapping;
        # without one it would inherit the last decode round's.
        self._bind_adapter_rows(forward_batch)
        logits_output = self.draft_runner.forward(forward_batch).logits_output
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")

        # Update spec_info for the next draft step
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        next_draft_input.topk_p, next_draft_input.topk_index = fast_topk(
            probs, self.topk, dim=-1
        )
        next_draft_input.hidden_states = logits_output.hidden_states
        return next_draft_input

    def _draft_extend_for_decode(
        self, batch: ModelWorkerBatch, batch_result: GenerationBatchResult
    ):
        # Batch 2: Draft extend
        draft_input = EagleDraftInput(
            hidden_states=batch_result.logits_output.hidden_states,
            num_tokens_per_req=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_req=self.speculative_num_steps + 1,
        )
        select_index = (
            torch.arange(len(batch.seq_lens), device=self.device)
            * self.speculative_num_draft_tokens
            + batch_result.accept_lens
            - 1
        )

        # Prepare for draft extend in a separate stream
        with self.plan_stream_ctx:
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                batch,
                batch_result.next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner,
                self.cuda_graph_runner_for_draft_extend,
            )

        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        if forward_batch.spec_info.num_correct_drafts is None:
            # `batch_result.accept_lens` already includes the bonus token, so use it
            # directly for `num_accept_tokens` and subtract 1 for `num_correct_drafts`.
            forward_batch.spec_info.num_correct_drafts = batch_result.accept_lens - 1
            forward_batch.spec_info.num_accept_tokens = batch_result.accept_lens

        # Run draft extend batch in the main compute stream
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        # The draft-extend pass only fills the draft KV cache and runs
        # unmodified; the TTS loss uses the rollout logits (see
        # draft_forward). `train_tts` gates the optimizer step, which needs
        # the target's verification logits captured earlier this round.
        train_tts = (
            self._tts_state is not None
            and self._tts_state.should_fire()
            and self._tts_target_logits is not None
        )

        # This forward is wider than the rollout (several accepted tokens per
        # request), so it needs its own per-row mapping rather than the one
        # `draft()` published for the one-row-per-request rollout.
        self._bind_adapter_rows(forward_batch)

        if can_cuda_graph:
            draft_logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                forward_batch
            )
        else:
            draft_logits_output = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output

        maybe_detect_nan(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_cuda_graph})",
        )

        if train_tts:
            # Pair the draft's proposal distributions, captured in the
            # `draft_forward` rollout, with the target's verification rows.
            # Rollout step i produced the distribution for tree node i+2, and
            # target verify row j predicts node j+1, so node i+2 pairs with
            # target row i+1: the same token position, with a full
            # distribution on both sides.
            #
            # Limitation: the rollout steps cover tree nodes 2..n_step+1, so
            # target row 0 is unused and node 1, the first drafted token, is
            # not part of the loss. Node 1's proposal is row `select_index` of
            # the draft-extend pass, so pairing it would require carrying this
            # round's logits over to the next round's target row 0.
            nd = self.speculative_num_draft_tokens
            steps = self._tts_draft_logits or []
            if steps:
                device = steps[0].device
                bs = steps[0].shape[0]
                n_step = len(steps)  # nodes 2 .. n_step+1
                # This must be inside enable_grad: the scheduler event loop
                # runs with grad globally disabled (`DynamicGradMode`), so
                # stack / reshape / index_select on grad-tracked tensors would
                # each return a tensor with grad_fn=None, cutting the tape
                # between the logits and the loss.
                with torch.enable_grad():
                    draft = torch.stack(steps, dim=1)  # [bs, n_step, V]
                    draft = draft.reshape(bs * n_step, draft.shape[-1])
                # With `correct_drafts_only` the rows past the accepted prefix
                # are dropped. Otherwise every drafted position is kept: those
                # rows are valid labels too, since at topk=1 the target's tree
                # mask makes verify row j attend nodes 0..j, i.e. the draft's
                # own proposed prefix, so the row is the target's distribution
                # given the context the draft built.
                keep, draft_rows, target_rows = tts_loss_rows(
                    bs,
                    n_step,
                    nd,
                    # The host copy is needed only when filtering.
                    (
                        batch_result.accept_lens.tolist()
                        if self._tts_state.cfg.correct_drafts_only
                        else None
                    ),
                    self._tts_state.cfg.correct_drafts_only,
                )
                if keep:
                    d_idx = torch.tensor(
                        draft_rows, device=device, dtype=torch.long
                    )
                    t_idx = torch.tensor(
                        target_rows, device=device, dtype=torch.long
                    )
                    # Depth weighting w = decay**depth. Rollout step i is tree
                    # node i+2, i.e. chain depth i+2. Acceptance is a chain, so
                    # a shallow node gates every node behind it. The weights
                    # are normalized in `step`, so the constant +2 offset
                    # cancels.
                    decay = self._tts_state.cfg.depth_decay
                    pos_w = None
                    if 0 < decay < 1.0:
                        pos_w = torch.tensor(
                            [decay ** (i + 2) for _, i in keep],
                            device=device,
                            dtype=torch.float32,
                        )
                    # Which request each kept row belongs to. Built from the
                    # same `keep` list as d_idx/t_idx/pos_w, so the grouping is
                    # aligned with those by construction. Pooled mode only:
                    # without a pool the adapter is shared and the loss is a
                    # single mean over the batch.
                    row_groups = None
                    if self._tts_state.is_pooled:
                        row_groups = torch.tensor(
                            [b for b, _ in keep], device=device, dtype=torch.long
                        )
                    with torch.enable_grad():
                        self._tts_state.step(
                            draft_logits=draft.index_select(0, d_idx),
                            target_logits=self._tts_target_logits.index_select(
                                0, t_idx
                            ),
                            hot_token_id=self.hot_token_id,
                            pos_weights=pos_w,
                            row_groups=row_groups,
                            num_groups=bs if row_groups is not None else 0,
                        )
                else:
                    self._tts_state.note_no_aligned_rows()
            else:
                self._tts_state.note_no_aligned_rows()
            # Detach so the autograd tape does not survive into the next round.
            draft_logits_output.next_token_logits = (
                draft_logits_output.next_token_logits.detach()
            )
            if draft_logits_output.hidden_states is not None:
                draft_logits_output.hidden_states = (
                    draft_logits_output.hidden_states.detach()
                )
        # Drop the captured rollout logits outside the `train_tts` branch. A
        # round can fire (so `draft_forward` captured a rollout tape) and still
        # not train, when `verify` left `_tts_target_logits` as None, e.g. for
        # an idle batch with no `next_token_logits`. Clearing only under
        # `train_tts` would keep that tape, and every activation it
        # references, alive until the next `draft_forward` overwrites it.
        self._tts_draft_logits = None
        self._tts_target_logits = None
        if self._tts_state is not None:
            self._tts_state.advance()

        # Reorganize the spec info for the next batch
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        if draft_logits_output.hidden_states is not None:
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]
        probs = torch.softmax(draft_logits_output.next_token_logits, dim=-1)
        ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
        ret_hidden_states = draft_logits_output.hidden_states

        # Construct the return values
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            ret_topk_p,
            ret_topk_index,
            ret_hidden_states,
        )


class EAGLEWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.tp_rank = tp_rank
        self._accept_log = make_accept_log(server_args, tp_rank)
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        self._draft_worker = EagleDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # Adaptive speculative
        self.adaptive_controller: Optional[AdaptiveController] = None
        if server_args.speculative_adaptive:
            self.adaptive_controller = AdaptiveController(
                self, config_path=server_args.speculative_adaptive_config
            )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

        # Build adaptive runtime states (must be after draft worker is fully initialized)
        if self.adaptive_controller is not None:
            with (
                self._draft_worker.draft_tp_context(
                    self._draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.adaptive_controller.register(
                    SpecRuntimeState(
                        speculative_num_steps=self.speculative_num_steps,
                        speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                        draft_attn_backend=self._draft_worker.draft_attn_backend,
                        cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                        target_attn_backend=self._target_worker.model_runner.attn_backend,
                        target_graph_runner=self._target_worker.model_runner.graph_runner,
                        draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                        cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
                    )
                )
                self.adaptive_controller.init_states()

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker, which are cleared in scheduler
        pass

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # A new prefill starts a new sequence: discard the adaptation
            # accumulated for the previous request. Pooled mode clears only
            # the slots this prefill is claiming. `req_pool_idx` values are
            # recycled, so the incoming request would otherwise inherit the
            # adaptation of the slot's previous holder; the requests still
            # decoding keep theirs.
            state = self.draft_worker._tts_state
            if state is not None:
                if state.is_pooled:
                    state.reset(model_worker_batch.req_pool_indices)
                else:
                    state.reset()
                self.draft_worker._tts_target_logits = None
            # Target prefill
            target_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            model_worker_batch.capture_hidden_mode = target_capture_mode
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            # Draft prefill
            draft_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.LAST
            )
            model_worker_batch.capture_hidden_mode = draft_capture_mode
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        model_worker_batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                    )
                )
                return batch_output
        else:
            if model_worker_batch.spec_info is None:
                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.speculative_algorithm.is_standalone()
                    else CaptureHiddenMode.LAST
                )
                model_worker_batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=EagleDraftInput.hidden_size_for(self.draft_worker),
                    dtype=EagleDraftInput.dtype_for(self.draft_worker),
                    topk=self.topk,
                    capture_hidden_mode=capture_mode,
                )
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                verify_input: EagleVerifyInput = self.draft_worker.draft(
                    model_worker_batch
                )
            assert verify_input.is_verify_input()
            # Record a CUDA event after draft() GPU work is dispatched.
            # This event will be waited on by plan_stream in verify()
            # to ensure draft CUDA graph kernels finish before plan_stream
            # begins metadata preparation.
            if self.plan_stream:
                self._draft_done_event = torch.get_device_module(self.device).Event()
                self._draft_done_event.record()
            model_worker_batch.spec_info = verify_input
            batch_output = self.verify(model_worker_batch)
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.draft_worker._draft_extend_for_decode(
                    model_worker_batch, batch_output
                )

            return batch_output

    def on_verify_complete_cpu(self, num_correct_drafts_per_req: list[int]) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(num_correct_drafts_per_req)

    # -- Adaptive speculative decoding protocol --

    def build_adaptive_runtime_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ) -> SpecRuntimeState:
        """Build a SpecRuntimeState for the given step configuration."""
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)

        with self._override_worker_state(
            speculative_num_steps, speculative_num_draft_tokens
        ):
            self._draft_worker.init_attention_backend()
            self._draft_worker.init_cuda_graphs()

            # Build target attention backend and CUDA graph runner
            target_model_runner = self._target_worker.model_runner
            backup_init = target_model_runner.init_new_workspace
            try:
                target_attn_backend = target_model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                target_model_runner.init_new_workspace = backup_init

            target_graph_runner = None
            if not self.server_args.disable_cuda_graph:
                target_graph_runner = CudaGraphRunner(
                    target_model_runner,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                )

            state = SpecRuntimeState(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                draft_attn_backend=self._draft_worker.draft_attn_backend,
                cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
                draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
            )

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        log_info_on_rank0(
            logger,
            f"Built adaptive runtime state steps={speculative_num_steps}: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB",
        )

        return state

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        """Apply a pre-built runtime state to this worker."""
        if self.speculative_num_steps == state.speculative_num_steps:
            return

        log_info_on_rank0(
            logger,
            "Switch adaptive runtime state: "
            f"steps {self.speculative_num_steps} -> {state.speculative_num_steps}, "
            f"draft_tokens {self.speculative_num_draft_tokens} -> "
            f"{state.speculative_num_draft_tokens}",
        )

        # Top-level
        self.speculative_num_steps = state.speculative_num_steps
        self.speculative_num_draft_tokens = state.speculative_num_draft_tokens

        # Draft side
        dw = self._draft_worker
        dw.speculative_num_steps = state.speculative_num_steps
        dw.speculative_num_draft_tokens = state.speculative_num_draft_tokens
        dw.draft_attn_backend = state.draft_attn_backend
        dw.draft_runner.draft_attn_backend = state.draft_attn_backend
        dw.cuda_graph_runner = state.cuda_graph_runner
        dw.draft_extend_attn_backend = state.draft_extend_attn_backend
        dw.cuda_graph_runner_for_draft_extend = state.cuda_graph_runner_for_draft_extend

        # Target side
        self._target_worker.model_runner.attn_backend = state.target_attn_backend
        self._target_worker.model_runner.graph_runner = state.target_graph_runner

        # Sync server_args
        self.server_args.speculative_num_steps = state.speculative_num_steps
        self.server_args.speculative_num_draft_tokens = (
            state.speculative_num_draft_tokens
        )

    @contextlib.contextmanager
    def _override_worker_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ):
        """Temporarily override server_args and worker attributes for graph capture."""
        sa = self.server_args
        dw = self._draft_worker
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            dw.speculative_num_steps,
            dw.speculative_num_draft_tokens,
            dw.draft_attn_backend,
            dw.draft_extend_attn_backend,
            dw.draft_runner.draft_attn_backend,
            dw.cuda_graph_runner,
            dw.cuda_graph_runner_for_draft_extend,
            sa.speculative_num_steps,
            sa.speculative_num_draft_tokens,
        )

        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        dw.speculative_num_steps = speculative_num_steps
        dw.speculative_num_draft_tokens = speculative_num_draft_tokens
        sa.speculative_num_steps = speculative_num_steps
        sa.speculative_num_draft_tokens = speculative_num_draft_tokens

        try:
            yield
        finally:
            (
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                dw.speculative_num_steps,
                dw.speculative_num_draft_tokens,
                dw.draft_attn_backend,
                dw.draft_extend_attn_backend,
                dw.draft_runner.draft_attn_backend,
                dw.cuda_graph_runner,
                dw.cuda_graph_runner_for_draft_extend,
                sa.speculative_num_steps,
                sa.speculative_num_draft_tokens,
            ) = backup

    def verify(self, batch: ModelWorkerBatch):
        # Since batch.seq_lens is allocated in another stream, we need
        # record_stream() to prevent pytorch gc and reuse the gpu memory
        # while forward_stream is still running.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        # Parse args
        verify_input: EagleVerifyInput = batch.spec_info
        verify_input.num_tokens_per_req = self.speculative_num_steps + 1
        bs = len(batch.seq_lens)

        # Batch 1: Target verify
        # Prepare for target verify in a separate stream
        with self.plan_stream_ctx:
            # Wait for the draft CUDA graph to finish before plan_stream
            # begins its work. Using an event is more targeted than
            # wait_stream(main_stream) — it only waits for draft GPU
            # work, not all queued main_stream operations.
            if self.plan_stream and hasattr(self, "_draft_done_event"):
                self.plan_stream.wait_event(self._draft_done_event)
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
            )

        # Correct some buffers due to the overlap plan
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )
            if (
                _is_npu
                and self._target_worker.model_runner.model_is_mrope
                and batch.spec_info is not None
                and getattr(batch.spec_info, "positions", None) is not None
                and not batch.forward_mode.is_idle()
            ):
                # mrope_position depends on draft output in default stream and is computed in plan stream,
                # causing errors. Compute it here for correct values.
                verify_forward_batch.compute_spec_mrope_positions(
                    self._target_worker.model_runner, batch
                )

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )

        # Prepare grammar data on CPU if needed
        if batch.has_grammar:
            retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrieve_next_token.shape
            ).cpu()

        # Run target verify batch in the main compute stream (GPU compute)
        forward_batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output

        # Copy the target's verification logits before `verify_input.sample`
        # modifies them in place (penalties, grammar mask) and before a later
        # CUDA graph replay overwrites the static buffer they may alias.
        tts_state = self.draft_worker._tts_state
        if (
            tts_state is not None
            and tts_state.should_fire()
            and logits_output.next_token_logits is not None
        ):
            self.draft_worker._tts_target_logits = (
                logits_output.next_token_logits.detach().clone()
            )

        # Generate vocab mask for constrained decoding
        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                # NOTE: otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        # Sample
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
        (
            predict,
            accept_lens,
            accept_index,
        ) = verify_input.sample(batch, logits_output, vocab_mask)
        new_seq_lens = batch.seq_lens + accept_lens
        if self._accept_log is not None:
            self._accept_log.write(batch.reqs, (accept_lens - 1).tolist())

        # Update mamba state for hybrid GDN models after verification
        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
        ):
            self._mamba_verify_update(
                batch, verify_input, accept_lens, accept_index, bs
            )

        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()

        if not batch.forward_mode.is_idle():
            accept_tokens = predict[accept_index]
            bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
            fill_bonus_tokens[(bs,)](
                accept_tokens,
                accept_lens,
                bonus_tokens,
                self.speculative_num_draft_tokens,
            )
        else:
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)

        if batch.return_logprob and not batch.forward_mode.is_idle():
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, self.speculative_num_steps
            )

        # Construct the next draft input
        next_draft_input = EagleDraftInput(
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            routed_experts_output=forward_batch_output.routed_experts_output,
            indexer_topk_output=forward_batch_output.indexer_topk_output,
        )

    def _mamba_verify_update(
        self,
        batch: ModelWorkerBatch,
        verify_input: EagleVerifyInput,
        accept_lens: torch.Tensor,
        accept_index: torch.Tensor,
        bs: int,
    ):
        """Update mamba state for hybrid GDN models after verification."""
        # `accept_lens` already includes the bonus token (drafts + 1 per req).
        if not batch.forward_mode.is_idle() and accept_index.numel() > 0:
            if verify_input.topk != 1:
                raise ValueError("Spec v2 currently only supports topk = 1.")

            accepted_indices_offset = torch.arange(
                0,
                bs * self.speculative_num_draft_tokens,
                step=self.speculative_num_draft_tokens,
                dtype=accept_lens.dtype,
                device=accept_lens.device,
            )
            last_correct_step_indices = accept_lens - 1

            if batch.mamba_track_indices is not None:
                # If after verify, the request's seq_lens has crossed a mamba track interval,
                # we need to update the mamba state for the request at the crossing point.
                seq_lens_pre_verify = batch.seq_lens
                seq_lens_post_verify = batch.seq_lens + accept_lens
                mamba_track_interval = self.server_args.mamba_track_interval
                to_track_mask = (
                    seq_lens_pre_verify // mamba_track_interval
                    != seq_lens_post_verify // mamba_track_interval
                )
                tracking_point = (
                    seq_lens_post_verify // mamba_track_interval * mamba_track_interval
                )
                to_track_ith = torch.clamp(
                    tracking_point - seq_lens_pre_verify - 1, min=0
                ).to(torch.int64)
                req_idx = torch.arange(
                    bs,
                    dtype=torch.int64,
                    device=accept_lens.device,
                )
                candidate_track_steps = (
                    accept_index[req_idx, to_track_ith] - accepted_indices_offset
                )
                mamba_steps_to_track = torch.where(
                    to_track_mask,
                    candidate_track_steps,
                    torch.full_like(candidate_track_steps, -1),
                )
            else:
                mamba_steps_to_track = None

            self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                model=self.target_worker.model_runner.model,
            )

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        num_correct_drafts: torch.Tensor,
    ):
        """
        Move accepted tokens (drafts + bonus) to the target KV cache.

        Args:
            batch: The batch to run.
            accept_index: The index of the accepted tokens (incl. bonus).
            num_correct_drafts: Per-req count of correct drafts (excludes bonus);
                seq_lens is advanced by ``num_correct_drafts + 1`` to cover the bonus slot.
        """
        bs = len(batch.seq_lens)
        size = bs * self.speculative_num_draft_tokens

        tgt_cache_loc = torch.zeros(
            size,
            dtype=torch.int64,
            device=self.device,
        )
        accepted_out_cache_loc = torch.zeros(
            size, dtype=torch.int64, device=self.device
        )
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + num_correct_drafts + 1,
            tgt_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
        fill_accepted_out_cache_loc[(size,)](
            accept_index,
            batch.out_cache_loc,
            accepted_out_cache_loc,
            next_power_of_2(size),
        )
        self.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
            tgt_cache_loc, accepted_out_cache_loc
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self._draft_worker.draft_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        success, message = self._draft_worker.draft_runner.update_weights_from_ipc(
            recv_req
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = self.draft_worker.draft_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message

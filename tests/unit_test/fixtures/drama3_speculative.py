"""Real single-request Drama3 execution for the offline recovery oracle."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)

from sglang_omni.models.fishaudio_drama3.engine_builder import Drama3EngineBuilder
from sglang_omni.models.fishaudio_drama3.sampling import (
    EMPTY_RAS_SLOT,
    RAS_WINDOW_FRAMES,
)
from sglang_omni.models.fishaudio_drama3.sglang_model import Drama3ForCausalLM
from sglang_omni.models.fishaudio_drama3.speculative import Drama3StateSnapshot
from sglang_omni.models.fishaudio_drama3.tower_prefill import PagedTowerAttention
from sglang_omni.scheduling.bootstrap import InfrastructureOptions
from sglang_omni.scheduling.engine_factory import GenerationDefaults


class OfflinePairEngineBuilder(Drama3EngineBuilder):
    def __init__(self) -> None:
        super().__init__(
            context_length=256,
            max_running_requests=2,
            first_frame_cuda_graphs=False,
            triton_sampling=False,
            tower_fused_layers=False,
            enable_async_decode=False,
        )

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            **super().generation_defaults(dtype=dtype),
            "disable_cuda_graph": True,
            "max_prefill_tokens": 256,
            "max_total_tokens": 1024,
        }

    def infra_kwargs(self) -> InfrastructureOptions:
        return {"total_gpu_memory_fraction": 0.3}


@dataclass(frozen=True, kw_only=True)
class OracleStep:
    hidden_states: torch.Tensor
    semantic_logits: torch.Tensor
    frame: torch.Tensor
    ras_window: torch.Tensor


class OfflinePairRequest:
    """Owns a fixed slot reservation; no scheduler, eviction or concurrent request."""

    def __init__(
        self,
        model: Drama3ForCausalLM,
        attention_backend: HybridLinearAttnBackend,
        request: Req,
        cache_slots: torch.Tensor,
    ) -> None:
        self.model = model
        self.attention_backend = attention_backend
        self.request = request
        self.cache_slots = cache_slots
        self.device = cache_slots.device
        request_slot = getattr(request, "req_pool_idx", None)
        if request_slot is None:
            assert request.kv is not None and request.kv.req_pool_idx is not None
            request_slot = request.kv.req_pool_idx
        else:
            pass
        self.request_slot = request_slot
        self.request_indices = torch.tensor(
            [request_slot], dtype=torch.int64, device=self.device
        )
        self.token_pool = attention_backend.token_to_kv_pool
        self.request_pool = attention_backend.req_to_token_pool
        assert isinstance(self.token_pool, HybridLinearKVPool)
        assert isinstance(self.request_pool, HybridReqToTokenPool)
        self.request_pool.mamba_pool.clear_slots(
            self.request_pool.get_mamba_indices(self.request_indices)
        )
        self.request_pool.req_to_token[request_slot, : cache_slots.numel()] = (
            cache_slots.to(torch.int32)
        )
        for layer_id in self.token_pool.full_attention_layer_id_mapping:
            self.token_pool.get_key_buffer(layer_id)[cache_slots] = 0
            self.token_pool.get_value_buffer(layer_id)[cache_slots] = 0
        self.temperature = torch.ones(1, device=self.device)
        self.top_k = torch.ones(1, dtype=torch.int64, device=self.device)
        self.top_p = torch.ones(1, device=self.device)

    def snapshot(self) -> Drama3StateSnapshot:
        return Drama3StateSnapshot.capture(
            self.token_pool, self.request_pool, self.cache_slots, self.request_indices
        )

    def batch(
        self, frames: torch.Tensor, start: int, mode: ForwardMode
    ) -> ForwardBatch:
        row_count = frames.shape[0]
        assert start + row_count <= self.cache_slots.numel()
        return ForwardBatch(
            forward_mode=mode,
            batch_size=1,
            input_ids=frames[:, 0].contiguous(),
            req_pool_indices=self.request_indices,
            seq_lens=torch.tensor(
                [start + row_count], dtype=torch.int32, device=self.device
            ),
            seq_lens_cpu=torch.tensor([start + row_count], dtype=torch.int32),
            out_cache_loc=self.cache_slots[start : start + row_count],
            seq_lens_sum=start + row_count,
            positions=torch.arange(start, start + row_count, device=self.device),
            extend_num_tokens=row_count,
            extend_seq_lens=torch.tensor(
                [row_count], dtype=torch.int32, device=self.device
            ),
            extend_prefix_lens=torch.tensor(
                [start], dtype=torch.int32, device=self.device
            ),
            extend_start_loc=torch.zeros(1, dtype=torch.int32, device=self.device),
            extend_prefix_lens_cpu=[start],
            extend_seq_lens_cpu=[row_count],
            extend_logprob_start_lens_cpu=[row_count],
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=torch.tensor(
                row_count, dtype=torch.int32, device=self.device
            ),
            num_token_non_padded_cpu=row_count,
            rids=[self.request.rid],
        )

    def target_logits(
        self, positions: torch.Tensor, *, eos_position: int | None = None
    ) -> torch.Tensor:
        """Controlled semantic draws; acoustic heads and all attention stay real."""
        decoder = self.model.audio_decoder
        token_ids = decoder.semantic_begin_id + (positions + 1) % decoder.codebook_size
        if eos_position is not None:
            token_ids = torch.where(
                positions == eos_position, self.model.config.eos_token_id, token_ids
            )
        logits = torch.full(
            (positions.numel(), decoder.main_token_candidate_ids.numel()),
            -torch.inf,
            device=self.device,
        )
        logits[
            decoder.main_token_candidate_ids.unsqueeze(0) == token_ids.unsqueeze(1)
        ] = 0
        return logits

    def prefill(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.batch(frames, 0, ForwardMode.EXTEND)
        self.attention_backend.init_forward_metadata(batch)
        hidden_states = self.model.model.forward_backbone(
            batch.input_ids,
            batch.positions,
            batch,
            input_embeds=self.model.audio_decoder.embed_backbone(
                frames, self.model.model.embed_tokens(frames[:, 0])
            ),
        )
        labels = torch.cat((frames[1:], frames[-1:]), dim=0)
        self.model.audio_decoder.prefill(
            hidden_states,
            batch.positions,
            frames,
            labels,
            PagedTowerAttention(self.model.tower_attention, batch),
        )
        last_position = frames.shape[0] - 1
        last_batch = self.batch(frames[-1:], last_position, ForwardMode.DECODE)
        self.attention_backend.full_attn_backend.init_forward_metadata(last_batch)
        window = torch.full(
            (1, RAS_WINDOW_FRAMES),
            EMPTY_RAS_SLOT,
            dtype=torch.int64,
            device=self.device,
        )
        history = frames[-RAS_WINDOW_FRAMES:, 0]
        window[:, -history.numel() :] = history
        anchor = self.model.audio_decoder.decode_step(
            self.target_logits(last_batch.positions),
            hidden_states[-1:],
            last_batch.positions,
            frames[-1:],
            window,
            self.temperature,
            self.top_k,
            self.top_p,
            PagedTowerAttention(self.model.tower_attention, last_batch),
        )
        return anchor, torch.cat((window[:, 1:], anchor[:, :1]), dim=1)

    def advance(
        self,
        input_frame: torch.Tensor,
        position: int,
        window: torch.Tensor,
        *,
        eos_position: int | None = None,
    ) -> OracleStep:
        batch = self.batch(input_frame, position, ForwardMode.DECODE)
        self.attention_backend.init_forward_metadata(batch)
        hidden_states, logits_output = self.model.backbone_logits(
            batch.input_ids, batch.positions, batch, input_frame
        )
        frame = self.model.audio_decoder.decode_step(
            self.target_logits(batch.positions, eos_position=eos_position),
            hidden_states,
            batch.positions,
            input_frame,
            window,
            self.temperature,
            self.top_k,
            self.top_p,
            PagedTowerAttention(self.model.tower_attention, batch),
        )
        return OracleStep(
            hidden_states=hidden_states.clone(),
            semantic_logits=logits_output.next_token_logits.clone(),
            frame=frame,
            ras_window=torch.cat((window[:, 1:], frame[:, :1]), dim=1),
        )

    def tower_steps(
        self, frames: torch.Tensor, start: int
    ) -> Iterator[PagedTowerAttention]:
        for row in range(frames.shape[0]):
            batch = self.batch(frames[row : row + 1], start + row, ForwardMode.DECODE)
            self.attention_backend.full_attn_backend.init_forward_metadata(batch)
            yield PagedTowerAttention(self.model.tower_attention, batch)


def compare_snapshots(
    actual: Drama3StateSnapshot, expected: Drama3StateSnapshot
) -> None:
    """Exact equality includes untouched suffix slots, conv and recurrent tensors."""
    actual_kv, actual_mamba = actual.cache_copy
    expected_kv, expected_mamba = expected.cache_copy
    assert actual_mamba is not None and expected_mamba is not None
    for actual_layer, expected_layer in zip(actual_kv, expected_kv, strict=True):
        for actual_chunk, expected_chunk in zip(
            actual_layer, expected_layer, strict=True
        ):
            for actual_tensor, expected_tensor in zip(
                actual_chunk, expected_chunk, strict=True
            ):
                torch.testing.assert_close(
                    actual_tensor, expected_tensor, rtol=0, atol=0
                )
    for actual_tensor, expected_tensor in zip(
        actual_mamba[0], expected_mamba[0], strict=True
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    torch.testing.assert_close(actual_mamba[1], expected_mamba[1], rtol=0, atol=0)

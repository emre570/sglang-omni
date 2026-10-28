"""Single-request, eager pair proposals and chronological target verification."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields

import torch
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MambaPool,
)

from sglang_omni.models.fishaudio_drama3.audio_decoder import AudioDecoder
from sglang_omni.models.fishaudio_drama3.decoder_layers import DecoderAttention
from sglang_omni.models.fishaudio_drama3.sampling import RAS_WINDOW_FRAMES

KVCacheCopy = list[list[list[torch.Tensor]]]
MambaCacheCopy = tuple[list[torch.Tensor], torch.Tensor]
HybridCacheCopy = tuple[KVCacheCopy, MambaCacheCopy | None]


@dataclass(frozen=True, kw_only=True)
class PairProposalBlock:
    """Backbone inputs are the complete anchor followed by all but the last pair."""

    pairs: torch.Tensor
    input_frames: torch.Tensor
    positions: torch.Tensor
    eos_token_id: int

    @classmethod
    def build(
        cls,
        decoder: AudioDecoder,
        anchor_frame: torch.Tensor,
        pairs: torch.Tensor,
        *,
        start_position: int,
        eos_token_id: int,
    ) -> PairProposalBlock:
        """Build K inputs for K proposed outputs; pairs use original semantic IDs."""
        if decoder.backbone_codebook_count != 1 or decoder.num_codebooks < 2:
            raise ValueError("pair proposals require exactly one backbone codebook")
        else:
            pass
        assert pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.shape[0] > 0
        assert anchor_frame.shape == (1, 1 + decoder.num_codebooks)
        assert pairs.dtype == anchor_frame.dtype == torch.int64
        assert (
            pairs.device
            == anchor_frame.device
            == decoder.main_token_candidate_ids.device
        )
        assert start_position >= 0
        assert bool(decoder.is_semantic(anchor_frame[:, 0]).all())
        assert bool(
            (
                (anchor_frame[:, 1:] >= 0)
                & (anchor_frame[:, 1:] < decoder.codebook_size)
            ).all()
        )
        assert bool(
            (anchor_frame[:, 1] == anchor_frame[:, 0] - decoder.semantic_begin_id).all()
        )
        assert bool((decoder.main_token_candidate_ids == eos_token_id).any())
        assert not bool(
            decoder.is_semantic(torch.tensor(eos_token_id, device=pairs.device))
        )
        semantic_ids = pairs[:, 0]
        assert bool(
            (decoder.is_semantic(semantic_ids) | (semantic_ids == eos_token_id)).all()
        )
        assert bool(((pairs[:, 1] >= 0) & (pairs[:, 1] < decoder.codebook_size)).all())
        assert not bool((semantic_ids[:-1] == eos_token_id).any())
        input_frames = anchor_frame.new_zeros((pairs.shape[0], anchor_frame.shape[1]))
        input_frames[:1].copy_(anchor_frame)
        input_frames[1:, 0] = semantic_ids[:-1]
        input_frames[1:, 1] = (semantic_ids[:-1] - decoder.semantic_begin_id).clamp(
            0, decoder.codebook_size - 1
        )
        input_frames[1:, 2] = pairs[:-1, 1]
        return cls(
            pairs=pairs.clone(),
            input_frames=input_frames,
            positions=torch.arange(
                start_position, start_position + pairs.shape[0], device=pairs.device
            ),
            eos_token_id=eos_token_id,
        )


@dataclass(frozen=True, kw_only=True)
class Drama3StateSnapshot:
    """Slow offline snapshot of owned KV slots and one plain conv/SSM request state."""

    token_to_kv_pool: HybridLinearKVPool
    request_pool: HybridReqToTokenPool
    cache_slots: torch.Tensor
    request_indices: torch.Tensor
    mamba_slots: torch.Tensor
    cache_copy: HybridCacheCopy

    @classmethod
    def capture(
        cls,
        token_to_kv_pool: HybridLinearKVPool,
        request_pool: HybridReqToTokenPool,
        cache_slots: torch.Tensor,
        request_indices: torch.Tensor,
    ) -> Drama3StateSnapshot:
        """Caller keeps all slots allocated until the snapshot is retired."""
        if (
            type(token_to_kv_pool) is not HybridLinearKVPool
            or type(request_pool) is not HybridReqToTokenPool
        ):
            raise ValueError("offline snapshots require static hybrid pools")
        else:
            pass
        state = request_pool.mamba_pool.mamba_cache
        if type(state) is not MambaPool.State or any(
            getattr(state, field.name) is not None
            for field in fields(state)
            if field.name not in {"conv", "temporal"}
        ):
            raise ValueError(
                "offline snapshots do not support speculative or ReplaySSM state"
            )
        else:
            pass
        assert token_to_kv_pool.mamba_pool is request_pool.mamba_pool
        assert cache_slots.ndim == request_indices.ndim == 1
        assert request_indices.numel() == 1 and cache_slots.numel() > 0
        mamba_slots = request_pool.get_mamba_indices(request_indices)
        return cls(
            token_to_kv_pool=token_to_kv_pool,
            request_pool=request_pool,
            cache_slots=cache_slots.clone(),
            request_indices=request_indices.clone(),
            mamba_slots=mamba_slots.clone(),
            cache_copy=token_to_kv_pool.get_cpu_copy(cache_slots, mamba_slots),
        )

    def restore(self) -> None:
        """Restore tensors, not allocator ownership, request lengths or RNG."""
        if not torch.equal(
            self.request_pool.get_mamba_indices(self.request_indices), self.mamba_slots
        ):
            raise RuntimeError(
                "request-to-Mamba mapping changed since snapshot capture"
            )
        else:
            self.token_to_kv_pool.load_cpu_copy(
                self.cache_copy, self.cache_slots, self.mamba_slots
            )


@dataclass(frozen=True, kw_only=True)
class PairVerificationResult:
    """Complete target frames, including a correction on mismatch; no bonus row."""

    frames: torch.Tensor
    matched_pair_count: int
    ras_window: torch.Tensor
    reached_eos: bool

    @property
    def consumed_input_count(self) -> int:
        """Backbone prefix length to retain or restore, not a cache commit."""
        return self.frames.shape[0]


@torch.inference_mode()
def verify_pair_block(
    decoder: AudioDecoder,
    block: PairProposalBlock,
    *,
    semantic_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    previous_main_tokens: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    attention_steps: Iterator[DecoderAttention],
) -> PairVerificationResult:
    """Sample full frames until mismatch/EOS; attention_steps binds one row lazily.

    Writes tower KV and advances target RNG. Does not restore backbone state.
    """
    if decoder.backbone_codebook_count != 1:
        raise ValueError("pair verification requires exactly one backbone codebook")
    else:
        pass
    row_count = block.pairs.shape[0]
    assert hidden_states.ndim == 2 and hidden_states.shape[0] == row_count
    assert semantic_logits.shape == (
        row_count,
        decoder.main_token_candidate_ids.numel(),
    )
    assert previous_main_tokens.shape == (1, RAS_WINDOW_FRAMES)
    assert temperature.shape == top_k.shape == top_p.shape == (1,)
    assert hidden_states.device == semantic_logits.device == block.pairs.device
    assert (
        previous_main_tokens.device
        == temperature.device
        == top_k.device
        == top_p.device
        == block.pairs.device
    )
    input_frame = block.input_frames[:1]
    ras_window = previous_main_tokens.clone()
    frames: list[torch.Tensor] = []
    matched_pair_count = 0
    reached_eos = False
    for row_index in range(row_count):
        frame = decoder.decode_step(
            semantic_logits[row_index : row_index + 1],
            hidden_states[row_index : row_index + 1],
            block.positions[row_index : row_index + 1],
            input_frame,
            ras_window,
            temperature,
            top_k,
            top_p,
            next(attention_steps),
        )
        frames.append(frame)
        ras_window = torch.cat((ras_window[:, 1:], frame[:, :1]), dim=1)
        reached_eos = int(frame[0, 0]) == block.eos_token_id
        semantic_matches = bool(frame[0, 0] == block.pairs[row_index, 0])
        pair_matches = semantic_matches and (
            reached_eos or bool(frame[0, 2] == block.pairs[row_index, 1])
        )
        matched_pair_count += int(pair_matches)
        if reached_eos or not pair_matches:
            break
        else:
            input_frame = frame
    return PairVerificationResult(
        frames=torch.cat(frames, dim=0),
        matched_pair_count=matched_pair_count,
        ras_window=ras_window,
        reached_eos=reached_eos,
    )

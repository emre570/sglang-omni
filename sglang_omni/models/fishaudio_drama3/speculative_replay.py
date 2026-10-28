"""Offline restore-and-replay oracle; deliberately not a serving commit path."""

from __future__ import annotations

from dataclasses import replace

import torch
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend

from sglang_omni.models.fishaudio_drama3.sglang_model import Drama3ForCausalLM
from sglang_omni.models.fishaudio_drama3.speculative import (
    Drama3StateSnapshot,
    PairProposalBlock,
    PairVerificationResult,
)
from sglang_omni.models.fishaudio_drama3.tower_prefill import PagedTowerAttention


@torch.inference_mode()
def restore_and_replay_prefix(
    model: Drama3ForCausalLM,
    snapshot: Drama3StateSnapshot,
    block: PairProposalBlock,
    result: PairVerificationResult,
    forward_batch: ForwardBatch,
) -> None:
    """Restore entry state, then teacher-force committed full frames without RNG.

    Caller retains slot ownership and publishes lengths/RAS only after success.
    """
    attention_backend = get_attn_backend()
    assert isinstance(attention_backend, HybridLinearAttnBackend)
    assert snapshot.token_to_kv_pool is attention_backend.token_to_kv_pool
    assert snapshot.request_pool is attention_backend.req_to_token_pool
    assert torch.equal(snapshot.request_indices, forward_batch.req_pool_indices)
    assert forward_batch.batch_size == 1
    assert forward_batch.forward_mode == ForwardMode.EXTEND
    assert torch.equal(forward_batch.positions, block.positions)
    assert 0 < result.consumed_input_count <= block.pairs.shape[0]
    assert result.frames.shape[1] == block.input_frames.shape[1]
    assert bool(torch.isin(forward_batch.out_cache_loc, snapshot.cache_slots).all())
    input_frames = torch.cat((block.input_frames[:1], result.frames[:-1]), dim=0)
    snapshot.restore()
    for row in range(result.consumed_input_count):
        sequence_length = int(block.positions[row]) + 1
        step_batch = replace(
            forward_batch,
            forward_mode=ForwardMode.DECODE,
            input_ids=input_frames[row : row + 1, 0],
            positions=block.positions[row : row + 1],
            out_cache_loc=forward_batch.out_cache_loc[row : row + 1],
            seq_lens=forward_batch.seq_lens.new_tensor([sequence_length]),
            seq_lens_cpu=torch.tensor([sequence_length], dtype=torch.int32),
            seq_lens_sum=sequence_length,
            num_token_non_padded=torch.tensor(
                1, dtype=torch.int32, device=forward_batch.input_ids.device
            ),
            num_token_non_padded_cpu=1,
            return_logprob=False,
            next_token_logits_buffer=None,
        )
        attention_backend.init_forward_metadata(step_batch)
        hidden_states, _ = model.backbone_logits(
            step_batch.input_ids,
            step_batch.positions,
            step_batch,
            input_frames[row : row + 1],
        )
        model.audio_decoder.prefill(
            hidden_states,
            step_batch.positions,
            input_frames[row : row + 1],
            result.frames[row : row + 1],
            PagedTowerAttention(model.tower_attention, step_batch),
        )

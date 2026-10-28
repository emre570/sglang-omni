"""GPU subprocess exercising real backbone blocks and slow prefix recovery."""

from __future__ import annotations

import argparse
import json
import subprocess
from array import array
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import sglang
import torch
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.model_runner.sglang_execution import attn_forward_context
from sglang_omni.models.fishaudio_drama3.sglang_model import Drama3ForCausalLM
from sglang_omni.models.fishaudio_drama3.speculative import (
    PairProposalBlock,
    verify_pair_block,
)
from sglang_omni.models.fishaudio_drama3.speculative_replay import (
    restore_and_replay_prefix,
)
from tests.unit_test.fishaudio_drama3.mtp_fixtures import prompt_rows
from tests.unit_test.fixtures.drama3_speculative import (
    OfflinePairEngineBuilder,
    OfflinePairRequest,
    OracleStep,
    compare_snapshots,
)


@dataclass(frozen=True, kw_only=True)
class RecoveryCase:
    block_length: int
    mismatch_row: int | None = None
    mismatch_component: Literal["semantic", "acoustic"] = "semantic"
    eos_row: int | None = None
    predict_eos: bool = False


@dataclass(frozen=True, kw_only=True)
class RecoveryMetrics:
    prefix_length: int
    case: RecoveryCase
    consumed_input_count: int
    continuation_frames: int
    hidden_relative_rms: float
    logits_relative_rms: float
    hidden_max_absolute_error: float
    logits_max_absolute_error: float


@torch.inference_mode()
def check_case(
    request: OfflinePairRequest,
    prefix: torch.Tensor,
    case: RecoveryCase,
    relative_rms_limit: float,
) -> RecoveryMetrics:
    start = prefix.shape[0]
    decoder = request.model.audio_decoder
    anchor, window = request.prefill(prefix)
    entry = request.snapshot()
    initial_rng = torch.cuda.get_rng_state(request.device)
    input_frame = anchor
    serial_window = window
    serial_steps: list[OracleStep] = []
    serial_states = []
    serial_rng = []
    eos_position = None if case.eos_row is None else start + case.eos_row
    for row in range(case.block_length + 3):
        step = request.advance(
            input_frame, start + row, serial_window, eos_position=eos_position
        )
        serial_steps.append(step)
        serial_states.append(request.snapshot())
        serial_rng.append(torch.cuda.get_rng_state(request.device))
        input_frame = step.frame
        serial_window = step.ras_window
        if int(step.frame[0, 0]) == request.model.config.eos_token_id:
            break
    proposed_count = case.eos_row + 1 if case.predict_eos else case.block_length
    pairs = anchor.new_zeros((proposed_count, 2))
    for row in range(proposed_count):
        if row < len(serial_steps):
            pairs[row] = serial_steps[row].frame[0, [0, 2]]
        else:
            pairs[row, 0] = (
                decoder.semantic_begin_id + (start + row + 1) % decoder.codebook_size
            )
    if case.eos_row is not None and not case.predict_eos:
        pairs[case.eos_row, 0] = decoder.semantic_begin_id
    if case.mismatch_row is not None:
        column = 0 if case.mismatch_component == "semantic" else 1
        base = decoder.semantic_begin_id if column == 0 else 0
        pairs[case.mismatch_row, column] = (
            base + (pairs[case.mismatch_row, column] - base + 1) % decoder.codebook_size
        )
    block = PairProposalBlock.build(
        decoder,
        anchor,
        pairs,
        start_position=start,
        eos_token_id=request.model.config.eos_token_id,
    )
    entry.restore()
    compare_snapshots(request.snapshot(), entry)
    torch.cuda.set_rng_state(initial_rng, request.device)
    batch = request.batch(block.input_frames, start, ForwardMode.EXTEND)
    request.attention_backend.init_forward_metadata(batch)
    hidden_states, logits_output = request.model.proposal_backbone_logits(block, batch)
    consumed = min(
        proposed_count,
        proposed_count if case.mismatch_row is None else case.mismatch_row + 1,
        proposed_count if case.eos_row is None else case.eos_row + 1,
    )
    reference_hidden = torch.cat(
        [step.hidden_states for step in serial_steps[:consumed]]
    )
    reference_logits = torch.cat(
        [step.semantic_logits for step in serial_steps[:consumed]]
    )
    hidden_difference = hidden_states[:consumed].float() - reference_hidden.float()
    logits_difference = (
        logits_output.next_token_logits[:consumed].float() - reference_logits.float()
    )
    hidden_error = float(
        hidden_difference.square().mean().sqrt()
        / reference_hidden.float().square().mean().sqrt().clamp_min(1e-6)
    )
    logits_error = float(
        logits_difference.square().mean().sqrt()
        / reference_logits.float().square().mean().sqrt().clamp_min(1e-6)
    )
    print(
        json.dumps(
            {
                "case": asdict(case),
                "prefix_length": start,
                "hidden_relative_rms": hidden_error,
                "logits_relative_rms": logits_error,
            }
        ),
        flush=True,
    )
    assert hidden_error <= relative_rms_limit, "block/step hidden divergence"
    assert logits_error <= relative_rms_limit, "block/step logits divergence"
    result = verify_pair_block(
        decoder,
        block,
        semantic_logits=request.target_logits(
            block.positions, eos_position=eos_position
        ),
        hidden_states=hidden_states,
        previous_main_tokens=window,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        attention_steps=request.tower_steps(block.input_frames, start),
    )
    expected_frames = torch.cat([step.frame for step in serial_steps[:consumed]])
    assert result.consumed_input_count == consumed
    expected_matches = consumed - int(
        case.mismatch_row is not None
        or (case.eos_row is not None and not case.predict_eos)
    )
    assert result.matched_pair_count == expected_matches
    assert result.reached_eos == (case.eos_row is not None)
    torch.testing.assert_close(result.frames, expected_frames, rtol=0, atol=0)
    torch.testing.assert_close(
        result.ras_window, serial_steps[consumed - 1].ras_window, rtol=0, atol=0
    )
    assert torch.equal(
        torch.cuda.get_rng_state(request.device), serial_rng[consumed - 1]
    )
    restore_and_replay_prefix(request.model, entry, block, result, batch)
    assert torch.equal(
        torch.cuda.get_rng_state(request.device), serial_rng[consumed - 1]
    )
    compare_snapshots(request.snapshot(), serial_states[consumed - 1])
    input_frame = result.frames[-1:]
    serial_window = result.ras_window
    continuation_count = 0 if result.reached_eos else 3
    for row in range(consumed, consumed + continuation_count):
        actual = request.advance(input_frame, start + row, serial_window)
        expected = serial_steps[row]
        torch.testing.assert_close(
            actual.hidden_states, expected.hidden_states, rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual.semantic_logits, expected.semantic_logits, rtol=0, atol=0
        )
        torch.testing.assert_close(actual.frame, expected.frame, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.ras_window, expected.ras_window, rtol=0, atol=0
        )
        compare_snapshots(request.snapshot(), serial_states[row])
        input_frame, serial_window = actual.frame, actual.ras_window
    return RecoveryMetrics(
        prefix_length=start,
        case=case,
        consumed_input_count=consumed,
        continuation_frames=continuation_count,
        hidden_relative_rms=hidden_error,
        logits_relative_rms=logits_error,
        hidden_max_absolute_error=float(hidden_difference.abs().max()),
        logits_max_absolute_error=float(logits_difference.abs().max()),
    )


def run_engine(model_dir: str, relative_rms_limit: float) -> list[RecoveryMetrics]:
    scheduler = OfflinePairEngineBuilder().build(model_dir, dtype="bfloat16")
    runner = scheduler.tp_worker.model_runner
    model = runner.model
    assert isinstance(model, Drama3ForCausalLM)
    model.eval()
    backend = runner.attn_backend
    request_pool = runner.req_to_token_pool
    token_pool = runner.token_to_kv_pool
    allocator = runner.token_to_kv_pool_allocator
    assert isinstance(backend, HybridLinearAttnBackend)
    assert isinstance(request_pool, HybridReqToTokenPool)
    assert isinstance(token_pool, HybridLinearKVPool)
    assert model.audio_decoder.sampler is None
    assert model.audio_decoder.fused_layers is None
    cases = []
    for length in (1, 2, 4):
        cases.append(RecoveryCase(block_length=length))
        cases.extend(
            RecoveryCase(
                block_length=length, mismatch_row=row, mismatch_component=component
            )
            for row in range(length)
            for component in ("semantic", "acoustic")
        )
        cases.append(
            RecoveryCase(block_length=length, eos_row=length - 1, predict_eos=True)
        )
        cases.append(RecoveryCase(block_length=length, eos_row=0))
    reports = []
    try:
        with torch.inference_mode(), attn_forward_context(backend):
            for prefix_length in (5, 65):
                prefix_rows = prompt_rows(prefix_length, seed=17)
                req = Req(
                    rid="pair-recovery-oracle",
                    origin_input_text="",
                    origin_input_ids=array("q", [frame[0] for frame in prefix_rows]),
                    sampling_params=SamplingParams(max_new_tokens=16),
                )
                free_counts = (
                    request_pool.available_size(),
                    request_pool.mamba_allocator.available_size(),
                    allocator.available_size(),
                )
                indices = request_pool.alloc([req])
                assert indices is not None
                slots = allocator.alloc(128)
                assert slots is not None
                try:
                    request = OfflinePairRequest(model, backend, req, slots)
                    prefix = torch.tensor(
                        prefix_rows, dtype=torch.int64, device=slots.device
                    )
                    empty_state = request.snapshot()
                    for case in cases:
                        empty_state.restore()
                        torch.manual_seed(123)
                        reports.append(
                            check_case(request, prefix, case, relative_rms_limit)
                        )
                finally:
                    allocator.free(slots)
                    request_pool.free_mamba_cache(req)
                    request_pool.free(req)
                assert free_counts == (
                    request_pool.available_size(),
                    request_pool.mamba_allocator.available_size(),
                    allocator.available_size(),
                )
    finally:
        scheduler.shutdown_resources()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-rms-limit", type=float, default=0.02)
    args = parser.parse_args()
    source_path = Path(sglang.__file__).resolve()
    source_commit = subprocess.run(
        ["git", "-C", str(source_path.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "sglang_version": version("sglang"),
        "sglang_source": str(source_path),
        "sglang_commit": (
            source_commit.stdout.strip() if source_commit.returncode == 0 else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "relative_rms_limit": args.relative_rms_limit,
        "cases": [],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    reports = run_engine(args.model_dir, args.relative_rms_limit)
    report["cases"] = [asdict(result) for result in reports]
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

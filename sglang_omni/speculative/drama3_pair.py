# SPDX-License-Identifier: Apache-2.0
"""SGLang speculative lifecycle adapter for the Drama 3 frame protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo

from sglang_omni.speculative.target_worker import SpeculativeTargetWorker

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.server_args import ServerArgs


DRAMA3_PAIR_ALGORITHM = "DRAMA3_PAIR"


class Drama3PairAlgorithm(CustomSpecAlgo):
    """Describe Drama 3 frame speculation to SGLang's scheduler."""

    def has_draft_kv(self) -> bool:
        return False

    def supports_target_verify_for_draft(self) -> bool:
        return False

    def get_num_tokens_per_req_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:
        del num_draft_tokens, is_draft_worker
        return 1


def validate_drama3_pair_server_args(server_args: ServerArgs) -> None:
    if server_args.speculative_draft_model_path is not None:
        raise ValueError(
            "DRAMA3_PAIR does not load a separate draft model; omit "
            "speculative_draft_model_path"
        )
    elif server_args.speculative_num_draft_tokens not in (None, 1):
        raise ValueError(
            "DRAMA3_PAIR currently exposes one committed frame per scheduler "
            "step; set speculative_num_draft_tokens=1"
        )
    elif server_args.speculative_num_steps not in (None, 0, 1):
        raise ValueError(
            "DRAMA3_PAIR currently supports speculative_num_steps in {0, 1}"
        )
    else:
        pass
    server_args.speculative_num_draft_tokens = 1
    server_args.speculative_num_steps = 0


class Drama3PairWorker(BaseSpecWorker):
    """Run Drama 3 through SGLang's spec-v2 commit and streaming contract.

    The current serving stage commits one target-produced frame. This keeps the
    scheduler integration correct while the multi-frame proposal transaction is
    added behind the same worker.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: SpeculativeTargetWorker,
    ) -> None:
        del server_args, gpu_id, nccl_port
        super().__init__()
        self._target_worker = target_worker
        self._draft_worker = None
        self.device = target_worker.device
        self.speculative_num_steps = 0
        self.speculative_num_draft_tokens = 1
        self.req_to_token_pool = None
        self.token_to_kv_pool_allocator = None

    @property
    def spec_v2_attn_backends(self) -> tuple[object, ...]:
        return (self.target_worker.model_runner.attn_backend,)

    def alloc_memory_pool(
        self,
        memory_pool_config: object = None,
        req_to_token_pool: object = None,
        token_to_kv_pool_allocator: object = None,
    ) -> None:
        del memory_pool_config
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self) -> None:
        return None

    def init_cuda_graphs(self) -> None:
        return None

    def update_weights_from_disk(self, payload: object) -> tuple[bool, str]:
        return self.target_worker.model_worker.update_weights_from_disk(payload)

    def update_weights_from_ipc(self, payload: object) -> tuple[bool, str]:
        return self.target_worker.model_worker.update_weights_from_ipc(payload)

    def update_weights_from_tensor(self, payload: object) -> tuple[bool, str]:
        return self.target_worker.model_worker.update_weights_from_tensor(payload)

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish: object = None,
        grammar_barrier: object = None,
    ) -> GenerationBatchResult:
        del grammar_barrier
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            result = self.target_worker.forward_batch_generation(
                batch,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
            new_seq_lens = batch.seq_lens
        elif batch.forward_mode.is_decode():
            self._set_single_frame_output_slots(batch)
            result = self.target_worker.forward_batch_generation(batch)
            batch_size = len(batch.reqs)
            result.accept_lens = torch.ones(batch_size, dtype=torch.int32)
            result.num_correct_drafts = 0
            result.num_correct_drafts_per_req_cpu = [0] * batch_size
            result.speculative_num_draft_tokens = 1
            new_seq_lens = batch.seq_lens + result.accept_lens.to(batch.seq_lens.device)
        else:
            result = self.target_worker.forward_batch_generation(batch)
            new_seq_lens = batch.seq_lens

        result.new_seq_lens = new_seq_lens
        result.next_draft_input = None
        if on_publish is not None:
            on_publish(new_seq_lens)
        return result

    def _set_single_frame_output_slots(self, batch: ScheduleBatch) -> None:
        if self.req_to_token_pool is None:
            raise RuntimeError("DRAMA3_PAIR memory pools are not initialized")
        batch.out_cache_loc = self.req_to_token_pool.req_to_token[
            batch.req_pool_indices, batch.seq_lens
        ]


@SpeculativeAlgorithm.register(
    DRAMA3_PAIR_ALGORITHM,
    supports_overlap=False,
    validate_server_args=validate_drama3_pair_server_args,
    spec_class=Drama3PairAlgorithm,
)
def create_drama3_pair_worker(server_args: ServerArgs) -> type[Drama3PairWorker]:
    del server_args
    return Drama3PairWorker


__all__ = [
    "DRAMA3_PAIR_ALGORITHM",
    "Drama3PairAlgorithm",
    "Drama3PairWorker",
    "create_drama3_pair_worker",
    "validate_drama3_pair_server_args",
]

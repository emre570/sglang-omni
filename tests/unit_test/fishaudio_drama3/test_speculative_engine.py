"""Real CUDA backbone block/step equivalence and rejected-prefix recovery."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit_test.fishaudio_drama3.mtp_fixtures import write_synthetic_checkpoint
from tests.unit_test.fixtures.accelerator import require_cuda

pytestmark = [pytest.mark.accelerator, pytest.mark.benchmark]


def test_pair_backbone_and_recovery_on_gpu(tmp_path: Path) -> None:
    """Real hybrid state must survive every rejection and three continuation frames."""
    require_cuda()
    checkpoint_dir = tmp_path / "checkpoint"
    write_synthetic_checkpoint(checkpoint_dir)
    output_path = tmp_path / "report.json"
    log_path = tmp_path / "engine.log"
    print(f"Drama3 pair oracle artifacts: {tmp_path}", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.unit_test.fishaudio_drama3.speculative_engine",
                "--model-dir",
                str(checkpoint_dir),
                "--output",
                str(output_path),
            ],
            cwd=Path(__file__).resolve().parents[3],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            pytest.fail(f"pair oracle timed out; see {log_path}")
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    assert return_code == 0, f"pair oracle failed; artifacts: {tmp_path}\n" + "\n".join(
        log_path.read_text(errors="replace").splitlines()[-100:]
    )
    report = json.loads(output_path.read_text())
    assert report["sglang_source"]
    assert len(report["cases"]) == 46
    assert {case["prefix_length"] for case in report["cases"]} == {5, 65}
    assert all(
        case["continuation_frames"] == 3
        for case in report["cases"]
        if case["case"]["eos_row"] is None
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"}, indent=2
        )
    )

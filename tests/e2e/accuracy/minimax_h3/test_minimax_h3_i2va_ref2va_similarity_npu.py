# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend variant of the official MiniMax-H3 I2VA/Ref2VA E2E gate.

Requests, reference assets, seeds, and default thresholds are shared with the
CUDA test. Only the server topology is replaced with the validated 8-card A3
configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import requests

from tests.e2e.accuracy.minimax_h3 import test_minimax_h3_i2va_ref2va_similarity as reference_test
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.benchmark, pytest.mark.diffusion, pytest.mark.full_model]

NPU_COUNT = 8
NPU_ATTENTION_BACKEND_ENV_VAR = "VLLM_TEST_MINIMAX_H3_NPU_ATTENTION_BACKEND"
NPU_SSIM_THRESHOLD_ENV_VAR = "VLLM_TEST_MINIMAX_H3_NPU_SSIM_THRESHOLD"
NPU_PSNR_THRESHOLD_ENV_VAR = "VLLM_TEST_MINIMAX_H3_NPU_PSNR_THRESHOLD"
LOCAL_ASSET_DIR_ENV_VAR = "VLLM_TEST_MINIMAX_H3_ASSET_DIR"


def _require_npu() -> None:
    from vllm_omni.platforms import current_omni_platform

    if not current_omni_platform.is_npu() or not current_omni_platform.is_available():
        pytest.skip("MiniMax H3 NPU accuracy test requires an Ascend NPU runtime.")
    device_count = current_omni_platform.device_count()
    if device_count < NPU_COUNT:
        pytest.skip(f"MiniMax H3 NPU accuracy test requires {NPU_COUNT} NPUs, found {device_count}.")


def _threshold(env_var: str, default: float) -> float:
    value = float(os.environ.get(env_var, default))
    if value < 0:
        raise ValueError(f"{env_var} must be non-negative, got {value}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_asset(
    output_dir: Path,
    *,
    filename: str,
    local_relative_path: str,
    url: str,
    sha256: str,
    label: str,
) -> Path:
    asset_dir = os.environ.get(LOCAL_ASSET_DIR_ENV_VAR)
    if not asset_dir:
        return reference_test._download_reference_asset(
            output_dir,
            filename=filename,
            url=url,
            sha256=sha256,
            label=label,
        )

    asset_path = Path(asset_dir).expanduser() / local_relative_path
    if not asset_path.is_file():
        raise FileNotFoundError(f"MiniMax H3 {label} asset not found: {asset_path}")
    digest = _file_sha256(asset_path)
    assert digest == sha256, f"MiniMax H3 {label} asset checksum mismatch: got {digest}, expected {sha256}"
    return asset_path


def _server_args() -> list[str]:
    attention_backend = os.environ.get(NPU_ATTENTION_BACKEND_ENV_VAR, "FLASH_ATTN")
    return [
        "--trust-remote-code",
        "--num-gpus",
        str(NPU_COUNT),
        "--usp",
        str(NPU_COUNT),
        "--ring",
        "1",
        "--text-encoder-tp-size",
        str(NPU_COUNT),
        "--enable-distributed-layerwise-offload",
        "--vae-patch-parallel-size",
        str(NPU_COUNT),
        "--vae-parallel-mode",
        "tile",
        "--vae-use-tiling",
        "--diffusion-attention-backend",
        attention_backend,
        "--stage-init-timeout",
        "1800",
        "--init-timeout",
        "1800",
    ]


def _server_env(output_dir: Path) -> dict[str, str]:
    return {
        "PYTHONDONTWRITEBYTECODE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": str(reference_test.REQUEST_TIMEOUT_SECONDS),
        "VLLM_OMNI_STORAGE_PATH": str(output_dir / "storage"),
    }


def _assert_official_video(
    generated_path: Path,
    reference_path: Path,
    *,
    frame_count: int,
    label: str,
) -> None:
    reference_test._assert_official_video(
        generated_path,
        reference_path,
        frame_count=frame_count,
        label=label,
        ssim_threshold=_threshold(NPU_SSIM_THRESHOLD_ENV_VAR, reference_test.SSIM_THRESHOLD),
        psnr_threshold=_threshold(NPU_PSNR_THRESHOLD_ENV_VAR, reference_test.PSNR_THRESHOLD),
    )


@hardware_test(res={"npu": "A3"}, num_cards=NPU_COUNT)
def test_minimax_h3_i2va_matches_official_reference_npu(
    accuracy_artifact_root: Path,
) -> None:
    _require_npu()
    reference_test.probe_binary("ffmpeg")
    reference_test.probe_binary("ffprobe")
    output_dir = reference_test.reset_artifact_dir(accuracy_artifact_root / "minimax_h3_i2va_npu")
    reference_path = _resolve_asset(
        output_dir,
        filename="reference.mp4",
        local_relative_path="i2va/reference.mp4",
        url=reference_test.I2VA_REFERENCE_VIDEO_URL,
        sha256=reference_test.I2VA_REFERENCE_VIDEO_SHA256,
        label="I2VA",
    )
    image_path = _resolve_asset(
        output_dir,
        filename="input.png",
        local_relative_path="i2va/input.png",
        url=reference_test.I2VA_IMAGE_URL,
        sha256=reference_test.I2VA_IMAGE_SHA256,
        label="I2VA input image",
    )
    generated_path = output_dir / "vllm_omni.mp4"
    request_data = {
        "prompt": reference_test.I2VA_PROMPT,
        "width": str(reference_test.WIDTH),
        "height": str(reference_test.HEIGHT),
        "fps": str(reference_test.FPS),
        "num_inference_steps": str(reference_test.NUM_INFERENCE_STEPS),
        "flow_shift": str(reference_test.FLOW_SHIFT),
        "seed": str(reference_test.SEED),
        "extra_params": json.dumps(
            {
                "task": "fl2va",
                "duration": reference_test.I2VA_DURATION_SECONDS,
                "aspect_ratio": "auto",
                "frame_indices": [0],
                "audio_flow_shift": reference_test.AUDIO_FLOW_SHIFT,
            }
        ),
    }

    with OmniServer(
        reference_test._model_name(),
        _server_args(),
        env_dict=_server_env(output_dir),
        use_omni=True,
    ) as server:
        with image_path.open("rb") as image_file:
            response = requests.post(
                f"http://{server.host}:{server.port}/v1/videos/sync",
                data=request_data,
                files={"input_reference": ("input.png", image_file, "image/png")},
                timeout=reference_test.REQUEST_TIMEOUT_SECONDS,
            )

    response.raise_for_status()
    assert response.headers["content-type"].startswith("video/mp4")
    generated_path.write_bytes(response.content)
    _assert_official_video(
        generated_path,
        reference_path,
        frame_count=reference_test.I2VA_NUM_FRAMES,
        label="minimax_h3_i2va_npu_official_reference",
    )


@hardware_test(res={"npu": "A3"}, num_cards=NPU_COUNT)
def test_minimax_h3_ref2va_matches_official_reference_npu(
    accuracy_artifact_root: Path,
) -> None:
    _require_npu()
    reference_test.probe_binary("ffmpeg")
    reference_test.probe_binary("ffprobe")
    output_dir = reference_test.reset_artifact_dir(accuracy_artifact_root / "minimax_h3_ref2va_npu")
    reference_path = _resolve_asset(
        output_dir,
        filename="reference.mp4",
        local_relative_path="ref2va/reference.mp4",
        url=reference_test.REF2VA_REFERENCE_VIDEO_URL,
        sha256=reference_test.REF2VA_REFERENCE_VIDEO_SHA256,
        label="Ref2VA",
    )
    input_video_path = _resolve_asset(
        output_dir,
        filename="input_video.mp4",
        local_relative_path="ref2va/input_video.mp4",
        url=reference_test.REF2VA_INPUT_VIDEO_URL,
        sha256=reference_test.REF2VA_INPUT_VIDEO_SHA256,
        label="Ref2VA input video",
    )
    input_audio_path = _resolve_asset(
        output_dir,
        filename="input_audio.mp3",
        local_relative_path="ref2va/input_audio.mp3",
        url=reference_test.REF2VA_INPUT_AUDIO_URL,
        sha256=reference_test.REF2VA_INPUT_AUDIO_SHA256,
        label="Ref2VA input audio",
    )
    generated_path = output_dir / "vllm_omni.mp4"
    request_data = {
        "prompt": reference_test.REF2VA_PROMPT,
        "width": str(reference_test.WIDTH),
        "height": str(reference_test.HEIGHT),
        "fps": str(reference_test.FPS),
        "num_inference_steps": str(reference_test.NUM_INFERENCE_STEPS),
        "flow_shift": str(reference_test.FLOW_SHIFT),
        "seed": str(reference_test.SEED),
        "extra_params": json.dumps(
            {
                "task": "ref2va",
                "duration": reference_test.REF2VA_DURATION_SECONDS,
                "aspect_ratio": "auto",
                "audio_flow_shift": reference_test.AUDIO_FLOW_SHIFT,
            }
        ),
    }

    with OmniServer(
        reference_test._model_name(reference_test.REF2VA_MODEL_ENV_VAR, "Ref2VA"),
        _server_args(),
        env_dict=_server_env(output_dir),
        use_omni=True,
    ) as server:
        with input_video_path.open("rb") as video_file, input_audio_path.open("rb") as audio_file:
            response = requests.post(
                f"http://{server.host}:{server.port}/v1/videos/sync",
                data=request_data,
                files=[
                    ("input_references", ("input_video.mp4", video_file, "video/mp4")),
                    ("input_references", ("input_audio.mp3", audio_file, "audio/mpeg")),
                ],
                timeout=reference_test.REQUEST_TIMEOUT_SECONDS,
            )

    response.raise_for_status()
    assert response.headers["content-type"].startswith("video/mp4")
    generated_path.write_bytes(response.content)
    _assert_official_video(
        generated_path,
        reference_path,
        frame_count=reference_test.REF2VA_NUM_FRAMES,
        label="minimax_h3_ref2va_npu_official_reference",
    )

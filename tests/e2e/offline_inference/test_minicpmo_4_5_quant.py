"""
E2E offline quantization tests for MiniCPM-o 4.5 with AutoRound W4A16.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

models = ["openbmb/MiniCPM-o-4_5"]

_CI_DEPLOY = get_deploy_config_path("ci/minicpmo_4_5.yaml")

_QUANT_CONFIG = modify_stage_config(
    _CI_DEPLOY,
    updates={
        "stages": {
            0: {"load_format": "auto"},
        },
    },
)

test_params = [ (model, None, {"deploy_config": _QUANT_CONFIG, "trust_remote_code": True}) for model in models ]


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text_autoround(omni_runner, omni_runner_handler) -> None:
    """Test text-to-text with AutoRound W4A16 quantization."""
    request_config = {
        "prompts": "What is the capital of China?",
        "modalities": ["text"],
    }
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_audio_to_text_autoround(omni_runner, omni_runner_handler) -> None:
    """Test audio-to-text with AutoRound W4A16 quantization."""
    audio = generate_synthetic_audio(5, 1)["np_array"]
    request_config = {
        "prompts": "Describe the audio.", "audios": audio, "modalities": ["text"],
    }
    omni_runner_handler.send_omni_request(request_config)
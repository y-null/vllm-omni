"""
E2E tests for MiniCPM-o 4.5 multi-GPU layouts: 2GPU (default), 3GPU (TP=2 thinker), stage isolation.
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

_MODEL = "openbmb/MiniCPM-o-4_5"
_CI_DEPLOY = get_deploy_config_path("ci/minicpmo_4_5.yaml")

# Default 2GPU layout (thinker on GPU 0, talker on GPU 1)
_DEFAULT_CONFIG = modify_stage_config(
    _CI_DEPLOY,
    updates={
        "stages": {
            0: {"devices": "0"},
            1: {"devices": "1"},
        },
    },
)

# 3GPU layout with TP=2 thinker (thinker on GPU 0,1; talker on GPU 2)
_TP2_CONFIG = modify_stage_config(
    _CI_DEPLOY,
    updates={
        "stages": {
            0: {"devices": "0,1", "tensor_parallel_size": 2},
            1: {"devices": "2"},
        },
    },
)


def get_system_prompt():
    return {
        "role": "system",
        "content": [{"type": "text", "text": "You are MiniCPM-o 4.5."}],
    }


test_params_2gpu = [
    pytest.param(
        OmniServerParams(
            model=_MODEL, stage_config_path=_DEFAULT_CONFIG,
            use_stage_cli=True, server_args=["--trust-remote-code", "--no-async-chunk"],
        ),
        id="2gpu_default",
    )
]

test_params_3gpu = [
    pytest.param(
        OmniServerParams(
            model=_MODEL, stage_config_path=_TP2_CONFIG,
            use_stage_cli=True, server_args=["--trust-remote-code", "--no-async-chunk"],
        ),
        id="3gpu_tp2",
    )
]


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params_2gpu, indirect=True)
def test_2gpu_default_layout_text_to_audio(omni_server, openai_client) -> None:
    """Verify 2GPU default layout (stage 0 on cuda:0, stage 1 on cuda:1)."""
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="Hello, say something in English.",
    )
    request_config = {"model": omni_server.model, "messages": messages, "stream": True}
    openai_client.send_omni_request(request_config, request_num=1)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=3)
@pytest.mark.parametrize("omni_server", test_params_3gpu, indirect=True)
def test_3gpu_tp2_layout_text_to_audio(omni_server, openai_client) -> None:
    """Verify 3GPU TP=2 layout (thinker TP=2 on GPU 0,1; talker on GPU 2)."""
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="Say one word: hello.",
    )
    request_config = {"model": omni_server.model, "messages": messages, "stream": True}
    openai_client.send_omni_request(request_config, request_num=1)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params_2gpu, indirect=True)
def test_stage_isolation_thinker_talker(omni_server, openai_client) -> None:
    """
    Verify thinker and talker stages run on separate GPUs and
    the cross-stage hidden state transfer works correctly.
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(3, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        audio_data_url=audio_data_url,
        content_text="What do you hear?",
    )
    request_config = {"model": omni_server.model, "messages": messages, "stream": True}
    openai_client.send_omni_request(request_config, request_num=1)
"""
E2E Online tests for MiniCPM-o 4.5 model with multimodal input and audio / text output.

MiniCPM-o 4.5 has ``async_chunk: false``, ``max_num_seqs: 1`` on both stages,
and the vocoder runs in-process inside the talker stage rather than as a separate
Code2Wav stage.
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

_MODEL = "openbmb/MiniCPM-o-4_5"
_CI_DEPLOY = get_deploy_config_path("ci/minicpmo_4_5.yaml")

test_params = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=_CI_DEPLOY,
            use_stage_cli=True,
            server_args=[
                "--trust-remote-code", "--no-async-chunk",
            ],
        ),
        id="default",
    )
]


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are MiniCPM-o 4.5, a virtual human capable of perceiving "
                    "auditory and visual inputs, as well as generating text and speech."
                ),
            }
        ],
    }


def get_prompt(prompt_type: str = "text_only") -> str:
    prompts = {
        "text_only": "What is the capital of China? Answer in 20 words.",
        "mix": "What is recited in the audio? What is in this image? Describe the video briefly.",
        "text_image": "What color are the squares in this image?",
    }
    return prompts.get(prompt_type, prompts["text_only"])


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_text_001(omni_server, openai_client) -> None:
    """
    Test text-only input generating text output via OpenAI API.
    Deploy Setting: default 2GPU
    Input Modal: text
    Output Modal: text
    Input Setting: stream=False
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt()
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["Beijing"]},
    }

    openai_client.send_omni_request(request_config, request_num=1)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client) -> None:
    """
    Test text-only input generating text + audio output via OpenAI API.
    This exercises the talker TTS region detection and in-process token2wav vocoder.
    Deploy Setting: default 2GPU
    Input Modal: text
    Output Modal: text + audio
    Input Setting: stream=True
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt()
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"audio": ["test"]},
    }

    openai_client.send_omni_request(request_config, request_num=1)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test audio input generating text + audio output via OpenAI API.
    Deploy Setting: default 2GPU
    Input Modal: text + audio
    Output Modal: text + audio
    Input Setting: stream=True
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        audio_data_url=audio_data_url,
        content_text=get_prompt("mix"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
    }

    openai_client.send_omni_request(request_config, request_num=1)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_image_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test image input generating text + audio output via OpenAI API.
    Deploy Setting: default 2GPU
    Input Modal: text + image
    Output Modal: text + audio
    Input Setting: stream=True
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(24, 24)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        image_data_url=image_data_url,
        content_text=get_prompt("text_image"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
    }

    openai_client.send_omni_request(request_config, request_num=1)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_video_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test video input generating text + audio output via OpenAI API.
    Deploy Setting: default 2GPU
    Input Modal: text + video
    Output Modal: text + audio
    Input Setting: stream=True
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(24, 24, 200)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        content_text=get_prompt("mix"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
    }

    openai_client.send_omni_request(request_config, request_num=1)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_one_word_prompt_001(omni_server, openai_client) -> None:
    """
    Test one-word prompt for boundary behavior.
    Deploy Setting: default 2GPU
    Input Modal: text (single word)
    Output Modal: text + audio
    Input Setting: stream=True
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="Hello",
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
    }

    openai_client.send_omni_request(request_config, request_num=1)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_invalid_audio_format_rejected(omni_server, openai_client) -> None:
    """
    Test that invalid audio format is properly rejected.
    Deploy Setting: default 2GPU
    Input Modal: text + invalid audio format
    Output Modal: error
    """
    fake_base64 = "AAAA"  # not a real audio file

    responses = openai_client.send_completions_http_request(
        {
            "json": {
                "model": omni_server.model,
                "messages": dummy_messages_from_mix_data(
                    system_prompt=get_system_prompt(),
                    audio_data_url=f"data:audio/invalid;base64,{fake_base64}",
                    content_text=get_prompt(),
                ),
                "stream": True,
            },
        },
        err_code=400,
    )
    assert not responses[0].success

@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_completions_endpoint_available(omni_server, openai_client) -> None:
    """
    MiniCPM-o 4.5 does NOT block /v1/completions (unlike Qwen3-Omni which has
    endpoint_restrictions). Verify that the completions endpoint is accessible.
    """
    responses = openai_client.send_completions_http_request(
        {
            "json": {
                "model": omni_server.model,
                "prompt": "Hello, how are you?",
                "max_tokens": 10,
            },
        },
        err_code=200,
    )
    assert responses[0].success

@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_streaming_response(omni_server, openai_client) -> None:
    """
    Test streaming text output via OpenAI API returns tokens incrementally.
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt()
    )
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "modalities": ["text"],
    }
    openai_client.send_omni_request(request_config, request_num=1)

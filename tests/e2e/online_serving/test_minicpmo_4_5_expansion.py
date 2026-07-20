"""
E2E Online expansion tests for MiniCPM-o 4.5 covering modality combinations,
in-process token2wav vocoder behavior, max_num_seqs=1 serial execution,
and edge-case validation.
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.full_model, pytest.mark.omni]

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
                "--no-async-chunk",
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
    }
    return prompts.get(prompt_type, prompts["text_only"])


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mix_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test multi-modal input (text + audio + video + image) generating text + audio output.
    Deploy Setting: default 2GPU
    Input Modal: text + audio + video + image
    Output Modal: text + audio
    Input Setting: stream=True
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        image_data_url=image_data_url,
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
def test_text_video_to_text_001(omni_server, openai_client) -> None:
    """
    Test text + video input generating text output.
    Deploy Setting: default 2GPU
    Input Modal: text + video
    Output Modal: text
    Input Setting: stream=False
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        content_text=get_prompt("mix"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config, request_num=1)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_video_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test text + video input generating text + audio output.
    Deploy Setting: default 2GPU
    Input Modal: text + video
    Output Modal: text + audio
    Input Setting: stream=True
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
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
def test_text_image_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test text + image input generating text + audio output.
    Deploy Setting: default 2GPU
    Input Modal: text + image
    Output Modal: text + audio
    Input Setting: stream=True
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        image_data_url=image_data_url,
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
def test_sequential_requests_independent(omni_server, openai_client) -> None:
    """
    Verify that sequential requests produce independent results and state
    does not leak between requests. MiniCPM-o 4.5 has max_num_seqs=1 so
    requests are processed one-at-a-time; the second request must not
    receive the first request's audio or token state.
    Deploy Setting: default 2GPU
    Input Modal: text (two different prompts)
    Output Modal: text + audio (both)
    """
    messages_1 = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="What is the capital of France?",
    )
    messages_2 = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="What is the capital of China?",
    )

    # Send first request and ensure it completes
    openai_client.send_omni_request(
        {
            "model": omni_server.model,
            "messages": messages_1,
            "stream": True,
        },
        request_num=1,
    )

    # Send second request with different prompt; must produce its own output
    openai_client.send_omni_request(
        {
            "model": omni_server.model,
            "messages": messages_2,
            "stream": True,
            "key_words": {"text": ["Beijing"]},
        },
        request_num=1,
    )


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_output_valid_waveform(omni_server, openai_client) -> None:
    """
    Verify that the in-process token2wav vocoder inside the talker stage
    produces a valid audio waveform (non-empty, correctly sampled).
    MiniCPM-o 4.5 runs the vocoder in the talker process unlike Qwen3-Omni
    which has a separate Code2Wav stage.
    Deploy Setting: default 2GPU
    Input Modal: text
    Output Modal: text + audio
    Input Setting: stream=True
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="Say hello world.",
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"audio": ["hello"]},
    }

    responses = openai_client.send_omni_request(request_config, request_num=1)
    for response in responses:
        assert response.success, f"Request failed: {response.error}"
        if response.audio_bytes is not None:
            assert len(response.audio_bytes) > 0, "Audio output should not be empty"
            break
    else:
        pytest.fail("No audio output found in any response")


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_audio_long_output_001(omni_server, openai_client) -> None:
    """
    Test text input generating a longer audio output to exercise the
    token2wav decoder across multiple frames.
    Deploy Setting: default 2GPU
    Input Modal: text (longer prompt)
    Output Modal: text + audio
    Input Setting: stream=True
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="Tell me a short story about a cat in about 50 words.",
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"audio": ["cat"]},
    }

    openai_client.send_omni_request(request_config, request_num=1)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_in_video_001(omni_server, openai_client) -> None:
    """
    Test video input with embedded audio track.
    Deploy Setting: default 2GPU
    Input Modal: text + video (with audio track)
    Output Modal: text + audio
    Input Setting: stream=True
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        content_text="Describe the video and what you hear.",
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
def test_text_audio_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test text + audio input (as audio_data_url) generating text + audio output.
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        audio_data_url=audio_data_url,
        content_text="What do you hear in this audio?",
    )
    request_config = {"model": omni_server.model, "messages": messages, "stream": True}
    openai_client.send_omni_request(request_config, request_num=1)

@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_large_image_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test large image input generating text + audio output.
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        image_data_url=image_data_url,
        content_text="Describe this image.",
    )
    request_config = {"model": omni_server.model, "messages": messages, "stream": True}
    openai_client.send_omni_request(request_config, request_num=1)

@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_talker_in_process_vocoder_path(omni_server, openai_client) -> None:
    """
    Verify that the in-process token2wav vocoder inside the talker stage
    produces audio (not via dummy fallback). Sending a prompt that triggers
    speech output exercises the TTS region detection path in llm2tts bridge.
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="Say the word hello three times.",
    )
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"audio": ["hello"]},
    }
    responses = openai_client.send_omni_request(request_config, request_num=1)
    has_audio = any(r.audio_bytes is not None and len(r.audio_bytes) > 0 for r in responses)
    assert has_audio, "Expected audio output from in-process token2wav vocoder path"

@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_output_duration_reasonable(omni_server, openai_client) -> None:
    """
    Verify that the generated audio has reasonable duration
    (not too short, not absurdly long for a short prompt).
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="Say hello world briefly.",
    )
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"audio": ["hello"]},
    }
    responses = openai_client.send_omni_request(request_config, request_num=1)
    for r in responses:
        if r.audio_bytes is not None and len(r.audio_bytes) > 0:
            # Very rough sanity: audio should be between 100ms and 30s for a short prompt
            assert 100 < len(r.audio_bytes) < 500000, (
                f"Audio duration {len(r.audio_bytes)} bytes outside reasonable range"
            )
            break

@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_chinese_text_to_audio(omni_server, openai_client) -> None:
    """
    Test Chinese text input generating audio output.
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="�������ļ򵥽���һ�±�����",
    )
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"text": ["����"]},
    }
    openai_client.send_omni_request(request_config, request_num=1)

@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_english_text_to_audio(omni_server, openai_client) -> None:
    """
    Test English text input generating audio output.
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="Please briefly introduce London.",
    )
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"text": ["London"]},
    }
    openai_client.send_omni_request(request_config, request_num=1)

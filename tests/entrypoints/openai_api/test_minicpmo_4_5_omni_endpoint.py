# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Copyright contributors to the vLLM project
"""
``POST /v1/completions`` availability for MiniCPM-o 4.5.

MiniCPM-o 4.5 does NOT block ``/v1/completions`` (unlike Qwen3-Omni which has
``endpoint_restrictions``). Verify that the completions endpoint is accessible.

Migrated from ``tests/e2e/online_serving/test_minicpmo_4_5.py`` to colocate with
other OpenAI-compatible entrypoint tests under ``tests/entrypoints``.
"""

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams, OpenAIClientHandler
from tests.helpers.stage_config import get_deploy_config_path

_MINICPMO_DEPLOY = get_deploy_config_path("ci/minicpmo_4_5.yaml")

_MINICPMO_SERVER = [
    pytest.param(
        OmniServerParams(
            model="openbmb/MiniCPM-o-4_5",
            stage_config_path=_MINICPMO_DEPLOY,
            use_stage_cli=True,
            server_args=[
                "--trust-remote-code",
                "--no-async-chunk",
            ],
        ),
        id="minicpmo_4_5",
    )
]


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _MINICPMO_SERVER, indirect=True)
def test_completions_endpoint_available(omni_server: OmniServerParams, openai_client: OpenAIClientHandler) -> None:
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

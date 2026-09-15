"""Draft the Thinker's terminator pair once the prompt copy runs out.

On the ranked simplex path the Thinker's answer is a verbatim copy of the
user's text followed by two terminators: ``<|tts_eos|>`` then ``<|im_end|>``.
The n-gram drafter copies its proposal out of the prompt, and the prompt
carries ``<|im_end|>`` right after the user text (the chat template's own
closer) but never ``<|tts_eos|>`` -- so the draft is structurally wrong at the
first terminator, every request. Measured on A3 / 910C (2026-08-29, d8s1 tree,
step probe): the tail then costs one or two *eager* 1-token steps at ~37 ms
each -- there is no captured graph for a no-draft step, the uniform decode
shape is 16 query tokens -- while a 16-token verify replay is ~7 ms of host.

Two rewrites close that, and both are exact by the rejection sampler's own
rule (a draft token is only ever emitted if the model's argmax equals it, so
the emitted text cannot change):

* A draft that runs past the copy into the template's ``<|im_end|>`` is
  rewritten at that point to ``<|tts_eos|>, <|im_end|>`` -- the tail the model
  actually produces -- and padded back to full width so the step keeps the
  captured 16-token shape.
* An *empty* draft right after the model emitted ``<|tts_eos|>`` (the n-gram
  lookup cannot match a token the prompt does not contain) becomes a full-width
  ``<|im_end|>`` draft: the terminator rides a graph replay instead of an eager
  single-token step.

Padding is ``<|im_end|>`` repeated: tokens past an accepted stop are dropped by
the stop checker, and a mismatched pad is rejected outright, so the padding can
never reach the output either way.

``VLLM_OMNI_MINICPMO_STAGE0_TAIL_DRAFT=off`` restores the stock drafts.

``=strict`` keeps only the second rewrite, and is **off by default**. The first
one is a guess: it fires on any ``<|im_end|>`` in the draft, the chat template
closes every turn, and a prompt-lookup draft that matched elsewhere carries a
closer too -- so on free-form generation it proposes "end the utterance" as the
next token, every step. That is harmless only while verification is exact, and
under this pipeline it is not.
"""

from __future__ import annotations

import os

_ENV = "VLLM_OMNI_MINICPMO_STAGE0_TAIL_DRAFT"
_OFF = frozenset({"0", "off", "false", "no"})
# ``strict`` keeps only the rewrite that needs no guess about the future. See
# the module docstring for what the other one costs and what it buys.
_STRICT = frozenset({"strict", "verified", "safe"})

# MiniCPM-o 4.5 tokenizer: the TTS terminator pair the answer always ends with.
_TTS_EOS = 151704
_IM_END = 151645


def enabled() -> bool:
    return os.environ.get(_ENV, "").strip().lower() not in _OFF


def strict() -> bool:
    """Only draft the closer the model has already asked for."""
    return os.environ.get(_ENV, "").strip().lower() in _STRICT


def applies(runner) -> bool:
    """Thinker stage of the MiniCPM-o pipeline: CPU n-gram drafts, 15 wide."""
    if not enabled():
        return False
    sc = getattr(runner, "speculative_config", None)
    if sc is None or getattr(sc, "method", None) != "ngram":
        return False
    # The Talker's constant drafter never reaches this path (frames-1 == 3);
    # 15 is the Thinker's width and doubles as the pipeline check.
    return getattr(sc, "num_speculative_tokens", None) == 15


def rewrite(
    drafts: list[list[int]] | None,
    sampled_token_ids: list[list[int]],
    k: int,
) -> list[list[int]] | None:
    if not isinstance(drafts, list):
        return drafts
    only_verified = strict()
    for i, d in enumerate(drafts):
        if not isinstance(d, list):
            continue
        if d:
            if only_verified:
                continue
            try:
                j = d.index(_IM_END)
            except ValueError:
                continue
            d[j:] = [_TTS_EOS, _IM_END]
            del d[k:]
        else:
            sampled = sampled_token_ids[i] if i < len(sampled_token_ids) else None
            if not sampled or sampled[-1] != _TTS_EOS:
                continue
            d = drafts[i] = []
        d += [_IM_END] * (k - len(d))
    return drafts

# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from functools import partial
from typing import Any, Optional

import torch
from torch.nn.attention.flex_attention import _score_mod_signature, and_masks

from llmfoundry.layers_registry import flex_attention_mods


class FlexAttentionMod(ABC):

    def _mask_mod_fn(
        self,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        sequence_id_info: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        del sequence_id_info, b, h, q_idx, kv_idx
        raise NotImplementedError

    def _score_mod_fn(
        self,
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        sequence_id_info: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        del sequence_id_info, score, b, h, q_idx, kv_idx
        raise NotImplementedError

    def __init__(self, mod_type: str) -> None:
        assert mod_type in ['mask', 'score']
        self.mod_type = mod_type
        self.mod_fn = self._mask_mod_fn if mod_type == 'mask' else self._score_mod_fn


@flex_attention_mods.register('causal_mask')
class CausalMaskMod(FlexAttentionMod):

    def _mask_mod_fn(
        self,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        sequence_id_info: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        del sequence_id_info, b, h
        return q_idx >= kv_idx

    def __init__(self) -> None:
        super().__init__(mod_type='mask')


@flex_attention_mods.register('sliding_window_mask')
class SlidingWindowMaskMod(FlexAttentionMod):

    def _mask_mod_fn(
        self,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        sequence_id_info: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        del sequence_id_info, b, h
        return q_idx - kv_idx <= self.sliding_window_size

    def __init__(self, sliding_window_size: int) -> None:
        super().__init__(mod_type='mask')
        self.sliding_window_size = sliding_window_size


@flex_attention_mods.register('sequence_id_mask')
class SequenceIdMaskMod(FlexAttentionMod):

    def _mask_mod_fn(
        self,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        sequence_id_info: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        del h
        if sequence_id_info is None:
            raise ValueError(
                'sequence_id_info is required for SequenceIdMaskMod',
            )
        sequence_id = sequence_id_info['sequence_id']
        # Check if the query and key belong to the same sequence and the query token is not a padding token.
        return (sequence_id[b, q_idx]
                == sequence_id[b, kv_idx]) & (sequence_id[b, q_idx] != -1)

    def __init__(self) -> None:
        super().__init__(mod_type='mask')


@flex_attention_mods.register('local_global_mask')
class LocalGlobalMaskMod(FlexAttentionMod):

    def _mask_mod_fn(
        self,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        sequence_id_info: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        del h
        if sequence_id_info is None:
            raise ValueError(
                'sequence_id_info is required for LocalGlobalMaskMod',
            )
        sequence_id = sequence_id_info['sequence_id']
        pos_in_seq = sequence_id_info['pos_in_seq']
        # Check if the query and key belong to the same sequence and the query token is not a padding token.

        sequence_id_mask = (sequence_id[b, q_idx] == sequence_id[b, kv_idx]
                           ) & (sequence_id[b, q_idx] != -1)
        global_window_mask = (pos_in_seq[b, kv_idx] <= self.global_window_size)
        sliding_window_mask = (q_idx - kv_idx <= self.sliding_window_size)

        return sequence_id_mask & (global_window_mask | sliding_window_mask)

    def __init__(
        self,
        sliding_window_size: int,
        global_window_size: int,
    ) -> None:
        super().__init__(mod_type='mask')
        self.sliding_window_size = sliding_window_size
        self.global_window_size = global_window_size


@flex_attention_mods.register('alibi_score_mod')
class AlibiScoreMod(FlexAttentionMod):

    def _score_mod_fn(
        self,
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        sequence_id_info: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        del sequence_id_info, b
        bias = -self.alibi_slopes[h] * (q_idx - kv_idx)
        return score + bias

    def __init__(self, alibi_slopes: torch.Tensor) -> None:
        super().__init__(mod_type='score')
        self.alibi_slopes = alibi_slopes


@flex_attention_mods.register('softcap_score_mod')
class SoftcapScoreMod(FlexAttentionMod):

    def _score_mod_fn(
        self,
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        sequence_id_info: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        del sequence_id_info, b, h, q_idx, kv_idx
        return self.attn_logit_softcapping * torch.tanh(
            score / self.attn_logit_softcapping,
        )

    def __init__(self, attn_logit_softcapping: int) -> None:
        super().__init__(mod_type='score')
        self.attn_logit_softcapping = attn_logit_softcapping


def generate_block_mask(
    Q_LEN: int,
    KV_LEN: int,
    B: int,
    block_mask_list: Optional[list[FlexAttentionMod]],
    compiled_create_block_mask: Any,
    sequence_id_info: Optional[dict[str, Any]],
):
    if block_mask_list is None:
        return None

    block_mask_fn = None
    for i, block_mask in enumerate(block_mask_list):
        if i == 0:
            block_mask_fn = partial(
                block_mask.mod_fn,
                sequence_id_info=sequence_id_info,
            )
        else:
            block_mask_fn = and_masks(
                block_mask_fn,
                partial(block_mask.mod_fn, sequence_id_info=sequence_id_info),
            )

    block_mask = compiled_create_block_mask(
        block_mask_fn,
        B=B,
        H=None, # Setting this to None speeds up block mask generation, but this means the mask has to be the same across all heads.
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
    )

    return block_mask


def generate_score_mod(
    score_mod_list: Optional[list[FlexAttentionMod]],
    sequence_id_info: Optional[dict[str, Any]],
):
    if score_mod_list is None:
        return None
    wrapped_score_mod = None
    for i, score_mod in enumerate(score_mod_list):
        if i == 0:
            wrapped_score_mod = partial(
                score_mod.mod_fn,
                sequence_id_info=sequence_id_info,
            )
        else:
            wrapped_score_mod = _wrap_score_mod_fns(
                wrapped_score_mod,
                partial(score_mod.mod_fn, sequence_id_info=sequence_id_info),
            )

    return wrapped_score_mod


def _wrap_score_mod_fns(
    score_mod_fn_1: _score_mod_signature,
    score_mod_fn_2: _score_mod_signature,
) -> _score_mod_signature:

    def wrapped_score_mod_fn(
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        score = score_mod_fn_1(score, b, h, q_idx, kv_idx)
        score = score_mod_fn_2(score, b, h, q_idx, kv_idx)
        return score

    return wrapped_score_mod_fn
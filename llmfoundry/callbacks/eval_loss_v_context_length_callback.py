# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch
import wandb
from composer.core import Callback, State
from composer.loggers import Logger


class LossVsContextLengthEvaluator(Callback):

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        perplexity = state.eval_metrics['eval'][
            'LanguagePerplexityNoReduce'].compute()
        seq_id = state.batch['sequence_id']
        seq_id_expanded = (torch.arange(seq_id.shape[-1]).repeat(
            seq_id.shape[-1],
            1).transpose(0, 1).to(seq_id) == seq_id.unsqueeze(-2))
        seq_id_expanded = (seq_id_expanded.cumsum(dim=-1) - 1) * seq_id_expanded
        tok_ids = seq_id_expanded.sum(dim=-2)
        tok_ids_expanded = (torch.arange(tok_ids.shape[-1]).repeat(
            tok_ids.shape[-1],
            1).transpose(0, 1).to(tok_ids) == tok_ids.unsqueeze(-2))
        perplexity = perplexity.view(tok_ids_expanded.shape[:-1]).unsqueeze(-2)
        sum_perp = state.eval_metrics['eval'][
            'LanguagePerplexityNoReduce'].sum_perp + torch.where(
                tok_ids_expanded, perplexity, 0).sum(dim=-1).sum(dim=0)
        sum_length = state.eval_metrics['eval'][
            'LanguagePerplexityNoReduce'].sum_length + tok_ids_expanded.sum(
                dim=-1).sum(dim=0)
        state.eval_metrics['eval']['LanguagePerplexityNoReduce'].reset()
        state.eval_metrics['eval'][
            'LanguagePerplexityNoReduce'].sum_perp = sum_perp
        state.eval_metrics['eval'][
            'LanguagePerplexityNoReduce'].sum_length = sum_length
        # print(sum_length)
        # print(sum_perp / sum_length)
        sum_perp = torch.where(sum_length!=0, sum_perp, -1)
        sum_length = torch.where(sum_length!=0, sum_length, 1)
        avg_perp = sum_perp / sum_length
        scatter_lcp = wandb.Table(data=[[i, b] for (i, b) in enumerate(avg_perp.tolist())], columns = ["seq_length", "perplexity"])
        wandb.log({f"long_context_perplexity": wandb.plot.scatter(scatter_lcp, "seq_length", "perplexity", title=f"long_context_perplexity")}, step=state.timestamp.batch.value)

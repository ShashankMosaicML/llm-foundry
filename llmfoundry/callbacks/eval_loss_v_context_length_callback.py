# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch
import wandb
from composer.core import Callback, State
from composer.loggers import Logger


class LossVsContextLengthEvaluator(Callback):

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        sum_perp, sum_length = state.eval_metrics['eval']['LanguagePerplexityNoReduce'].compute()
        sum_perp = torch.where(sum_length!=0, sum_perp, -1)
        sum_length = torch.where(sum_length!=0, sum_length, 1)
        avg_perp = sum_perp / sum_length
        scatter_lcp = wandb.Table(data=[[i, b] for (i, b) in enumerate(avg_perp.tolist())], columns = ["seq_length", "perplexity"])
        wandb.log({f"long_context_perplexity": wandb.plot.scatter(scatter_lcp, "seq_length", "perplexity", title=f"long_context_perplexity")}, step=state.timestamp.batch.value)

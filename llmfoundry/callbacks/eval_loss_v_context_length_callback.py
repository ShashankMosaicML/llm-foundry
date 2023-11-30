# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch
import wandb
from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist


class LossVsContextLengthEvaluator(Callback):

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        sum_perp, sum_length = state.eval_metrics['eval']['LanguagePerplexityNoReduce'].compute()
        sum_perp = torch.where(sum_length != 0, sum_perp, -1)
        sum_length = torch.where(sum_length != 0, sum_length, 1)
        avg_perp = sum_perp / sum_length
        scatter_lcp = wandb.Table(
            data=[[i, b] for (i, b) in enumerate(avg_perp.tolist())],
            columns=['seq_length', 'loss'])
        if dist.get_global_rank() == 0:
            wandb.log(
                {
                    f'long_context_loss':
                        wandb.plot.scatter(scatter_lcp,
                                           'seq_length',
                                           'loss',
                                           title=f'long_context_loss')
                },
                commit=True)

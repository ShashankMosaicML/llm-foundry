# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

PROMPTFILE_PREFIX = 'file::'

__all__ = [
    'load_prompts',
    'load_prompts_from_file',
]


def load_prompts(prompts: list[str],
                 prompt_delimiter: Optional[str] = None) -> list[str]:
    """Loads a set of prompts, both free text and from file.

    Args:
        prompts (List[str]): List of free text prompts and prompt files
        prompt_delimiter (Optional str): Delimiter for text file
            If not provided, assumes the prompt file is a single prompt (non-delimited)

    Returns:
        List of prompt string(s)
    """
    prompt_strings = []
    for prompt in prompts:
        if prompt.startswith(PROMPTFILE_PREFIX):
            prompts = load_prompts_from_file(prompt, prompt_delimiter)
            prompt_strings.extend(prompts)
        else:
            prompt_strings.append(prompt)
    return prompt_strings


def load_prompts_from_file(
    prompt_path: str,
    prompt_delimiter: Optional[str] = None,
) -> list[str]:
    """Load a set of prompts from a text fie.

    Args:
        prompt_path (str): Path for text file
        prompt_delimiter (Optional str): Delimiter for text file
            If not provided, assumes the prompt file is a single prompt (non-delimited)

    Returns:
        List of prompt string(s)
    """
    if not prompt_path.startswith(PROMPTFILE_PREFIX):
        raise ValueError(f'prompt_path_str must start with {PROMPTFILE_PREFIX}')

    _, prompt_file_path = prompt_path.split(PROMPTFILE_PREFIX, maxsplit=1)
    prompt_file_path = os.path.expanduser(prompt_file_path)
    if not os.path.isfile(prompt_file_path):
        raise FileNotFoundError(
            f'{prompt_file_path=} does not match any existing files.',
        )

    with open(prompt_file_path, 'r') as f:
        prompt_string = f.read()

    if prompt_delimiter is None:
        return [prompt_string]
    return [i for i in prompt_string.split(prompt_delimiter) if i]

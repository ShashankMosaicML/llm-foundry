# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import transformers
from transformers import PreTrainedTokenizerBase

from llmfoundry.data.finetuning.tasks import (
    _DEFAULT_CHAT_TEMPLATE,
    _slice_chat_formatted_example,
    dataset_constructor,
    tokenize_formatted_example,
)
from llmfoundry.tokenizers import get_date_string
from llmfoundry.utils.exceptions import (
    ALLOWED_PROMPT_KEYS,
    ALLOWED_RESPONSE_KEYS,
    ChatTemplateError,
    InvalidExampleTypeError,
    InvalidMessageTypeError,
)


def test_tokenize_chat_example_malformed(
    tiny_mpt_chat_tokenizer: PreTrainedTokenizerBase,
):
    no_content = {'messages': [{'role': 'user'}]}
    too_few_messages = {
        'messages': [{
            'role': 'assistant',
            'content': 'Hi, User!',
        }],
    }
    ends_with_user_role = {
        'messages': [{
            'role': 'user',
            'content': 'Hello GPT!',
        }, {
            'role': 'assistant',
            'content': 'Hi, User!',
        }, {
            'role': 'user',
            'content': 'user message not followed by an assistant label',
        }],
    }
    no_assistant_message = {
        'messages': [{
            'role': 'user',
            'content': 'Hello GPT!',
        }, {
            'role': 'user',
            'content': 'user message not followed by an assistant label',
        }],
    }
    wrong_example_type = ['this is not a dictionary']
    wrong_messages_type = {'messages': 'this is not a list of messages'}
    wrong_role = {
        'messages': [{
            'role': 'user',
            'content': 'Hello GPT!',
        }, {
            'role': 'misnamed_assistant',
            'content': 'user message not followed by an assistant label',
        }],
    }
    malformed_chat_examples = [
        too_few_messages,
        no_content,
        ends_with_user_role,
        no_assistant_message,
        wrong_role,
    ]
    my_tokenizer = tiny_mpt_chat_tokenizer
    for example in malformed_chat_examples:
        with pytest.raises(Exception):
            tokenize_formatted_example(
                example,
                my_tokenizer,
            )  # type: ignore (the typing here is supposed to be malformed)
    with pytest.raises(InvalidExampleTypeError):
        # Ignore the type here because it's the mistyping that we're
        # trying to test.
        tokenize_formatted_example( # type: ignore
            wrong_example_type, # type: ignore
            my_tokenizer, # type: ignore
        )
    with pytest.raises(InvalidMessageTypeError):
        tokenize_formatted_example(
            wrong_messages_type,
            my_tokenizer,
        )


def test_tokenize_chat_example_well_formed(
    tiny_mpt_chat_tokenizer: PreTrainedTokenizerBase,
):
    chat_examples = [
        {
            'messages': [{
                'role': 'user',
                'content': 'Hello, GPT',
            }, {
                'role': 'assistant',
                'content': 'this is my response',
            }],
        },  # prompt/response but in chat format
        {
            'messages': [
                {
                    'role': 'user',
                    'content': 'Hello, GPT',
                },
                {
                    'role': 'assistant',
                    'content': 'this is my response',
                },
                {
                    'role': 'user',
                    'content': 'Nice to hear that.',
                },
                {
                    'role': 'assistant',
                    'content': 'multi-way chat works too!',
                },
            ],
        },  # multi-way chat
    ]

    expected = [
        [{
            'prompt':
                '''<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.
<|im_start|>user
Hello, GPT<|im_end|>
<|im_start|>assistant
''',
            'response':
                'this is my response<|im_end|>',
        }],
        [{
            'prompt':
                '''<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.
<|im_start|>user
Hello, GPT<|im_end|>
<|im_start|>assistant
''',
            'response':
                'this is my response<|im_end|>',
        }, {
            'prompt':
                '''
<|im_start|>user
Nice to hear that.<|im_end|>
<|im_start|>assistant
''',
            'response':
                'multi-way chat works too!<|im_end|>',
        }],
    ]

    chat_tokenizer = tiny_mpt_chat_tokenizer
    assert len(expected) == len(
        chat_examples,
    )  # if we add a new example, zip shouldn't fail silently
    for chat_example, expected_stringification in zip(chat_examples, expected):
        templatized_prompt_response_turns = _slice_chat_formatted_example(
            chat_example,
            chat_tokenizer,
        )
        tokenized_example = tokenize_formatted_example(
            chat_example,
            chat_tokenizer,
        )
        for (prompt, response), exp_str, turn in zip(
            templatized_prompt_response_turns,
            expected_stringification,
            tokenized_example['turns'],
        ):
            assert prompt == exp_str['prompt']
            assert response == exp_str['response']
            assert 'input_ids' in turn
            assert 'labels' in turn


def test_tokenize_instruct_example_malformed():
    no_keys = {}
    no_prompt_key = {'response': 'response'}
    no_response_key = {'prompt': 'prompt'}
    extra_keys_with_prompt = {'prompt': 'prompt', 'extra': 'extra'}
    extra_keys_with_response = {'response': 'response', 'extra': 'extra'}
    multiple_allowed_response_keys = {
        'prompt': 'prompt',
        'response': 'response',
        'completion': 'completion',
    }

    malformed_prompt_response_examples = [
        no_keys,
        no_prompt_key,
        no_response_key,
        extra_keys_with_prompt,
        extra_keys_with_response,
        multiple_allowed_response_keys,
    ]

    for example in malformed_prompt_response_examples:
        with pytest.raises(Exception):
            tokenize_formatted_example(example, MagicMock())


def test_tokenize_instruct_example_well_formed(
    tiny_gpt2_tokenizer: PreTrainedTokenizerBase,
):
    tokenizer = tiny_gpt2_tokenizer

    for prompt_key in ALLOWED_PROMPT_KEYS:
        for response_key in ALLOWED_RESPONSE_KEYS:

            example = {prompt_key: 'prompt', response_key: 'response'}
            tokenized_example = tokenize_formatted_example(example, tokenizer)
            assert 'input_ids' in tokenized_example['turns'][0]
            assert 'labels' in tokenized_example['turns'][0]


@pytest.mark.parametrize(
    'tokenizer_name',
    [
        'EleutherAI/gpt-neox-20b',
        'HuggingFaceH4/zephyr-7b-beta',
        't5-base',
        'meta-llama/Meta-Llama-3.1-8B-Instruct',
    ],
)
@pytest.mark.parametrize('messages_format', [True, False])
@pytest.mark.parametrize('use_date_string', [True, False])
def test_multi_turn_chat_slicing(
    tokenizer_name: str,
    messages_format: bool,
    use_date_string: bool,
):
    if 'meta-llama' in tokenizer_name:
        pytest.skip('Model is gated. Skipping test.')
    is_llama_3_1_instruct = 'Meta-Llama-3.1' in tokenizer_name and 'Instruct' in tokenizer_name
    if is_llama_3_1_instruct and use_date_string:
        pytest.skip(
            'Llama 3.1 Instruct models use date_string in chat template already. Skipping test.',
        )
    if messages_format:
        convo = [
            {
                'role': 'system',
                'content': 'everyone thinks you are so cool',
            },
            {
                'role': 'user',
                'content': 'hiiii',
            },
            {
                'role': 'assistant',
                'content': 'yassss',
            },
            {
                'role': 'user',
                'content': 'HIIIIII!!!',
            },
            {
                'role': 'assistant',
                'content': 'YASSSSSS',
            },
        ]
    else:
        convo = [
            {
                'from': 'system',
                'value': 'everyone thinks you are so cool',
            },
            {
                'from': 'human',
                'value': 'hiiii',
            },
            {
                'from': 'gpt',
                'value': 'yassss',
            },
            {
                'from': 'tool',
                'value': 'HIIIIII!!!',
            },
            {
                'from': 'gpt',
                'value': 'YASSSSSS',
            },
        ]
        tmp = {'conversations': convo}
        preprocessor = dataset_constructor.get_preprocessing_fn_from_str(
            'teknium/OpenHermes-2.5',
        )
        assert preprocessor is not None
        convo = preprocessor(tmp)['messages']
        assert isinstance(convo, list)

    example = {'messages': convo}

    tok = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

    # Manually set a chat template to test if the date_string is being used.
    if use_date_string:
        tok.chat_template = "{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{{- \"Today Date: \" + date_string }}\n"

    if not tok.chat_template:
        tok.chat_template = _DEFAULT_CHAT_TEMPLATE

    templated_prompt_response_turns = _slice_chat_formatted_example(
        example,
        tok,
    )

    reconstructed_chat = ''
    for prompt, response in templated_prompt_response_turns:
        reconstructed_chat += prompt + response

    date_string = get_date_string()
    full_chat = tok.apply_chat_template(
        convo,
        tokenize=False,
        date_string=date_string,
    )
    assert reconstructed_chat == full_chat

    if is_llama_3_1_instruct or use_date_string:
        assert date_string in full_chat
    else:
        assert date_string not in full_chat


def test_fail_chat_template():
    convo = [
        {
            'role':
                'system',  # this will fail because the tokenizer doesn't have a system role
            'content': 'everyone thinks you are so cool',
        },
        {
            'role': 'user',
            'content': 'hiiii',
        },
        {
            'role': 'assistant',
            'content': 'yassss',
        },
    ]

    example = {'messages': convo}

    class DummyTokenizer:

        def __init__(self) -> None:
            self.chat_template = 'Hello, World!'

        def apply_chat_template(self, **_):
            raise ValueError('This tokenizer does not support the system role')

    tok = DummyTokenizer()

    with pytest.raises(ChatTemplateError):
        _slice_chat_formatted_example(example, tok)  # type: ignore


def test_tokenize_no_labels_bos_pr():
    # This tokenizer automatically adds bos tokens
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'ai21labs/Jamba-v0.1',
        add_bos_token=True,
    )

    example = {'prompt': 'prompt', 'response': 'response'}

    assert tokenizer.add_bos_token == True

    tokenized_example = tokenize_formatted_example(example, tokenizer)

    # Extract the first turn
    tokenized_example = tokenized_example['turns'][0]

    assert len(tokenized_example['labels']) == 1
    assert tokenized_example['labels'][0] != tokenizer.bos_token_id
    assert tokenized_example['input_ids'][0] == tokenizer.bos_token_id

    # This tokenizer does not have the add_bos_token attribute
    tokenizer = transformers.AutoTokenizer.from_pretrained('mosaicml/mpt-7b')

    assert not tokenizer.add_bos_token

    tokenized_example = tokenize_formatted_example(example, tokenizer)

    # Extract the first turn
    tokenized_example = tokenized_example['turns'][0]

    assert len(tokenized_example['labels']) == 1
    assert tokenized_example['labels'][0] != tokenizer.bos_token_id
    assert tokenized_example['input_ids'][0] != tokenizer.bos_token_id

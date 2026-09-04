# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import MethodType, SimpleNamespace

import pytest
import torch

from nemo.collections.tts.models.easy_magpietts_preference_optimization import EasyMagpieTTSModelOnlinePO


pytestmark = pytest.mark.unit


def test_unstack_rollout_phoneme_tokens_uses_per_item_spans():
    model = SimpleNamespace(
        phoneme_tokenizer=SimpleNamespace(
            bos_token_id=100,
            eos_token_id=101,
        )
    )
    output = SimpleNamespace(
        predicted_phoneme_tokens=torch.tensor(
            [
                [[1, 2, 3, 4], [11, 12, 13, 14]],
                [[5, 6, 7, 8], [15, 16, 17, 18]],
            ]
        ),
        predicted_phoneme_tokens_lens=torch.tensor([2, 3]),
        phoneme_prediction_start_idx=torch.tensor([1, 0]),
    )

    tokens, token_lens = EasyMagpieTTSModelOnlinePO._unstack_rollout_phoneme_tokens(model, output)

    assert token_lens.tolist() == [5, 7]
    assert tokens[0, :5].tolist() == [100, 2, 12, 3, 13]
    assert tokens[1, :7].tolist() == [100, 5, 15, 6, 16, 7, 17]


def test_action_po_components_backpropagates_advantage_to_action_logits():
    model = SimpleNamespace(
        reference_free=True,
        loss_type='grpo',
        max_decoder_steps=4,
        cfg={},
    )
    model._get_per_token_logps = MethodType(EasyMagpieTTSModelOnlinePO._get_per_token_logps, model)

    logits = torch.zeros(2, 2, 3, requires_grad=True)
    targets = torch.tensor([[[0, 1]], [[1, 0]]])
    target_lens = torch.tensor([2, 2])
    advantages = torch.tensor([1.0, -1.0])
    validities = torch.ones(2)

    po_loss, kl_loss, entropy = EasyMagpieTTSModelOnlinePO._compute_action_po_components(
        model,
        logits=logits,
        reference_logits=None,
        targets=targets,
        target_lens=target_lens,
        vocab_size=3,
        advantages=advantages,
        group_validities=validities,
    )
    po_loss.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) > 0
    assert kl_loss.item() == 0.0
    assert entropy.item() == pytest.approx(torch.log(torch.tensor(3.0)).item())


def test_sampling_transform_matches_temperature_and_topk_policy():
    logits = torch.tensor([[[4.0, 2.0, 1.0, -1.0]]])

    transformed = EasyMagpieTTSModelOnlinePO._apply_sampling_transform(logits, temperature=0.5, topk=2)

    assert transformed[0, 0, :2].tolist() == [8.0, 4.0]
    assert torch.isneginf(transformed[0, 0, 2:]).all()


def test_action_po_entropy_is_finite_with_topk_masking():
    model = SimpleNamespace(
        reference_free=True,
        loss_type='grpo',
        max_decoder_steps=2,
        cfg={},
    )
    model._get_per_token_logps = MethodType(EasyMagpieTTSModelOnlinePO._get_per_token_logps, model)
    model._apply_sampling_transform = EasyMagpieTTSModelOnlinePO._apply_sampling_transform

    _, _, entropy = EasyMagpieTTSModelOnlinePO._compute_action_po_components(
        model,
        logits=torch.tensor([[[4.0, 2.0, 1.0, -1.0]]], requires_grad=True),
        reference_logits=None,
        targets=torch.tensor([[[0]]]),
        target_lens=torch.tensor([1]),
        vocab_size=4,
        advantages=torch.tensor([1.0]),
        group_validities=torch.ones(1),
        sampling_temperature=0.5,
        sampling_topk=2,
    )

    assert torch.isfinite(entropy)

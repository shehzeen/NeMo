# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import re

import torch
import torch.utils.data
import torchaudio
import random

from lhotse import CutSet, MonoCut, Recording, Seconds, SupervisionSegment, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import ifnone

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.utils import logging
from nemo.collections.speechlm2.modules.ear_tts_commons import SCRIPT_PLACEHOLDER


def get_mask_from_lengths(
    lengths: torch.Tensor = None,
    x: torch.Tensor = None,
) -> torch.Tensor:
    """Constructs binary mask from a 1D torch tensor of input lengths
    Args:
        lengths: torch.tensor (torch.tensor): 1D tensor with lengths
        x: torch.tensor = tensor to be used on, last dimension is for mask
    Returns:
        mask (torch.tensor): num_sequences x max_length binary tensor
    """
    if lengths is None:
        assert x is not None
        return torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
    else:
        if x is None:
            max_len = torch.max(lengths)
        else:
            max_len = x.shape[-1]

    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = ids < lengths.unsqueeze(1)
    return mask


class DuplexEARTTSDataset(torch.utils.data.Dataset):
    """
    A dataset for duplex speech-to-speech models that handles bidirectional conversations.

    This dataset processes Lhotse CutSet objects containing recordings with supervision segments
    from different speakers (roles). It creates aligned representations of audio and text for
    both source (input) and target (output) channels, preserving temporal alignment between
    audio frames and text tokens.

    Args:
        tokenizer (TokenizerSpec):
            Tokenizer for converting text to token IDs and vice versa. Must support BOS and EOS tokens.
            It's expected to support PAD token as well, otherwise we will use 0 as the pad token
            and emit a warning.

        frame_length (Seconds):
            Duration of a single frame in seconds. Used to calculate frame positions for token alignment.

        source_sample_rate (int):
            Sample rate for source audio (e.g., 16000 Hz).

        target_sample_rate (int):
            Sample rate for target audio (e.g., 22050 Hz).

        input_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as inputs. Defaults to ["user"].

        output_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as outputs. Defaults to ["agent"].

    Returns:
        A dictionary with the following keys:
            - source_audio: Tensor of source waveform samples [B, T]
            - source_audio_lens: Tensor of source audio lengths [B]
            - target_audio: Tensor of target waveform samples [B, T]
            - target_audio_lens: Tensor of target audio lengths [B]
            - input_text_tokens: Tensor of target text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - target_token_lens: Tensor of target token sequence lengths [B]
            - source_tokens: Tensor of source text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - source_token_lens: Tensor of source token sequence lengths [B]
            - target_texts: List of full target texts joined from output_roles supervisions [B]

    Notes:
        - The dataset ensures frame-level alignment between audio and text by inserting tokens at
          specific frame positions based on the timing of supervision segments.
        - PAD tokens (typically 0) are used to fill gaps where there's no text.
        - BOS tokens mark the beginning of each speech segment.
        - EOS tokens mark the end of each speech segment.
        - Text tokens from each speaker are placed at frame positions corresponding to their
          timestamp in the original recording, preserving the temporal relationship.
          This is a segment-level alignment only, not word-level alignment.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        source_sample_rate: int,
        target_sample_rate: int,
        input_roles: list[str] = None,
        output_roles: list[str] = None,
        add_description: bool = True,
        p_drop_description: float = 0.1,
    ):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        self.add_description = add_description
        self.p_drop_description = p_drop_description
        
        assert tokenizer.bos is not None, "BOS support in the tokenizer is required for S2S models."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required for S2S models."

    def generate_prompt_description(self, device):
        messages = []
        if random.random() > self.p_drop_description:
            # ToDo: add extra system prompts
            system_prompt = (
                "You engage in conversation with the user. When delivering your response as speech, "
                "if the user provides a description such as emotions, scene details, "
                "or speaker style, you adjust your speaking style accordingly when delivering the response. "
                "However, this description should influence only the delivery of your response, not its content. "
                "Your response should remain independent of any stylistic instructions."
            )
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": ""})
        
        # ToDo: implement dataloading support for descriptions
        """for desc in example["descriptions"]:
            user_prompt = ""
            if random.random() > self.p_drop_description and desc:
                user_prompt += f"```\n{desc}\n```"
            if random.random() > self.p_drop_description:
                if user_prompt:
                    user_prompt += "\n\n"
                user_prompt += self.rng.choice(self.user_prompts)
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})
            messages.append({"role": "assistant", "content": SCRIPT_PLACEHOLDER})
        """

        # given that descriptions are currently not supported, only added the user prompt
        # ToDo: add extra user prompts
        user_prompt = "Can you tell me something interesting?"
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": SCRIPT_PLACEHOLDER})
        non_script_list = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ).split(SCRIPT_PLACEHOLDER + self.tokenizer.eos_token)[:-1]

        input_ids = []
        for i, non_script in enumerate(non_script_list):
            desc_ids = self.tokenizer.text_to_ids(non_script)
            input_ids.extend(desc_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).view(1, -1)
        return input_ids

    def __getitem__(self, cuts: CutSet) -> dict:
        cuts = cuts.transform_text(_strip_timestamps)
        source_audio, source_audio_lens = collate_audio(cuts.resample(self.source_sample_rate))
        target_audio, target_audio_lens = collate_audio(
            cuts.resample(self.target_sample_rate), recording_field="target_audio"
        )
        input_text_tokens, target_token_lens = collate_token_channel(
            cuts, self.tokenizer, self.frame_length, roles=self.output_roles
        )
        source_tokens, source_token_lens = collate_token_channel(
            cuts, self.tokenizer, self.frame_length, roles=self.input_roles
        )

        # extract target speaker first turn audio to uses for speaker conditioning
        target_first_turn_audio, target_first_turn_audio_lens = collate_first_turn_audio(
            cuts.resample(self.target_sample_rate), roles=self.output_roles, recording_field="target_audio"
        )

        # ensures that input_text_tokens is not longer than its duration
        input_text_tokens = input_text_tokens[:, :target_token_lens.max()]

        if self.add_description:
            text_pad_id = get_pad_id(self.tokenizer)
            source_fps = self.source_sample_rate / (
                self.source_sample_rate * self.frame_length
            )
            source_samples_per_frame = int(self.source_sample_rate//source_fps)
            target_fps = self.target_sample_rate / (
                self.target_sample_rate * self.frame_length
            )
            target_samples_per_frame = int(self.target_sample_rate//target_fps)
            input_text_tokens_ = []
            source_tokens_ = []
            source_audio_ = []
            target_audio_ = []
            desc_lens = []
            # for each sample in the batch
            for i in range(input_text_tokens.size(0)):
                desc_tokens_ids = self.generate_prompt_description(device=input_text_tokens[i].device).squeeze(0)
                # add description to target text tokens
                input_text_tokens_.append(torch.cat([desc_tokens_ids, input_text_tokens[i]]))
                target_token_lens[i] = target_token_lens[i] + len(desc_tokens_ids)
                # add description to source text tokens
                source_tokens_.append(torch.cat([desc_tokens_ids, source_tokens[i]]))
                source_token_lens[i] = source_token_lens[i] + len(desc_tokens_ids)
                # add silence in the source audio while the prompt is being processed
                pad_size = len(desc_tokens_ids) * source_samples_per_frame
                pad_audio = torch.zeros(pad_size, device=source_audio.device, dtype=source_audio.dtype)
                source_audio_.append(torch.cat([pad_audio, source_audio[i]]))
                source_audio_lens[i] = source_audio_lens[i] + pad_size
                # add silence in the target audio while the prompt is being processed
                pad_size = len(desc_tokens_ids) * target_samples_per_frame
                pad_audio = torch.zeros(pad_size, device=target_audio.device, dtype=target_audio.dtype)
                target_audio_.append(torch.cat([pad_audio, target_audio[i]]))
                target_audio_lens[i] = target_audio_lens[i] + pad_size

                # save des
                desc_lens.append(len(desc_tokens_ids))

            # collate tensors
            input_text_tokens = collate_vectors(input_text_tokens_, padding_value=text_pad_id)
            source_tokens = collate_vectors(source_tokens_, padding_value=text_pad_id)
            source_audio = collate_vectors(source_audio_, padding_value=0)
            target_audio = collate_vectors(target_audio_, padding_value=0)

            # recreate audio mask
            audio_mask = get_mask_from_lengths(target_token_lens)
            # ignore desc len in audio mask
            for i, frame in enumerate(desc_lens):
                audio_mask[i, :frame] = 0.0

            # desc mask is totally the oposite of audio mask
            desc_mask = ~ audio_mask
        else:
            # create a mask for audio using target tokens that suppose to have the same size of the tokenized audio
            audio_mask = get_mask_from_lengths(target_token_lens)
            # create a full zero desc mask
            desc_mask = torch.zeros_like(audio_mask)
        # Create segment IDs and attention masks
        """aligned_segment_ids = []
        for i, size in enumerate(target_token_lens):
            aligned_segment_ids.extend([i] * size.item())
        aligned_segment_ids = torch.tensor(aligned_segment_ids, dtype=torch.long).unsqueeze(0)
        aligned_attention_mask = (
            (aligned_segment_ids.unsqueeze(-2) == aligned_segment_ids.unsqueeze(-1)).tril().unsqueeze(1)
        )"""

        batch_size = len(target_token_lens)
        max_len = max(target_token_lens)

        # Segment IDs per sequence (padded)
        aligned_segment_ids = torch.stack([
            torch.nn.functional.pad(torch.full((l,), i), (0, max_len - l), value=-1)  # -1 for padding
            for i, l in enumerate(target_token_lens)
        ], dim=0)  # [B, max_len]

        # Attention mask: same-segment & causal
        aligned_attention_mask = (
            (aligned_segment_ids.unsqueeze(-2) == aligned_segment_ids.unsqueeze(-1))  # [B, max_len, max_len]
            & (torch.arange(max_len).unsqueeze(0).unsqueeze(1) 
            <= torch.arange(max_len).unsqueeze(0).unsqueeze(-1))  # causal tril
        )

        aligned_attention_mask = aligned_attention_mask.unsqueeze(1)  # [B, 1, max_len, max_len]

        # create pos ids from the aligned lenght
        # aligned_position_ids = torch.tensor([torch.arange(l) for l in target_token_lens], dtype=torch.long)
        aligned_position_ids = torch.stack([
            torch.nn.functional.pad(torch.arange(l), (0, max(target_token_lens) - l), value=0)  # value=0 is safe for padding
            for l in target_token_lens
        ], dim=0)

        return {
            "sample_id": [str(cut.id) for cut in cuts],
            "audio_mask": audio_mask.bool(),
            "desc_mask": desc_mask.bool(),
            "desc_lens": desc_lens,
            "aligned_attention_mask": aligned_attention_mask.bool(),
            "aligned_position_ids": aligned_position_ids,
            "source_audio": source_audio,
            "source_audio_lens": source_audio_lens,
            "target_audio": target_audio,
            "target_audio_lens": target_audio_lens,
            "input_text_tokens": input_text_tokens,
            "target_token_lens": target_token_lens,
            "source_tokens": source_tokens,
            "source_token_lens": source_token_lens,
            "target_texts": [
                " ".join(s.text for s in cut.supervisions if s.speaker in self.output_roles) for cut in cuts
            ],
            "target_first_turn_audio": target_first_turn_audio,
            "target_first_turn_audio_lens": target_first_turn_audio_lens,
            "formatter": [getattr(cut, "formatter", "s2s_duplex") for cut in cuts],
        }


def collate_first_turn_audio(
    cuts: CutSet,
    roles: set[str],
    recording_field: str = "target_audio",
) -> tuple[torch.Tensor, torch.Tensor]:
    first_turn_audios = []
    first_turn_audios_lens = []
    for cut in cuts:
        first_supervision = [s for s in cut.supervisions if s.speaker in roles][0]
        truncated_audio = cut.truncate(offset=max(0, first_supervision.start), duration=first_supervision.duration).load_custom(recording_field)
        first_turn_audios.append(truncated_audio.squeeze(0))
        first_turn_audios_lens.append(truncated_audio.shape[-1])

    return collate_vectors(first_turn_audios, padding_value=0), torch.tensor(first_turn_audios_lens)


def collate_token_channel(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = get_pad_id(tokenizer)
    tokens = [
        build_token_channel(c, tokenizer=tokenizer, frame_length=frame_length, roles=roles, pad_id=pad_id)
        for c in cuts
    ]
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def build_token_channel(
    cut: Cut,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    pad_id: int = -1,
) -> torch.Tensor:
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id
    for supervision in cut.supervisions:
        if supervision.speaker in roles:
            text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(supervision.text))

            # Determine the frame offset for the start of the supervision to insert the text tokens.
            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos > len(tokens):
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {pos} is larger than the example's length {len(tokens)}. {diagnostic}"
                )
                continue

            # Determine the frame offset for the last non-EOS text token to form a valid range for insertion;
            # Note that EOS will be placed possibly much later, at the frame that coincides with end of speech,
            # rather than end of text. The gap between last non-EOS token and EOS token will be filled with `pad_id`.
            endpos = pos + len(text_ids)
            if endpos > len(tokens):
                trunc_len = len(tokens) - pos
                logging.warning(
                    f"Truncating training example's text_ids of length {len(text_ids)} by {trunc_len} because {endpos=} > {len(tokens)=}. {diagnostic}"
                )
                text_ids = text_ids[:trunc_len]
            try:
                tokens[pos:endpos] = text_ids
            except Exception as e:
                raise RuntimeError(f"{tokens.shape=} {pos=} {endpos=} {text_ids.shape=} {diagnostic}") from e

            # Insert EOS at the end of the supervision segment.
            eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
            if eospos < len(tokens):  # skip otherwise - unfinished turn
                tokens[eospos] = tokenizer.eos

    return tokens


def _strip_timestamps(
    text: str, _TIMESTAMP_PATTERN=re.compile(r"<\|\d+\|>"), _SPACE_PATTERN=re.compile(r"\s+")
) -> str:
    """
    Strips timestamp tokens from text, e.g. turns:
      '<|0|> Hey <|3|> <|3|> how <|5|> <|7|> are <|8|> <|8|> <|10|> you? <|12|>'
      into:
      'Hey how are you?'
    """
    # Regexp pattern args are cached compiled patterns (micro-optimization).
    text = _TIMESTAMP_PATTERN.sub("", text)  # strip timestamp tokens if present
    return _SPACE_PATTERN.sub(" ", text).strip()  # strip multi-whitespaces

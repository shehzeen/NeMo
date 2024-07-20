import torch.utils.data
import random
from lhotse import CutSet
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse
from lhotse.dataset.collation import _read_features

from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import (
    TextProcessing,
    build_loss_mask,
    ceil_to_nearest,
)
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    beta_binomial_prior_distribution,
)



def collate_vectors(items, max_length: int, padding_value):
    vectors = collate_vectors_lhotse(items, padding_value=padding_value)
    if max_length > vectors.size(1):
        vectors = torch.cat(
            [vectors, padding_value * torch.ones(vectors.size(0), max_length - vectors.size(1), dtype=vectors.dtype)],
            dim=1,
        )
    if items[0].shape[0] < 1:
        vectors = vectors.long()
    return vectors


class LhotseAudioQuestionAnswerDataset(torch.utils.data.Dataset):
    """
    Known issues:
        Text limits might still be incorrect in certain batches
        Decoder length (esp with pad to 8) seems to break sometimes
        Does not always return current batch size
    """

    def __init__(
        self,
        text_processor: TextProcessing,
        default_context: str,
        tokens_to_generate: int,
        pad_to_max_length: bool,
        max_seq_length: int,
        task_templates: dict,
        pseudo_tokens,
        context_key: str = "context",
        default_context_key: str = "default_context",
        cross_attention_epsilon = 0.0,
        attention_prior_scaling_factor = 0.05,
    ):
        super().__init__()
        self.text_processor = text_processor
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.tokens_to_generate = tokens_to_generate
        self.pad_to_max_length = pad_to_max_length
        self.max_seq_length = max_seq_length

        self.default_context = default_context
        self.context_key = context_key
        self.default_context_key = default_context_key
        self.task_templates = task_templates
        self.cross_attention_epsilon = cross_attention_epsilon
        self.use_attention_prior = True
        self.context_conditioning ="decoder"
        self.decoder_context_len = 3*86
        self.beta_binomial_interpolator = None
        self.attention_prior_scaling_factor = attention_prior_scaling_factor
        self.pseudo_tokens = pseudo_tokens

    def _insert_virtual_token_placeholders(self, input_example, virtual_token_splits):
        """Insert the correct number of pseudo tokens at the <|VIRTUAL_PROMPT_n|> markers"""
        total_inserted_tokens = 0

        for idx in range(len(virtual_token_splits)):
            split_start = total_inserted_tokens
            split_end = total_inserted_tokens + virtual_token_splits[idx]
            pseudo_tokens_for_split = "".join(self.pseudo_tokens[split_start:split_end])
            input_example = input_example.replace(f'<|VIRTUAL_PROMPT_{idx}|>', pseudo_tokens_for_split)
            total_inserted_tokens = split_end

        return input_example

    def get_position_ids(self, context_and_question):
        enc_input = []
        # enc_input.append(virtual_token)
        if context_and_question.dim() > 2:
            enc_input.append(context_and_question[:, 0, :])
        else:
            enc_input.append(context_and_question)

        enc_input = torch.cat(enc_input, dim=1)

        enc_input_p = enc_input[:, 0, :] if enc_input.dim() == 3 else enc_input
        return build_position_ids(enc_input_p).contiguous()

    def __getitem__(self, cuts) -> dict[str, torch.Tensor | list[str] | dict]:
        cuts = cuts.sort_by_duration()
        cuts_tts = cuts
        print(f"{len(cuts)}")
        # cuts_asr = []

        # for cut in cuts:
        #     try:
        #         if getattr(cut, "tts"):
        #             cuts_tts.append(cut)
        #         else:
        #             cuts_asr.append(cut)
        #     except AttributeError:
        #         cuts_asr.append(cut)

        # cuts = CutSet(cuts_asr)
        # cuts_tts = CutSet(cuts_tts)
        # logging.debug(f"Len_asr: {len(cuts)}")
        # logging.debug(f"Len_tts: {len(cuts_tts)}")
        # return_batch = {}

        # if len(cuts) > 0:
        #     audio, audio_lens, cuts = self.load_audio(cuts)

        #     audio_ratio = []
        #     for id, cut in enumerate(cuts):
        #         audio_ratio.append(1.0)

        #     for _, cut in enumerate(cuts):
        #         if hasattr(cut, self.context_key):
        #             cut.context = getattr(cut, self.context_key)
        #         elif hasattr(cut, self.default_context_key):
        #             cut.context = getattr(cut, self.default_context_key)
        #         else:
        #             cut.context = self.default_context

        #     metadata = []
        #     for id, cut in enumerate(cuts):
        #         metadata.append({'audio_filepath': cut.id + '.wav'})

        #     collated_text_data = collate_text_data(
        #         cuts=cuts,
        #         default_context=self.default_context,
        #         text_processor=self.text_processor,
        #         tokens_to_generate=self.tokens_to_generate,
        #         pad_to_max_length=self.pad_to_max_length,
        #         max_seq_length=self.max_seq_length,
        #     )
        #     # collate_text_data returns 4 fields:
        #     #   - tokens: context + answer; not used in T5 model
        #     #   - labels: tokens rotated; not used in T5 model
        #     #   - answers: Gets sent to decoder in T5 model
        #     #   - context: Gets sent to encoder in T5 model
        #     asr_batch = {
        #         "sample_ids": list(cuts.ids),
        #         "audio_signal": audio,
        #         "audio_signal_length": audio_lens,
        #         "audio_ratio": torch.FloatTensor(audio_ratio),
        #         "metadata": metadata,
        #         **collated_text_data,
        #     }

        # # Now handle TTS if any
        # if len(cuts_tts) > 0:
        # handle text data
        tts_text_data = [
            self.text_processor._process_example(
                context=cut.supervisions[0].text, output=""
            ) for cut in cuts_tts
        ]
        num_samples = len(tts_text_data)
        # tts_text_data = as_dict(tts_text_data)
        # max_length = self.tokens_to_generate + get_max_len(tts_text_data["context_ids"])
        # if self.pad_to_max_length:
        #     max_length = self.max_seq_length
        # else:
        #     max_length = min(self.max_seq_length, ceil_to_nearest(max_length, 8))
        # tts_text_data = collate_vectors(tts_text_data["context_ids"], max_length=max_length, padding_value=pad_id)

        # Build answer and label tensor
        # @shehzeen: cut.num_frames don't always match with feat_i.shape
        # features_lens = torch.tensor([cut.num_frames for cut in cuts_tts], dtype=torch.int)
        _max_ans_len = self.max_seq_length - (3 * 86) - 2
        features_lens = torch.tensor([ min(cut.load_features().shape[0], _max_ans_len) for cut in cuts_tts], dtype=torch.int)
        tts_answer = torch.zeros(len(cuts_tts), max(features_lens).item(), 8) + 1001  # 1001 for speech pad_id
        # Loop through cuts and build tts_answer, label, and context tensors
        speaker_context_list = []
        answers_lens = []
        for i, cut_t in enumerate(cuts_tts):
            feat_i = cut_t.load_features()
            feat_i = feat_i[:_max_ans_len, :] # @shehzeen: Cut to max length. TODO: Fix this to filter out longer examples.
            print("feat_i", feat_i.shape, features_lens[i])
            tts_answer[i,:feat_i.shape[0],:] = torch.tensor(feat_i)
            speaker_context = cut_t.load_context()
            print("Speaker context", speaker_context.shape)
            # take random 3s splice from context
            # TODO: fix hardcode
            rng = random.Random()  # Custom random generator (since random uses fixed seeds). Else context remains fixed
            reference_codec_len = 3 * 86
            reference_codec_len = min(reference_codec_len, speaker_context.shape[0])
            si = rng.randint(0, speaker_context.shape[0] - reference_codec_len)
            speaker_context = speaker_context[si : si + reference_codec_len, :]
            speaker_context_list.append(torch.tensor(speaker_context))
            answers_lens.append(feat_i.shape[0])
        tts_answer = tts_answer.to(torch.int)
        speaker_context = torch.stack(speaker_context_list)

        def get_max_len(input_list):
            return max([len(x) for x in input_list])
        # import ipdb; ipdb.set_trace()

        pad_id = self.text_processor.pad_id
        
        # TODO: Should probably remove this
        taskname = "squad"
        virtual_token_splits = self.task_templates[taskname]["virtual_token_splits"]
        prompt_template = self.task_templates[taskname]["prompt_template"]
        virtual_tokens = self._insert_virtual_token_placeholders(prompt_template.split(' ')[0], virtual_token_splits)
        virtual_tokens = self.text_processor._get_text_tokens(virtual_tokens)
        virtual_tokens = torch.tensor(virtual_tokens)
        # virtual_tokens = virtual_tokens.repeat(num_samples, 1)

        all_text_data = tts_text_data
        bos_tensor = torch.zeros([len(cuts_tts), 1, 8])
        bos_tensor[:,:,0] = self.text_processor.bos_id
        speech_token_offset = self.text_processor.lm_vocab_size  #TODO: should be speech offset not lm_vocab_size
        # for i in range(speaker_context.shape[-1]):
        #     speaker_context[:,:,i] += speech_token_offset + i*1024
        #     tts_answer[:,:,i] += speech_token_offset + i*1024
        speaker_context[:,:,0] += speech_token_offset + i*1024  #Only offset 0th codebook for now
        tts_answer[:,:,0] += speech_token_offset + i*1024
        answers = torch.concat([speaker_context, bos_tensor, tts_answer], 1)
        print("answers", answers.shape)
        # Move wav_tokens above current text token range

        loss_mask = None  #Need to mask out speaker_context potion of audio
        virtual_text_data = [torch.cat([virtual_tokens, i], 0) for i in all_text_data]
        max_length = self.tokens_to_generate + get_max_len(virtual_text_data)
        # virtual_text_data =
        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            # max_length = min(self.max_seq_length, ceil_to_nearest(max_length, 8))
            # @shehzeen not making it a multiple of 8, easier to debug.
            max_length = min(self.max_seq_length, max_length)
        contexts = collate_vectors(virtual_text_data, max_length=max_length, padding_value=pad_id)
        context_lengths = torch.LongTensor([len(seq) for seq in all_text_data])  # text length not including virutal tokens
        max_context_and_question_tokens_len = torch.max(context_lengths).item()

        enc_mask = get_mask_from_lengths(context_lengths + 3)  # 3 virtual tokens
        # print(enc_mask)
        if enc_mask.shape[1] < max_length:
            enc_mask = torch.nn.functional.pad(enc_mask, (0, max_length-enc_mask.shape[1]), "constant", 0)
        # print(enc_mask)

        position_ids = self.get_position_ids(contexts)

        # import ipdb; ipdb.set_trace()
        dec_input = answers[:,:-1,:]
        dec_labels = answers[:,1:,:]
        print("dec_input", dec_input.shape)
        print("dec_labels", dec_labels.shape)

        dec_mask_list = []
        cross_attention_prior_list = []
        text_limits_list = []
        max_dec_labels_len = dec_input.shape[1]

        cross_attention_prior = torch.zeros(dec_input.shape[0], dec_input.shape[1], position_ids.shape[1]) + self.cross_attention_epsilon
        print("cross_attention_prior base", cross_attention_prior.shape)
        # B, Dec_len, enc_len
        for i in range(answers.shape[0]):
            dec_len_i = answers_lens[i] + self.decoder_context_len # @shehzeen: removing + 1, because we have sliced the answer
            loss_mask = (
                torch.as_tensor(([1] * dec_len_i) + ([0] * (max_dec_labels_len - dec_len_i)))
                .long()
                .contiguous()
            )
            dec_mask_list.append(loss_mask)

            start_of_question_offset = 4  # For both "Text to Speech this" and "Phoneme TTS"
            end_of_question_offset = 1
            if self.use_attention_prior:
                prior_dec_len_i = dec_len_i
                prior_dec_start_idx = 0
                if self.context_conditioning == "decoder":
                    prior_dec_len_i = dec_len_i - self.decoder_context_len
                    prior_dec_start_idx = self.decoder_context_len # @shehzeen: removing +1, because we want the prior from the first step of label
                text_len = context_lengths[i].item() - start_of_question_offset - end_of_question_offset
                audio_len = prior_dec_len_i
                if self.beta_binomial_interpolator is not None:
                    cross_attention_question_prior = torch.from_numpy(self.beta_binomial_interpolator(audio_len, text_len))
                else:
                    cross_attention_question_prior = torch.from_numpy(
                        beta_binomial_prior_distribution(
                            text_len,
                            audio_len,
                            scaling_factor=self.attention_prior_scaling_factor,
                        )
                    )
                
                _start_of_text_id = 3 + start_of_question_offset
                _end_of_text_id = _start_of_text_id + text_len
                
                cross_attention_prior[
                    i,
                    prior_dec_start_idx:prior_dec_start_idx+prior_dec_len_i,
                    _start_of_text_id : _end_of_text_id,  # 3 virtual tokens
                ] = cross_attention_question_prior
                
                cross_attention_prior[
                    i,
                    prior_dec_start_idx+prior_dec_len_i:,
                    :,
                ] = 1.
                cross_attention_prior[
                    i,
                    :,
                    _end_of_text_id:,
                ] = 1.

                text_limits_list.append(torch.tensor([_start_of_text_id, _end_of_text_id]))  # Might be incorrect, used in ctc loss so needs to be fixed

        dec_mask = torch.stack(dec_mask_list)
        dec_attn_mask = dec_mask.clone()
        dec_labels_mask = dec_mask.clone()
        dec_labels_mask[:, : self.decoder_context_len + 1] = 0
        speech_mask = dec_mask
        # # Merge batch
        # return_batch ={
        #     "audio_signal": audio,
        #     "audio_signal_length": audio_lens,
        #     "contexts": contexts,
        #     "context_lengths": context_lengths,
        #     "answers": answers,
        #     "loss_mask": loss_mask,
        # }
        # return return_batch
        contexts_expanded = torch.zeros(contexts.shape[0], 8, contexts.shape[1]).long()
        contexts_expanded[:,0,:] = contexts
        return [
            virtual_tokens.repeat(num_samples, 2),  # Not needed anymore since I add it in contexts
            contexts_expanded,
            enc_mask,
            dec_input.transpose(2, 1).long(),
            dec_attn_mask,
            dec_labels.transpose(2, 1).long(),
            dec_labels_mask,
            position_ids.long(),
            torch.tensor(-1).repeat(dec_input.shape[0]),  # taskname_id
            speech_mask,
            None,
            cross_attention_prior,
            torch.stack(text_limits_list),
            None,
            None
        ]


def collate_text_data(
    cuts,
    default_context: str,
    text_processor: TextProcessing,
    tokens_to_generate: int,
    pad_to_max_length: bool,
    max_seq_length: int,
) -> dict:
    """Perform text collation equivalent to nemo/collections/multimodal/data/audio_text_qa_dataset.py:121"""
    batch_size = len(cuts)
    pad_id = text_processor.pad_id
    examples = [
        {
            k: torch.as_tensor(v)
            for k, v in text_processor._process_example(
                context=cut.context,
                output=cut.supervisions[0].text,
            ).items()
        }
        for cut in cuts
    ]
    fields = as_dict(examples)

    def get_max_len(input_list):
        return max([len(x) for x in input_list])

    max_length = tokens_to_generate + max(
        get_max_len(fields["input_ids"]), get_max_len(fields["context_ids"]), get_max_len(fields["answer_ids"])
    )
    # increase max length to nearest multiple of 4 or 8
    if pad_to_max_length:
        max_length = max_seq_length
    else:
        max_length = min(max_seq_length, ceil_to_nearest(max_length, 8))

    all_tokens = collate_vectors(fields["input_ids"], max_length=max_length, padding_value=pad_id)
    full_lengths = torch.LongTensor([len(item) for item in fields["input_ids"]])

    assert max_length <= max_seq_length, f"{max_length=} <= {max_seq_length=}"

    return {
        "tokens": all_tokens[:, :-1],
        "tokens_length": full_lengths - 1,
        "labels": all_tokens[:, 1:],
        "loss_mask": collate_vectors(
            [torch.as_tensor(build_loss_mask(item)) for item in examples], max_length=max_length, padding_value=0
        )[:, 1:],
        "position_ids": torch.arange(max_length, dtype=torch.long).repeat(batch_size, 1),
        "contexts": collate_vectors(fields["context_ids"], max_length=max_length, padding_value=pad_id),
        "context_lengths": torch.LongTensor([len(seq) for seq in fields["context_ids"]]),
        "answers": collate_vectors(fields["answer_ids"], max_length=max_length, padding_value=pad_id),
        "max_length": torch.LongTensor([max_length] * batch_size),
        "context_ids": fields["context_ids"]
    }


def as_dict(arg: list[dict]) -> dict[str, list]:
    return {k: [item[k] for item in arg] for k in arg[0].keys()}

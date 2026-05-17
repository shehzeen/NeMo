.. _easy-magpietts-multiturn:

=================================
EasyMagpieTTS Multi-Turn Training
=================================

This page is a code-oriented guide for researchers and agents extending the
multi-turn EasyMagpieTTS path. It focuses on where data enters the model, what
the batch fields mean, and which files to inspect before adding features.

Source Map
##########

The main files are:

* ``examples/tts/conf/magpietts/easy_magpietts_lhotse_multiturn.yaml``:
  Hydra recipe that enables the multi-turn path with
  ``model.use_lhotse=true`` and ``model.use_multiturn_dataset=true``.
* ``nemo/collections/tts/data/text_to_speech_dataset_lhotse_multiturn.py``:
  Lhotse dataset that converts conversation cuts into frame-aligned channels.
* ``nemo/collections/tts/models/easy_magpietts.py``:
  Lightning training model. The important methods are
  ``get_lhotse_dataloader()``, ``training_step()``, ``validation_step()``, and
  ``process_batch()``.
* ``nemo/collections/tts/models/easy_magpietts_inference.py``:
  Inference/base model. It owns the decoder, codec helpers, special tokens,
  training-mode setup, context preparation, and streaming inference state.
* ``examples/tts/easy_magpietts.py``:
  Script entry point used with the multi-turn config.

High-Level Flow
###############

Training starts from the YAML config. The config selects Lhotse data loading and
sets ``use_multiturn_dataset``. When the model builds the dataloader,
``EasyMagpieTTSModel.get_lhotse_dataloader()`` instantiates
``MagpieTTSLhotseMultiturnDataset`` instead of the regular single-turn dataset.

Each Lhotse cut represents a target conversation segment. Supervisions inside
the cut carry role labels through ``speaker``. By default, user roles are
``user``/``User`` and assistant roles are ``assistant``/``Assistant``/``agent``/
``Agent``. Assistant turns become the speech target. User turns are returned as
``source_*`` fields for experiments that need them; the current supervised
training step does not feed those fields into ``process_batch()``. For cuts
whose ``task`` is ``tts``, the dataset normalizes all supervisions to the
assistant role so that regular single-turn TTS data can be mixed with
duplex/multi-turn data.

The dataset returns a batch dictionary. ``training_step()`` encodes raw audio to
codec codes when cached codes are absent, then calls ``process_batch()`` with:

* assistant text tokens as ``text`` and ``text_lens``
* user/source audio as ``source_audio``/``source_audio_lens`` or
  ``source_codes``/``source_codes_lens``
* optional assistant phoneme tokens as ``phoneme_tokens`` and
  ``phoneme_tokens_lens``
* target assistant audio/codes as ``audio``/``audio_lens`` or
  ``audio_codes``/``audio_codes_lens``
* context audio/codes as ``context_audio``/``context_audio_lens`` or
  ``context_audio_codes``/``context_audio_codes_lens``
* context text tokens as ``context_text_tokens`` and
  ``context_text_tokens_lens``
* ``task`` and ``agent_mask`` metadata

Inside ``process_batch()``, context, text, phoneme, and audio are treated as
separate time channels. Text starts after the context. Phonemes start after the
context plus ``streaming_phonemes_delay``. Audio starts after either
``context_lens + streaming_speech_delay`` in streaming mode or
``context_lens + text_lens + streaming_speech_delay`` in full-text mode. The
channels are padded to a common length, summed elementwise, passed through the
decoder, and sliced back into prediction regions for audio-code and optional
phoneme losses.

Dataset Contract
################

``MagpieTTSLhotseMultiturnDataset.__getitem__()`` receives a Lhotse ``CutSet``
batch and builds aligned tensors on the codec frame grid. Important custom
fields and conventions are:

* ``target_audio`` is the assistant-side waveform used as the training target.
* ``source_audio`` is loaded from the original cut for multi-turn tasks and is
  the user-audio stream consumed by the model when
  ``model.use_user_audio_channel`` is enabled. For regular ``tts`` cuts, it is
  zeroed to keep the tensor shape available without adding user speech.
* ``target_codes``, ``source_codes``, and ``context_codes`` are optional cached
  codec-code arrays. If present and ``load_cached_codes_if_available=true``,
  they avoid on-the-fly codec encoding.
* ``context_audio`` or ``context_codes`` provide voice/style context. If neither
  is present, the dataset samples an assistant turn from the target as context.
* supervision ``context_text`` is optional text conditioning. If missing, the
  dataset emits either ``[NO TEXT CONTEXT]`` or a language tag when
  ``add_language_to_context_text`` is enabled.
* ``tokenizer_names`` can be attached as tags/custom data. The dataset randomly
  selects one tokenizer name per cut and uses ``english_phoneme`` as fallback.
* ``lang`` or supervision ``language`` drives language-aware phoneme handling.

The frame-aligned helper functions near the bottom of
``text_to_speech_dataset_lhotse_multiturn.py`` are central:

* ``build_token_channel()`` places tokenized text at each matching turn start
  and inserts an ``interruption_token_id`` at each matching turn end.
* ``build_phoneme_channel()`` does the same for assistant phoneme tokens.
* ``build_speaker_mask_channel()`` marks assistant frames with ``1.0`` and all
  other frames with ``0.0``.

The dataset also supports ``remove_user_turns_prob``. When triggered on a
multi-turn cut, it collapses audio and token channels down to assistant turns
only. This augmentation currently rejects cached target/source codes because the
cached arrays would no longer match the collapsed timeline.

User Audio And Agent Activity
#############################

The multi-turn model can consume a separate input-only user-audio channel. During
``training_step()`` and ``validation_step()``, ``source_codes`` are used when
present; otherwise ``source_audio`` is encoded by the codec. The resulting codes
are stacked like the assistant audio codes, embedded with the shared audio-code
embeddings, and added to the decoder input as another channel.

The user-audio channel starts after the context plus a configurable delay. During
training, this delay is sampled uniformly from
``model.user_audio_delay_min`` through ``model.user_audio_delay_max`` for each
batch item. During validation it uses ``model.user_audio_delay_min``. This
models possible user-audio latency differences without changing the dataset.
During training, the full user-audio channel embedding is dropped to zero with
probability ``model.user_audio_dropout_prob``.

The dataset's ``agent_mask`` is also used to train an agent-activity stream. The
model aligns ``agent_mask`` to the assistant audio prediction length and converts
it into four classes: inactive, active, beginning-of-turn (BOT), and end-of-turn
(EOT). The activity head predicts these classes from the same decoder hidden
states used for audio-code prediction, and adds a weighted CE loss scaled by
``model.agent_activity_loss_weight``.

BOT is placed before the detected turn start by a configurable extension
(``model.agent_activity_bot_extension_steps``). EOT is placed after a
configurable extension from the detected turn end
(``model.agent_activity_eot_extension_steps``), before the following inactive
region, so EOT is predicted while the teacher-forced assistant audio input is
still non-zero. The teacher-forced assistant audio input is zeroed only for
inactive timesteps. The assistant audio-code and local-transformer losses are
computed only on active agent-speaking timesteps; inactive regions are covered by
the agent-activity loss instead.

Multi-Turn-Specific Model Behavior
##################################

When ``use_multiturn_dataset`` is true, the inference/base model adds one extra
text special token, ``interruption_token_id``, after the normal text BOS, EOS,
and CFG-UNK tokens. The dataset uses this token at assistant turn boundaries.

For ordinary tasks, ``process_batch()`` replaces ``interruption_token_id`` with
text padding before text embedding so turn-boundary markers do not become normal
text input tokens. For tasks whose name contains ``interruption``, the token
remains visible in the text channel. The assistant audio-code stream no longer
receives audio EOS tokens at every turn boundary; it keeps only the regular
sequence-level EOS added by ``prepare_audio_channel_embeddings()``. Turn
activity is modeled by the agent-activity stream.

For multi-turn batches, text and phoneme padding are masked by pad token IDs
rather than only by sequence lengths. This matters because the channel tensors
are sparse over the conversation timeline: many frames intentionally contain
pad IDs between turns.

``agent_mask`` is emitted by the dataset and passed through
``training_step()``/``validation_step()`` into ``process_batch()``. It controls
the agent-activity target, masks inactive assistant audio inputs, and restricts
audio-code losses to active agent-speaking regions.

Training Modes And Streaming
############################

The config's ``model.training_modes`` list is parsed into ``TrainingMode``
objects in ``easy_magpietts_inference.py``. Each mode defines:

* ``text_input_mode``: ``streaming`` or ``full``
* ``streaming_phonemes_delay``: delay before phoneme prediction starts
* ``streaming_speech_delay``: delay before audio prediction starts
* ``mode_idx``: generated from list position

If multiple modes are configured, the model creates a task embedding and
prepends it to the context. During training, ``process_batch()`` randomly picks
a mode per batch unless one is explicitly passed. During validation and default
inference, it uses the first configured mode.

Inference is implemented as a streaming state machine:

* ``streaming_init()`` prepares context audio/text, optional CFG state, cached
  decoder KV state, and selected training mode.
* ``streaming_step()`` advances each item through context, prompt,
  phoneme-only, and audio-generation phases.
* ``streaming_finalize()`` removes special tokens, un-stacks frame groups, and
  decodes codec codes back to waveform.
* ``infer_batch()`` wraps these methods for validation or batch evaluation.

Key Config Knobs
################

Use ``examples/tts/conf/magpietts/easy_magpietts_lhotse_multiturn.yaml`` as the
starting point. The most relevant options are:

* ``model.use_multiturn_dataset``: selects
  ``MagpieTTSLhotseMultiturnDataset``.
* ``model.context_duration_min`` and ``model.context_duration_max``: control
  random slicing or repetition of context audio/codes.
* ``model.load_cached_codes_if_available``: prefers cached code arrays over
  waveform codec encoding.
* ``model.training_modes``: controls full vs streaming alignment and delays.
* ``model.frame_stacking_factor`` and ``model.local_transformer_type``:
  determine stacked-code sequence length and intra-frame codebook prediction.
* ``model.phoneme_tokenizer`` and ``model.phoneme_stacking_factor``: enable the
  auxiliary phoneme channel and phoneme prediction loss.
* ``model.dropout_text_input_prob`` and ``model.cfg_unconditional_prob``:
  regularize text and conditioning inputs.
* ``model.phoneme_corruption_*``: simulates predicted-phoneme errors during
  training.
* ``model.remove_user_turns_prob``: optional assistant-only timeline
  augmentation for multi-turn cuts.
* ``model.use_user_audio_channel``: enables the input-only user/source audio
  stream.
* ``model.user_audio_delay_min`` and ``model.user_audio_delay_max``: inclusive
  train-time delay range for the user-audio stream.
* ``model.user_audio_dropout_prob``: train-time probability of zeroing the
  entire user-audio channel embedding.
* ``model.agent_activity_loss_weight``: scales the weighted CE loss for
  predicting the agent activity class.
* ``model.agent_activity_class_weights``: CE class weights for inactive, active,
  BOT, and EOT. Boundary classes should usually be weighted higher than ordinary
  active/inactive classes.
* ``model.agent_activity_bot_extension_steps``: number of timesteps to extend
  each assistant turn before placing the BOT target.
* ``model.agent_activity_eot_extension_steps``: number of timesteps to extend
  each assistant turn before placing the EOT target.
* ``model.train_ds.dataset``: can mix regular TTS and duplex data with Lhotse
  ``multi_config`` and sampler weights.

Adding New Features
###################

For dataset features, start by deciding whether the new signal is per cut, per
supervision/turn, or per frame. Per-frame features should usually be built next
to ``build_token_channel()``, ``build_phoneme_channel()``, or
``build_speaker_mask_channel()`` so their lengths stay aligned with
``frame_length``.

For model features, first update the batch contract in
``MagpieTTSLhotseMultiturnDataset.__getitem__()``, then thread the new field
through ``training_step()``, ``validation_step()``, and ``process_batch()``.
Keep the channel-delay rules explicit: every feature should define whether it
starts at context time, text time, phoneme time, or audio time.

For inference features, mirror the training-time alignment in
``streaming_init()``, ``streaming_step()``, and ``streaming_finalize()``. If a
feature changes ``training_modes`` semantics, update both mode parsing and the
mode-name lookup used by inference.

Before changing cached-code behavior, check all three paths: cached
``target_codes``/``source_codes``, cached ``context_codes``, and raw waveform
fallback. Cached arrays are fast but easy to desynchronize from timeline
augmentations such as ``remove_user_turns_prob``.

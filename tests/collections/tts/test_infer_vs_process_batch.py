"""
Test script to verify that infer_batch (teacher-forced) produces the same audio code
and phoneme predictions as process_batch (single forward pass).

Usage:
    python tests/collections/tts/test_infer_vs_process_batch.py --codecmodel_path /path/to/codec.nemo

The script:
1. Builds a tiny NemotronH-backed EasyMagpieTTSModel with a real codec model.
2. Creates synthetic random inputs (with variable lengths per batch item).
3. Runs process_batch (full-sequence forward) and infer_batch (streaming, teacher-forced).
4. Compares the argmax audio code predictions and phoneme predictions from both paths.
5. Repeats for multiple configurations.
"""

import argparse
import sys
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel


def build_minimal_config(codecmodel_path: str) -> OmegaConf:
    """Build a minimal OmegaConf config for a tiny NemotronH model."""
    hidden_size = 256

    cfg_dict = {
        # Decoder backend
        'decoder_type': 'nemotron_h',
        'nemotron_h_config': {
            'hidden_size': hidden_size,
            'num_hidden_layers': 2,
            'vocab_size': 131072,
            'num_attention_heads': 4,
            'num_key_value_heads': 2,
            'attention_dropout': 0.0,
            'attention_bias': False,
            'max_position_embeddings': 4096,
            'mamba_num_heads': 16,
            'mamba_head_dim': 16,
            'ssm_state_size': 128,
            'conv_kernel': 4,
            'n_groups': 8,
            'chunk_size': 256,
            'mamba_hidden_act': 'silu',
            'use_conv_bias': True,
            'use_bias': False,
            'intermediate_size': 512,
            'mlp_hidden_act': 'silu',
            'mlp_bias': False,
            'hybrid_override_pattern': 'M*',  # All Mamba layers
            'layer_norm_epsilon': 1e-5,
            'residual_in_fp32': True,
        },
        'embedding_dim': hidden_size,
        'hidden_dim': hidden_size,
        'audio_embedding_dim': hidden_size,
        'codecmodel_path': codecmodel_path,
        # Text tokenizer - use a simple AutoTokenizer
        'text_tokenizers': {
            'test_tokenizer': {
                '_target_': 'AutoTokenizer',
                'pretrained_model': 'gpt2',
            },
        },
        # Phoneme tokenizer
        'phoneme_tokenizer': {
            '_target_': 'nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.IPABPETokenizer',
            'tokenizer_path': 'scripts/tts_dataset_files/bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh.json',
        },
        'phoneme_stacking_factor': 1,
        # Training modes (single streaming mode)
        'training_modes': [
            {
                'name': 'streaming_4_8',
                'text_input_mode': 'streaming',
                'streaming_phonemes_delay': 4,
                'streaming_speech_delay': 8,
            },
        ],
        'frame_stacking_factor': 2,
        'cfg_unconditional_prob': 0.0,
        'dropout_text_input_prob': 0.0,
        'dropout_phoneme_input_prob': 0.0,
        'local_transformer_type': 'none',
        'run_val_inference': False,
        # Optim placeholder (required by ModelPT but not used)
        'optim': {
            '_target_': 'torch.optim.AdamW',
            'lr': 1e-4,
        },
        # No dataloaders
    }
    return OmegaConf.create(cfg_dict)


def create_synthetic_batch(
    model,
    batch_size=2,
    text_lens_list=None,
    audio_frames_list=None,
    context_text_lens_list=None,
    context_audio_frames_list=None,
    phoneme_lens_list=None,
    device='cpu',
):
    """Create a synthetic batch with random valid token IDs and variable lengths per item.

    If *_list args are None, defaults to uniform lengths for all items.
    """
    num_codebooks = model.num_audio_codebooks
    codebook_size = model.codebook_size
    text_vocab_size = model.bos_id  # valid text tokens are [0, bos_id)
    phoneme_vocab_size = model.phoneme_tokenizer.vocab_size - 2  # exclude BOS/EOS

    # Defaults
    if text_lens_list is None:
        text_lens_list = [20] * batch_size
    if audio_frames_list is None:
        audio_frames_list = [30] * batch_size
    if context_text_lens_list is None:
        context_text_lens_list = [10] * batch_size
    if context_audio_frames_list is None:
        context_audio_frames_list = [15] * batch_size
    if phoneme_lens_list is None:
        phoneme_lens_list = [25] * batch_size

    assert len(text_lens_list) == batch_size
    assert len(audio_frames_list) == batch_size
    assert len(context_text_lens_list) == batch_size
    assert len(context_audio_frames_list) == batch_size
    assert len(phoneme_lens_list) == batch_size

    # Max lengths for padding
    max_text_len = max(text_lens_list)
    max_audio_frames = max(audio_frames_list)
    max_context_text_len = max(context_text_lens_list)
    max_context_audio_frames = max(context_audio_frames_list)
    max_phoneme_len = max(phoneme_lens_list)

    # Text tokens: random tokens + EOS at the end (matching dataset behavior)
    text = torch.zeros(batch_size, max_text_len, dtype=torch.long, device=device)
    for b in range(batch_size):
        tl = text_lens_list[b]
        text[b, :tl - 1] = torch.randint(0, text_vocab_size, (tl - 1,), device=device)
        text[b, tl - 1] = model.eos_id  # EOS as last valid token
    text_lens = torch.tensor(text_lens_list, dtype=torch.long, device=device)

    # Context text tokens
    context_text_tokens = torch.zeros(batch_size, max_context_text_len, dtype=torch.long, device=device)
    for b in range(batch_size):
        cl = context_text_lens_list[b]
        context_text_tokens[b, :cl] = torch.randint(0, text_vocab_size, (cl,), device=device)
    context_text_tokens_lens = torch.tensor(context_text_lens_list, dtype=torch.long, device=device)

    # Audio codes (raw, without BOS/EOS)
    audio_codes = torch.zeros(batch_size, num_codebooks, max_audio_frames, dtype=torch.long, device=device)
    for b in range(batch_size):
        af = audio_frames_list[b]
        audio_codes[b, :, :af] = torch.randint(0, codebook_size, (num_codebooks, af), device=device)
    audio_codes_lens = torch.tensor(audio_frames_list, dtype=torch.long, device=device)

    # Context audio codes (raw, without BOS/EOS)
    context_audio_codes = torch.zeros(batch_size, num_codebooks, max_context_audio_frames, dtype=torch.long, device=device)
    for b in range(batch_size):
        caf = context_audio_frames_list[b]
        context_audio_codes[b, :, :caf] = torch.randint(0, codebook_size, (num_codebooks, caf), device=device)
    context_audio_codes_lens = torch.tensor(context_audio_frames_list, dtype=torch.long, device=device)

    # Phoneme tokens (raw IDs, BOS/EOS will be added by the model)
    phoneme_tokens = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long, device=device)
    for b in range(batch_size):
        pl = phoneme_lens_list[b]
        phoneme_tokens[b, :pl] = torch.randint(0, phoneme_vocab_size, (pl,), device=device)
    phoneme_tokens_lens = torch.tensor(phoneme_lens_list, dtype=torch.long, device=device)

    batch = {
        'text': text,
        'text_lens': text_lens,
        'context_text_tokens': context_text_tokens,
        'context_text_tokens_lens': context_text_tokens_lens,
        'audio_codes': audio_codes,
        'audio_codes_lens': audio_codes_lens,
        'context_audio_codes': context_audio_codes,
        'context_audio_codes_lens': context_audio_codes_lens,
        'phoneme_tokens': phoneme_tokens,
        'phoneme_tokens_lens': phoneme_tokens_lens,
    }
    return batch


def compare_audio_codes(model, pb_output, ib_output, batch):
    """Compare audio codes from process_batch and infer_batch. Returns True if all match."""
    C = model.num_audio_codebooks
    S = model.frame_stacking_factor
    C_stacked = C * S
    V = model.num_all_tokens_per_codebook
    pb_logits = pb_output.logits  # (B, T_stacked, C_stacked * V)
    T_stacked = pb_logits.size(1)
    batch_size = batch['text'].size(0)

    # Extract per-codebook argmax at stacked resolution
    pb_stacked_codes_list = []
    for cb_idx in range(C_stacked):
        si = cb_idx * V
        ei = si + V
        cb_logits = pb_logits[:, :, si:ei]  # (B, T_stacked, V)
        cb_preds = cb_logits.argmax(dim=-1)  # (B, T_stacked)
        pb_stacked_codes_list.append(cb_preds)
    pb_stacked_codes = torch.stack(pb_stacked_codes_list, dim=1)  # (B, C_stacked, T_stacked)

    # Unstack: (B, C*S, T_stacked) -> (B, C, S, T_stacked) -> (B, C, T_stacked, S) -> (B, C, T_stacked*S)
    pb_unstacked = pb_stacked_codes.view(batch_size, C, S, T_stacked)
    pb_unstacked = pb_unstacked.permute(0, 1, 3, 2).contiguous()
    pb_unstacked = pb_unstacked.reshape(batch_size, C, T_stacked * S)
    pb_unstacked_lens = pb_output.audio_codes_lens_target * S

    ib_codes = ib_output.predicted_codes
    ib_codes_lens = ib_output.predicted_codes_lens

    print(f"  process_batch argmax codes (unstacked): {pb_unstacked.shape}, lens: {pb_unstacked_lens.tolist()}")
    print(f"  infer_batch predicted codes: {ib_codes.shape}, lens: {ib_codes_lens.tolist()}")

    all_match = True
    for b in range(batch_size):
        pb_len = pb_unstacked_lens[b].item()
        ib_len = ib_codes_lens[b].item()
        compare_len = min(pb_len, ib_len)

        if compare_len == 0:
            print(f"  Batch item {b}: No codes to compare (pb_len={pb_len}, ib_len={ib_len})")
            continue

        pb_codes_b = pb_unstacked[b, :, :compare_len]
        ib_codes_b = ib_codes[b, :, :compare_len]

        matches = (pb_codes_b == ib_codes_b).all()
        num_matching = (pb_codes_b == ib_codes_b).sum().item()
        total = pb_codes_b.numel()
        match_pct = 100.0 * num_matching / total if total > 0 else 0.0

        print(f"  Batch item {b}: pb_len={pb_len}, ib_len={ib_len}, compare_len={compare_len}")
        print(f"    Audio match: {matches.item()}, {num_matching}/{total} ({match_pct:.1f}%)")

        if not matches:
            all_match = False
            mismatch_mask = pb_codes_b != ib_codes_b
            mismatch_positions = mismatch_mask.nonzero(as_tuple=False)
            num_show = min(10, mismatch_positions.size(0))
            for i in range(num_show):
                cb, t = mismatch_positions[i].tolist()
                print(f"    Mismatch at codebook={cb}, time={t}: "
                      f"pb={pb_codes_b[cb, t].item()}, ib={ib_codes_b[cb, t].item()}")

    return all_match


def compare_phoneme_predictions(model, pb_output, ib_output, batch):
    """Compare phoneme predictions from process_batch and infer_batch. Returns True if all match."""
    if pb_output.phoneme_logits is None:
        print("  No phoneme logits from process_batch (no phoneme tokenizer?). Skipping.")
        return True
    if ib_output.predicted_phoneme_tokens is None:
        print("  No phoneme predictions from infer_batch. Skipping.")
        return True

    batch_size = batch['text'].size(0)
    phoneme_stacking_factor = model.phoneme_stacking_factor
    phoneme_vocab_size = model.phoneme_vocab_size

    # Extract argmax phoneme predictions from process_batch logits
    # phoneme_logits: (B, T_phoneme, phoneme_stacking_factor * phoneme_vocab_size)
    pb_phoneme_logits = pb_output.phoneme_logits
    T_phoneme = pb_phoneme_logits.size(1)

    pb_phoneme_preds_list = []
    for sf_idx in range(phoneme_stacking_factor):
        si = sf_idx * phoneme_vocab_size
        ei = si + phoneme_vocab_size
        sf_logits = pb_phoneme_logits[:, :, si:ei]  # (B, T_phoneme, V_phoneme)
        sf_preds = sf_logits.argmax(dim=-1)  # (B, T_phoneme)
        pb_phoneme_preds_list.append(sf_preds)
    pb_phoneme_preds = torch.stack(pb_phoneme_preds_list, dim=1)  # (B, phoneme_stacking_factor, T_phoneme)
    pb_phoneme_lens = pb_output.phoneme_tokens_lens_target  # (B,) number of phoneme prediction steps

    # infer_batch phoneme predictions: (B, phoneme_stacking_factor, T_all_steps)
    ib_phoneme_preds = ib_output.predicted_phoneme_tokens
    ib_phoneme_lens = ib_output.predicted_phoneme_tokens_lens

    print(f"  process_batch phoneme preds: {pb_phoneme_preds.shape}, lens: {pb_phoneme_lens.tolist()}")
    print(f"  infer_batch phoneme preds: {ib_phoneme_preds.shape}, lens: {ib_phoneme_lens.tolist()}")

    # Get start indices for infer_batch phoneme predictions
    ib_start_idx = ib_output.phoneme_prediction_start_idx  # (B,)

    all_match = True
    for b in range(batch_size):
        pb_len = pb_phoneme_lens[b].item()
        ib_len = ib_phoneme_lens[b].item()
        compare_len = min(pb_len, ib_len)

        if compare_len == 0:
            print(f"  Batch item {b}: No phonemes to compare (pb_len={pb_len}, ib_len={ib_len})")
            continue

        # process_batch phoneme preds start from 0 (already sliced to prediction region)
        pb_ph_b = pb_phoneme_preds[b, :, :compare_len]

        # infer_batch phoneme preds: slice from start_idx for this batch item
        start = max(0, ib_start_idx[b].item())
        ib_ph_b = ib_phoneme_preds[b, :, start:start + compare_len]

        matches = (pb_ph_b == ib_ph_b).all()
        num_matching = (pb_ph_b == ib_ph_b).sum().item()
        total = pb_ph_b.numel()
        match_pct = 100.0 * num_matching / total if total > 0 else 0.0

        print(f"  Batch item {b}: pb_len={pb_len}, ib_len={ib_len}, compare_len={compare_len}")
        print(f"    Phoneme match: {matches.item()}, {num_matching}/{total} ({match_pct:.1f}%)")

        if not matches:
            all_match = False
            mismatch_mask = pb_ph_b != ib_ph_b
            mismatch_positions = mismatch_mask.nonzero(as_tuple=False)
            num_show = min(10, mismatch_positions.size(0))
            for i in range(num_show):
                sf, t = mismatch_positions[i].tolist()
                print(f"    Mismatch at stacking_factor={sf}, time={t}: "
                      f"pb={pb_ph_b[sf, t].item()}, ib={ib_ph_b[sf, t].item()}")

    return all_match


def run_single_test(model, batch, test_name, device):
    """Run a single test comparing process_batch and infer_batch outputs."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # Run process_batch
    print("\n  Running process_batch...")
    training_mode = model.training_modes[0]
    with torch.inference_mode():
        pb_output = model.process_batch(
            text=batch['text'],
            text_lens=batch['text_lens'],
            context_text_tokens=batch['context_text_tokens'],
            context_text_tokens_lens=batch['context_text_tokens_lens'],
            audio_codes=batch['audio_codes'],
            audio_codes_lens=batch['audio_codes_lens'],
            context_audio_codes=batch['context_audio_codes'],
            context_audio_codes_lens=batch['context_audio_codes_lens'],
            phoneme_tokens=batch['phoneme_tokens'],
            phoneme_tokens_lens=batch['phoneme_tokens_lens'],
            mode='val',
            training_mode=training_mode,
        )

    # Run infer_batch (teacher-forced)
    print("  Running infer_batch (teacher-forced)...")
    ib_output = model.infer_batch(
        batch=batch,
        max_decoder_steps=1000,
        temperature=0.0,
        topk=80,
        use_cfg=False,
        use_local_transformer_for_inference=False,
        phoneme_input_type='gt',
        phoneme_sampling_method='argmax',
        use_teacher_forced=True,
    )

    # Compare audio codes
    print("\n  --- Audio Codes Comparison ---")
    audio_match = compare_audio_codes(model, pb_output, ib_output, batch)

    # Compare phoneme predictions
    print("\n  --- Phoneme Predictions Comparison ---")
    phoneme_match = compare_phoneme_predictions(model, pb_output, ib_output, batch)

    success = audio_match and phoneme_match
    if success:
        print(f"\n  ✓ {test_name}: PASSED (audio + phoneme match)")
    else:
        parts = []
        if not audio_match:
            parts.append("audio")
        if not phoneme_match:
            parts.append("phoneme")
        print(f"\n  ✗ {test_name}: FAILED ({' and '.join(parts)} mismatch)")

    return success


def main():
    parser = argparse.ArgumentParser(description='Test infer_batch vs process_batch')
    parser.add_argument('--codecmodel_path', type=str, required=True, help='Path to codec model .nemo file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    # 1. Build config and model
    print("Building minimal config...")
    cfg = build_minimal_config(args.codecmodel_path)

    print("Instantiating EasyMagpieTTSModel (tiny NemotronH + real codec)...")
    model = EasyMagpieTTSModel(cfg=cfg, trainer=None)
    model = model.to(device)
    model.eval()
    print(f"  num_audio_codebooks={model.num_audio_codebooks}, codebook_size={model.codebook_size}")
    print(f"  frame_stacking_factor={model.frame_stacking_factor}")
    print(f"  phoneme_vocab_size={model.phoneme_tokenizer.vocab_size}")

    # Define test configurations: (test_name, kwargs_for_create_synthetic_batch)
    test_configs = [
        (
            "Uniform lengths (B=2, text=20, audio=30, ctx_text=10, ctx_audio=15, phoneme=25)",
            dict(
                batch_size=2,
                text_lens_list=[20, 20],
                audio_frames_list=[30, 30],
                context_text_lens_list=[10, 10],
                context_audio_frames_list=[15, 15],
                phoneme_lens_list=[25, 25],
            ),
        ),
        (
            "Variable text & context lens (B=2, text=[15,25], ctx_text=[8,12], ctx_audio=[10,20])",
            dict(
                batch_size=2,
                text_lens_list=[15, 25],
                audio_frames_list=[30, 30],
                context_text_lens_list=[8, 12],
                context_audio_frames_list=[10, 20],
                phoneme_lens_list=[20, 30],
            ),
        ),
        (
            "Variable audio & phoneme lens (B=2, audio=[20,40], phoneme=[15,35])",
            dict(
                batch_size=2,
                text_lens_list=[20, 20],
                audio_frames_list=[20, 40],
                context_text_lens_list=[10, 10],
                context_audio_frames_list=[15, 15],
                phoneme_lens_list=[15, 35],
            ),
        ),
        (
            "All different (B=3)",
            dict(
                batch_size=3,
                text_lens_list=[12, 20, 28],
                audio_frames_list=[20, 30, 40],
                context_text_lens_list=[6, 10, 14],
                context_audio_frames_list=[8, 15, 22],
                phoneme_lens_list=[15, 25, 35],
            ),
        ),
    ]

    all_passed = True
    for test_name, kwargs in test_configs:
        batch = create_synthetic_batch(model, device=device, **kwargs)
        passed = run_single_test(model, batch, test_name, device)
        if not passed:
            all_passed = False

    # Final summary
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

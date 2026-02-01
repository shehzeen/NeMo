#!/usr/bin/env python3
"""Simple test script for IPABPETokenizer."""

import sys
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import IPABPETokenizer

# tokenizer_path = "scripts/tts_dataset_files/bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh.json"
def main():
    if len(sys.argv) < 2:
        print("Usage: python test_ipa_tokenizer.py <tokenizer_path>")
        sys.exit(1)

    tokenizer_path = sys.argv[1]
    tokenizer = IPABPETokenizer(tokenizer_path)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print()

    # Test IPA strings
    test_texts = [
        "həˈloʊ wɝːld",
        "ðɪs ɪz ə tɛst",
        "aɪ kæn spik ˈɪŋɡlɪʃ",
        "ˈwɛlkəm tuː ðə ˈfjuːtʃɚ",
        "ˌɑːtɪfɪʃəl ɪnˈtɛlɪdʒəns",
    ]

    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"Original: {text}")
        print(f"IDs:      {ids}")
        print(f"Decoded:  {decoded}")
        print(f"Match:    {text == decoded}")
        print()


if __name__ == "__main__":
    main()

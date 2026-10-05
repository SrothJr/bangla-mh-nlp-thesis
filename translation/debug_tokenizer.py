
import sys
import os

print("Trying to load GPT2 tokenizer from local ./gpt2 directory...")
try:
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2")
    print("GPT2Tokenizer loaded successfully!")
    print("tokenizer.__dict__.keys():", list(tokenizer.__dict__.keys()))
    print("tokenizer.vocab_size:", tokenizer.vocab_size)
    print("tokenizer._vocab (first 10 items):", dict(list(tokenizer._vocab.items())[:10]))
    print("len(tokenizer._vocab):", len(tokenizer._vocab))
    print("Testing tokenize \"Hello\":", tokenizer.tokenize("Hello"))
    print("Testing encode \"Hello\":", tokenizer.encode("Hello"))
    print("Testing decode:", tokenizer.decode(tokenizer.encode("Hello")))
except Exception as e:
    print("ERROR loading GPT2Tokenizer:", type(e), str(e))
    import traceback
    traceback.print_exc()

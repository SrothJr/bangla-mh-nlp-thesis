
import sys

print("Python version:", sys.version)
print("Trying to import tokenizers...")
try:
    import tokenizers
    print("Imported tokenizers successfully!")
    print("tokenizers version:", tokenizers.__version__)
    print("Testing Tokenizer class...")
    from tokenizers import Tokenizer
    print("Tokenizer imported successfully!")
except Exception as e:
    print("ERROR importing tokenizers:", type(e), str(e))
    import traceback
    traceback.print_exc()

print("\nTrying to import sentencepiece...")
try:
    import sentencepiece as spm
    print("Imported sentencepiece successfully!")
    print("sentencepiece version:", spm.__version__)
except Exception as e:
    print("ERROR importing sentencepiece:", type(e), str(e))
    import traceback
    traceback.print_exc()

print("\nTrying to import tiktoken...")
try:
    import tiktoken
    print("Imported tiktoken successfully!")
    print("tiktoken version:", tiktoken.__version__)
except Exception as e:
    print("ERROR importing tiktoken:", type(e), str(e))
    import traceback
    traceback.print_exc()

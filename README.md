# Learning to (Learn at Test Time): RNNs with Expressive Hidden States

[**Arxiv**]()
| [**JAX Main Codebase**](https://github.com/test-time-training/ttt-lm-jax)
| [**Setup**](#environment-setup)
| [**Quick Start**](#quick-start)

This is official PyTorch implementation of [Learning to (Learn at Test Time): RNNs with Expressive Hidden States]().

## Environment setup

```bash
pip install "transformers[torch]"
```

## Quick start

Our implementation is based on Huggingface Transformers. You can use the following code to load the model and generate text.

```python
from transformers import AutoTokenizer
from modeling_ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS

# Initializing a TTT ttt-1b style configuration
# configuration = TTTConfig(**TTT_STANDARD_CONFIGS['ttt-1b']) is equivalent to the following
configuration = TTTConfig()

# Initializing a model from the ttt-1b style configuration
model = TTTForCausalLM(configuration)
model.eval()

# Accessing the model configuration
configuration = model.config

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

# prefill
input_ids = tokenizer("Greeting from TTT!", return_tensors="pt").input_ids
logits = model(input_ids=input_ids)
print(logits)

# decoding
out_ids = model.generate(input_ids=input_ids, max_length=50)
out_str = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
print(out_str)
```

Note: this is only a naive implementation of the TTT model for tutorial purpose. We will release faster kernel implementation shortly.
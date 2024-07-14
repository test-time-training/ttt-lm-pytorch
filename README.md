# Learning to (Learn at Test Time): RNNs with Expressive Hidden States

[**Paper**](https://arxiv.org/abs/2407.04620)
| [**JAX Codebase**](https://github.com/test-time-training/ttt-lm-jax)
| [**Setup**](#environment-setup)
| [**Quick Start**](#quick-start)
| [**Inference Benchmark**](https://github.com/test-time-training/ttt-lm-kernels)

This is the official PyTorch model implementation of [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620). 
We **do not recommend training** with this codebase, because it is written in pure PyTorch without any systems optimization, so training will be slow, especially when the per-device batch size is small.


For training code, or to replicate results from our paper, please view our [JAX codebase](https://github.com/test-time-training/ttt-lm-jax). For inference kernels, or to replicate speed benchmarks from our paper, please view our [kernel implementations](https://github.com/test-time-training/ttt-lm-kernels).

## Abstract

Self-attention performs well in long context but has quadratic complexity. Existing RNN layers
have linear complexity, but their performance in long context is limited by the expressive power
of their hidden state. We propose a new class of sequence modeling layers with linear complexity
and an expressive hidden state. The key idea is to make the hidden state a machine learning
model itself, and the update rule a step of self-supervised learning. 

Since the hidden state is updated by training even on test sequences, our layers are called **Test-Time Training (TTT) layers**.
We consider two instantiations: TTT-Linear and TTT-MLP, whose hidden state is a linear model
and a two-layer MLP respectively. 

## Environment Setup

```bash
pip install "transformers[torch]"
```

## Quick Start

Our implementation is based on Huggingface Transformers. You can use the following code to load the model and generate text.

```python
from transformers import AutoTokenizer
from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS

# Initializing a TTT ttt-1b style configuration
# configuration = TTTConfig(**TTT_STANDARD_CONFIGS['1b']) is equivalent to the following
configuration = TTTConfig()

# Initializing a model from the ttt-1b style configuration
model = TTTForCausalLM(configuration)
model.eval()

# Accessing the model configuration
configuration = model.config

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

# Prefill
input_ids = tokenizer("Greeting from TTT!", return_tensors="pt").input_ids
logits = model(input_ids=input_ids)
print(logits)

# Decoding
out_ids = model.generate(input_ids=input_ids, max_length=50)
out_str = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
print(out_str)
```

**Note: This is a naive implementation of TTT layers for tutorial purposes.** This model can be trained using Huggingface Accelerate, or custom training loops. We have released our faster inference kernel and its speed benchmark [here](https://github.com/test-time-training/ttt-lm-kernels).

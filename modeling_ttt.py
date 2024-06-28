"""PyTorch TTT model."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import einops
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils._pytree import tree_map

from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers.utils.import_utils import is_causal_conv1d_available

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None


logger = logging.get_logger(__name__)

TTT_STANDARD_CONFIGS = {
    "1b": {
        "hidden_size": 2048,
        "intermediate_size": 5504,
        "num_hidden_layers": 24,
        "num_attention_heads": 32,
    },
    "125m": {
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
    },
    "350m": {
        "hidden_size": 1024,
        "intermediate_size": 2736,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
    },
    "760m": {
        "hidden_size": 1536,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
    },
}

class TTTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TTTModel`]. It is used to instantiate an TTT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the TTT-1B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        use_mixer (`bool`, *optional*, defaults to `False`): whether use gating in Mamba backbone
        share_qk (`bool`, *optional*, defaults to `False`): whether share Q/K projection matrix
        inner_net_type (`str`, *optional*, defaults to `"linear"`): inner net type, "linear" or "mlp", stands for TTT-Linear and TTT-MLP
        conv_before_ttt (`bool`, *optional*, defaults to `False`): whether use conv before TTT
        conv_kernel (`int`, *optional*, defaults to 4): kernel size of the conv layer
        scan_checkpoint_group (`int`, *optional*, defaults to 0): 
            gradient checkpoint group size on seq dimension, 0 means no checkpointing.
            In JAX implementation, we set it 4, which means we group 4 mini-batches together in 1 gradient checkpointg to save memory.


    ```python
    >>> from . import TTTModel, TTTConfig

    >>> # Initializing a TTT ttt-1b style configuration
    >>> configuration = TTTConfig()

    >>> # Initializing a model from the ttt-1b style configuration
    >>> model = TTTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ttt"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5504,
        num_hidden_layers=24,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        use_mixer=False,
        share_qk=False,
        inner_net_type="linear",
        inner_net_lr=1.0,
        inner_net_chunk_size=16,
        conv_before_ttt=False,
        conv_kernel=4,
        scan_checkpoint_group=0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta

        self.use_mixer = use_mixer
        self.share_qk = share_qk
        self.inner_net_type = inner_net_type
        self.inner_net_lr = inner_net_lr
        self.inner_net_chunk_size = inner_net_chunk_size

        self.conv_before_ttt = conv_before_ttt
        self.conv_kernel = conv_kernel
        self.scan_checkpoint_group = scan_checkpoint_group

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class TTTCache:
    """
    Arguments:
        model: TTTModel
        batch_size: int

    Attributes:
        seqlen_offset: int
        inner_chunk_size: int
        params_dic: Dict[str, Dict[int, torch.Tensor]]  *_states, *_grad -> # layer_idx -> [batch_sise, ...]
        conv_states_dic: Dict[str, Dict[int, torch.Tensor]]  *_states -> # layer_idx -> [batch_sise, ...]

    """

    def __init__(self, model, batch_size: int):
        config = model.config
        self.seqlen_offset = 0
        self.inner_chunk_size = config.inner_net_chunk_size

        self.inner_params_dic = defaultdict(dict)
        if "linear" in config.inner_net_type:
            self.inner_param_names = ["W1", "b1"]
        elif "mlp" in config.inner_net_type:
            self.inner_param_names = ["W1", "b1", "W2", "b2"]
        else:
            raise ValueError(f"inner_net_type {config.inner_net_type} not supported yet")

        self.conv_states_dic = defaultdict(dict)
        logger.info(f"Creating cache of size: {batch_size}")
        for layer_idx in range(config.num_hidden_layers):
            for name in self.inner_param_names:
                weight = getattr(model.layers[layer_idx].seq_modeling, name)
                tiled_weight = torch.tile(weight.unsqueeze(0), (batch_size,) + (1,) * weight.dim()).to(model.device)
                self.inner_params_dic[f"{name}_states"][layer_idx] = tiled_weight
                # for decoding, we need to store the gradients as well
                self.inner_params_dic[f"{name}_grad"][layer_idx] = torch.zeros_like(tiled_weight)

            if config.conv_before_ttt:
                self.conv_states_dic["conv_before_ttt"][layer_idx] = torch.zeros(
                    batch_size, config.hidden_size, config.conv_kernel, device=model.device
                )
            if config.use_mixer and config.share_qk:
                self.conv_states_dic["ttt_conv_q"][layer_idx] = torch.zeros(
                    batch_size, config.hidden_size, config.conv_kernel, device=model.device
                )
                self.conv_states_dic["ttt_conv_k"][layer_idx] = torch.zeros(
                    batch_size, config.hidden_size, config.conv_kernel, device=model.device
                )

    def update(self, py_tree, layer_idx, seq_len):
        if seq_len % self.inner_chunk_size == 0:
            # copy last chunk states, clear gradients
            for name in self.inner_param_names:
                self.inner_params_dic[f"{name}_states"][layer_idx].copy_(py_tree[f"{name}_states"])
                self.inner_params_dic[f"{name}_grad"][layer_idx].zero_()
        elif seq_len < self.inner_chunk_size:
            if seq_len != 1 and self.seqlen_offset > 0 and self.seqlen_offset % self.inner_chunk_size != 0:
                raise ValueError("fractional update not supported yet.")
            if (seq_len + self.seqlen_offset) % self.inner_chunk_size == 0:
                # copy last chunk states, clear gradients
                for name in self.inner_param_names:
                    self.inner_params_dic[f"{name}_states"][layer_idx].copy_(py_tree[f"{name}_states"])
                    self.inner_params_dic[f"{name}_grad"][layer_idx].zero_()
            else:
                # copy gradients for the next update
                for name in self.inner_param_names:
                    self.inner_params_dic[f"{name}_grad"][layer_idx].copy_(py_tree[f"{name}_grad"])
        else:
            raise ValueError(f"seq_len {seq_len} is a partial update not supported yet")

    def inner_params_to_dic(self, layer_idx):
        return {name: self.inner_params_dic[name][layer_idx] for name in self.inner_params_dic}


# Copied from transformers.models.llama.modeling_llama with Llama->TTT
class TTTRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        TTTRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.llama.modeling_llama with Llama->TTT
class TTTRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=16, base=10000, device=None, scaling_factor=1.0):
        """
        TTTRotary is equivalent to LlamaLayerLlamaRotaryEmbedding in implementation, except TTT sets max_position_embedding to inner_chunk_size.
        """

        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def permute_qk(q, k):
    # NOTE: EasyLM and transformers use different method to compute rotary emebdding
    # we manually reorder the dim here to match our JAX implementation
    # which may not be optimal for speed
    # reference: https://github.com/young-geng/EasyLM/blob/981a2ed9630f44258a94b6f44dff2b7bd203ae8d/EasyLM/models/llama/convert_hf_to_easylm.py#L33
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)

    return q, k


def undo_permute_qk(q, k):
    # NOTE: EasyLM and transformers use different method to compute rotary emebdding
    # we manually undo the reorder the dim here to match our JAX implementation
    # which may not be optimal for speed
    # reference: https://github.com/young-geng/EasyLM/blob/981a2ed9630f44258a94b6f44dff2b7bd203ae8d/EasyLM/models/llama/convert_hf_to_easylm.py#L33
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)

    return q, k


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama with Llama->TTT
class TTTMLP(nn.Module):
    def __init__(self, config):
        """
        TTTMLP is equivalent to LlamaMLP, it's applied after temporal mixing module.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class TTTConv(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.norm = TTTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            kernel_size=config.conv_kernel,
            groups=config.hidden_size,
            padding=config.conv_kernel - 1,
        )

    def __call__(self, hidden_states, cache_params=None):
        seq_len = hidden_states.shape[1]
        hidden_states = self.norm(hidden_states)
        # [B, C, L]
        hidden_states = hidden_states.transpose(1, 2)

        if causal_conv1d_fn is None:
            if cache_params is not None:
                if cache_params.seqlen_offset > 0:
                    conv_state = cache_params.conv_states_dic["conv_before_ttt"][self.layer_idx]
                    conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                    conv_state[:, :, -1] = hidden_states[:, :, 0]
                    cache_params.conv_states_dic["conv_before_ttt"][self.layer_idx].copy_(conv_state)
                    hidden_states = torch.sum(conv_state * self.conv.weight[:, 0, :], dim=-1)
                    hidden_states += self.conv.bias
                    hidden_states = hidden_states.unsqueeze(-1)
                else:
                    conv_state = nn.functional.pad(
                        hidden_states, (self.config.conv_kernel - hidden_states.shape[-1], 0)
                    )
                    cache_params.conv_states_dic["conv_before_ttt"][self.layer_idx].copy_(conv_state)
                    hidden_states = self.conv(hidden_states)[..., :seq_len]
            else:
                hidden_states = self.conv(hidden_states)[..., :seq_len]
        else:
            conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
            if cache_params is not None and cache_params.seqlen_offset > 0:
                hidden_states = causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache_params.conv_states_dic["conv_before_ttt"][self.layer_idx],
                    conv_weights,
                    self.conv.bias,
                    None,
                )
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(
                        hidden_states, (self.config.conv_kernel - hidden_states.shape[-1], 0)
                    )
                    cache_params.conv_states_dic["conv_before_ttt"][self.layer_idx].copy_(conv_states)
                hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv.bias, activation=None)

        # [B, L, C]
        hidden_states = hidden_states.transpose(1, 2)

        return hidden_states


def scan(f, init, xs, out, checkpoint_group=0):
    """Minic jax.lax.scan function."""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            out[i] = y
        return carry

    if checkpoint_group > 0:
        ckpt_every_n = num_items // checkpoint_group
        for k in range(0, num_items, ckpt_every_n):
            carry = torch.utils.checkpoint.checkpoint(
                scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False
            )
    else:
        carry = scan_fn(carry, 0, num_items)

    return carry, out


class TTTBaseModule(nn.Module):
    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.width = config.hidden_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.inner_chunk_size = config.inner_net_chunk_size

        # token_idx is a scale factor that scale the summation in Eqn. 4
        token_idx = 1.0 / torch.arange(1, self.inner_chunk_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)
        # make the scale factor learnable
        self.learnable_token_idx = nn.Parameter(torch.zeros((self.inner_chunk_size,)))

        self.share_qk = config.share_qk
        self.conv_kernel = config.conv_kernel
        self._init_qkvo_proj()
        self._init_rope()
        # Learnable eta in Sec. 2.7
        self._init_gated_ilr()
        self._init_decoder_ln()

        # use gating as in Mamba backbone
        self.use_mixer = config.use_mixer
        if self.use_mixer:
            self.g_proj = nn.Linear(self.width, self.width, bias=False)

        self.out_ln = nn.LayerNorm(self.width, eps=1e-6)

    def _init_qkvo_proj(self):
        self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        # we share Q/K projection when using Mamba backbone
        if not self.share_qk:
            self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)

        # after share Q/K projection, we use different conv layers for Q and K
        if self.share_qk:
            self.conv_q = nn.Conv1d(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                kernel_size=self.conv_kernel,
                groups=self.hidden_size,
                padding=self.conv_kernel - 1,
            )
            self.conv_k = nn.Conv1d(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                kernel_size=self.conv_kernel,
                groups=self.hidden_size,
                padding=self.conv_kernel - 1,
            )

    def _init_rope(self):
        self.rope_theta = self.config.rope_theta
        self.rotary_emb = TTTRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.inner_chunk_size,
            base=self.rope_theta,
        )

    def _init_gated_ilr(self):
        # [width, 1]
        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        # prepending head dim -> [num_heads, width, 1]
        self.linear_weight = nn.Parameter(
            torch.stack([torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)], dim=0)
        )
        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        # init bias to 0 following original JAX impl.
        # [num_heads, 1]
        self.linear_bias = nn.Parameter(
            torch.stack([torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)], dim=0)
        )

    def _init_decoder_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        # prepending head dim -> [num_heads, width]
        self.ln_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ln_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

    def get_qkv_projections(self, hidden_states, cache_params: Optional[TTTCache] = None):
        if self.share_qk:
            xq, XV = self.q_proj(hidden_states), self.v_proj(hidden_states)
            seq_len = xq.shape[1]
            xq = xq.transpose(1, 2)
            if causal_conv1d_fn is None:
                if cache_params is not None:
                    if cache_params.seqlen_offset > 0:
                        conv_q_state = cache_params.conv_states_dic["ttt_conv_q"][self.layer_idx]
                        conv_q_state = torch.roll(conv_q_state, shifts=-1, dims=-1)
                        conv_q_state[:, :, -1] = xq[:, :, 0]
                        cache_params.conv_states_dic["ttt_conv_q"][self.layer_idx].copy_(conv_q_state)
                        XQ = torch.sum(conv_q_state * self.conv_q.weight[:, 0, :], dim=-1)
                        XQ += self.conv_q.bias
                        XQ = XQ.unsqueeze(-1)

                        conv_k_state = cache_params.conv_states_dic["ttt_conv_k"][self.layer_idx]
                        conv_k_state = torch.roll(conv_k_state, shifts=-1, dims=-1)
                        conv_k_state[:, :, -1] = xq[:, :, 0]
                        cache_params.conv_states_dic["ttt_conv_k"][self.layer_idx].copy_(conv_k_state)
                        XK = torch.sum(conv_k_state * self.conv_k.weight[:, 0, :], dim=-1)
                        XK += self.conv_k.bias
                        XK = XK.unsqueeze(-1)
                    else:
                        conv_q_state = nn.functional.pad(xq, (self.config.conv_kernel - xq.shape[-1], 0))
                        cache_params.conv_states_dic["ttt_conv_q"][self.layer_idx].copy_(conv_q_state)
                        XQ = self.conv_q(xq)[..., :seq_len]
                        conv_k_state = nn.functional.pad(xq, (self.config.conv_kernel - xq.shape[-1], 0))
                        cache_params.conv_states_dic["ttt_conv_k"][self.layer_idx].copy_(conv_k_state)
                        XK = self.conv_k(xq)[..., :seq_len]
                else:
                    XQ = self.conv_q(xq)[..., :seq_len]
                    XK = self.conv_k(xq)[..., :seq_len]
            else:
                conv_q_weights = self.conv_q.weight.view(self.conv_q.weight.size(0), self.conv_q.weight.size(2))
                conv_k_weights = self.conv_k.weight.view(self.conv_k.weight.size(0), self.conv_k.weight.size(2))
                if cache_params is not None and cache_params.seqlen_offset > 0:
                    XQ = causal_conv1d_update(
                        xq.squeeze(-1),
                        cache_params.conv_states_dic["ttt_conv_q"][self.layer_idx],
                        conv_q_weights,
                        self.conv_q.bias,
                        None,
                    )
                    XQ = XQ.unsqueeze(-1)
                    XK = causal_conv1d_update(
                        xq.squeeze(-1),
                        cache_params.conv_states_dic["ttt_conv_k"][self.layer_idx],
                        conv_k_weights,
                        self.conv_k.bias,
                        None,
                    )
                    XK = XK.unsqueeze(-1)
                else:
                    if cache_params is not None:
                        conv_q_states = nn.functional.pad(xq, (self.config.conv_kernel - xq.shape[-1], 0))
                        cache_params.conv_states_dic["ttt_conv_q"][self.layer_idx].copy_(conv_q_states)
                        conv_k_states = nn.functional.pad(xq, (self.config.conv_kernel - xq.shape[-1], 0))
                        cache_params.conv_states_dic["ttt_conv_k"][self.layer_idx].copy_(conv_k_states)
                    XQ = causal_conv1d_fn(xq, conv_q_weights, self.conv_q.bias, activation=None)
                    XK = causal_conv1d_fn(xq, conv_k_weights, self.conv_k.bias, activation=None)

            XQ = XQ.transpose(1, 2)
            XK = XK.transpose(1, 2)
        else:
            XQ, XK, XV = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        return XQ, XK, XV

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _split_chunks(self, hidden_states, inner_chunk_size=None):
        assert hidden_states.ndim == 4
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], -1, inner_chunk_size, self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)  # [B,nh,n_chunk,K,f]
        return hidden_states

    def get_coeff(self, X, inner_chunk_step_offset, inner_chunk_size):
        # [B, num_heads, n_chunk, inner_chunk_size, 1]
        ilr_gated = torch.einsum("bnkc,hdc->bhnkd", X, self.linear_weight) + self.linear_bias.reshape(1, -1, 1, 1, 1)
        ilr_gated = F.sigmoid(ilr_gated)

        # [B, num_heads, n_chunk, 1, inner_chunk_size]
        ilr_gated = ilr_gated.permute(0, 1, 2, 4, 3)
        # [B, L]
        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = token_idx[inner_chunk_step_offset : inner_chunk_step_offset + inner_chunk_size]

        # token idx should be greast than 0
        token_idx = torch.clamp_min(token_idx, 0.0)

        # return coeff
        # [B, num_heads, n_chunk, inner_chunk_size, 1]
        token_coeff = torch.broadcast_to(
            token_idx.reshape(1, 1, 1, inner_chunk_size, 1),
            (X.shape[0], self.num_heads, X.shape[1], inner_chunk_size, 1),
        )
        # [B, num_heads, n_chunk, inner_chunk_size, 1] or [B, num_heads, n_chunk, 1, inner_chunk_size]
        ilr_coeff = self.config.inner_net_lr * ilr_gated / self.head_dim

        return token_coeff, ilr_coeff

    def gate_with_mixer(self, hidden_states, ttt_output):
        y = self.g_proj(hidden_states)
        # use 'tanh' approximation for matching JAX impl.
        y = F.gelu(y, approximate="tanh")
        output = y * ttt_output
        return output

    def prepare_inner_loop_chunk_inputs(self, inputs, inner_chunk_size, cache_params):
        XQ = inputs["XQ"]
        XK = inputs["XK"]
        XV = inputs["XV"]
        X = inputs["X"]
        B, L, C = X.shape
        n_chunk = L // inner_chunk_size
        # [B ,n_chunk, inner_chunk_size, C]
        X = X.reshape(B, n_chunk, inner_chunk_size, self.width)

        XQ = XQ.reshape(B, self.num_heads, L // inner_chunk_size, inner_chunk_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, L // inner_chunk_size, inner_chunk_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, L // inner_chunk_size, inner_chunk_size, self.head_dim)

        if cache_params is not None:
            inner_chunk_step_offset = cache_params.seqlen_offset % self.inner_chunk_size
        else:
            inner_chunk_step_offset = 0
        token_coeff, ilr_coeff = self.get_coeff(X, inner_chunk_step_offset, inner_chunk_size)
        coeff = token_coeff * ilr_coeff
        # decouple token_coeff and ilr_coeff for decoding
        inputs = {"XQ": XQ, "XK": XK, "XV": XV, "coeff": coeff, "token_coeff": token_coeff, "ilr_coeff": ilr_coeff}
        return inputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TTTCache] = None,
    ):
        B, L = hidden_states.shape[:2]
        reminder_len = L % self.inner_chunk_size
        num_chunks = L // self.inner_chunk_size
        last_chunk_params_dic = None

        XQ, XK, XV = self.get_qkv_projections(hidden_states, cache_params=cache_params)

        # [B, L, C] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(XV, position_ids % self.inner_chunk_size)

        # permute_qk and undo_permute_qk is just for aligning pytorch with jax pre-training
        XQ, XK = permute_qk(XQ, XK)
        XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin)
        XQ, XK = undo_permute_qk(XQ, XK)

        output_hidden_states = []
        # when input sequence length is not a multiple of inner_chunk_size
        # we need to compute them seperately, when computing the reminder,
        # we will need the last_chunk_params_dic to continue inner loop learning
        if num_chunks > 0:
            inputs = {
                "XQ": XQ[:, :, : num_chunks * self.inner_chunk_size],
                "XK": XK[:, :, : num_chunks * self.inner_chunk_size],
                "XV": XV[:, :, : num_chunks * self.inner_chunk_size],
                "X": hidden_states[:, : num_chunks * self.inner_chunk_size],
            }
            output_chunk, last_chunk_params_dic = self.process_inner_loop(
                self.prepare_inner_loop_chunk_inputs(inputs, self.inner_chunk_size, cache_params),
                inner_chunk_size=self.inner_chunk_size,
                last_chunk_params_dic=last_chunk_params_dic,
                cache_params=cache_params,
            )
            output_hidden_states.append(output_chunk)
        if reminder_len > 0:
            inputs = {
                "XQ": XQ[:, :, -reminder_len:],
                "XK": XK[:, :, -reminder_len:],
                "XV": XV[:, :, -reminder_len:],
                "X": hidden_states[:, -reminder_len:],
            }
            output_chunk, _ = self.process_inner_loop(
                self.prepare_inner_loop_chunk_inputs(inputs, reminder_len, cache_params),
                inner_chunk_size=reminder_len,
                last_chunk_params_dic=last_chunk_params_dic,
                cache_params=cache_params,
            )
            output_hidden_states.append(output_chunk)

        output_hidden_states = torch.cat(output_hidden_states, dim=1)
        output_hidden_states = self.out_ln(output_hidden_states)
        if self.use_mixer:
            output_hidden_states = self.gate_with_mixer(hidden_states, output_hidden_states)
        output_hidden_states = self.o_proj(output_hidden_states)

        return output_hidden_states


def ln_fwd(x, gamma, beta, eps=1e-6):
    "Batch forward for LayerNorm."

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    return y


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    "Batch backward for LayerNorm fused with L2 loss."
    D = x.shape[-1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )

    return z


# Modified from https://github.com/NVIDIA/Megatron-LM/blob/e33c8f78a35765d5aa37475a144da60e8a2349d1/megatron/core/fusions/fused_bias_gelu.py#L26
def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff


class TTTLinearModule(TTTBaseModule):
    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # inner loop weight matrix initialization for TTT-Linear
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        # inner loop bias initialization
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def process_inner_loop(
        self, inputs, inner_chunk_size, last_chunk_params_dic, cache_params: Optional[TTTCache] = None
    ):
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size

        # in this case, we are decoding
        if last_chunk_params_dic is None and cache_params is not None:
            last_chunk_params_dic = cache_params.inner_params_to_dic(self.layer_idx)

        B = inputs["XV"].shape[0]  # [B, nh, NC, CS, f]
        num_chunks = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        # NOTE:
        # for prefilling, we will always use dual form for faster computation
        # we need to use primal form if inner_chunk_size is not a multiple of self.inner_chunk_size
        # since we need store the gradient for the next mini-batch computation
        use_dual_form = cache_params is None or inner_chunk_size % self.inner_chunk_size == 0

        def compute_chunk(params_dic, inputs):
            W1_init = params_dic["W1_states"]  # [B,nh,f,f]
            b1_init = params_dic["b1_states"]  # [B,nh,1,f]

            XV_chunk = inputs["XV"]  # [B,nh,K=1,f]
            XK_chunk = inputs["XK"]
            XQ_chunk = inputs["XQ"]
            coeff_chunk = inputs["coeff"]  # [B,nh,K=1,1]
            token_coeff_chunk = inputs["token_coeff"]
            ilr_coeff_chunk = inputs["ilr_coeff"]

            X1 = XK_chunk
            Z1 = X1 @ W1_init + b1_init  # [B,nh,K=1,f] @ [B,nh,f,f] -> [B,nh,K=1,f]
            reconstruction_target = XV_chunk - XK_chunk

            ln_weight = self.ln_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ln_bias.reshape(self.num_heads, 1, self.head_dim)
            grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)  # [B,nh,K,f]

            if use_dual_form:
                Attn1 = torch.tril(XQ_chunk @ X1.transpose(-2, -1))  # [B,nh,K,K]
                b1_bar = (
                    b1_init - torch.tril(coeff_chunk) @ grad_l_wrt_Z1
                )  # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
                Z1_bar = (
                    XQ_chunk @ W1_init - (coeff_chunk * Attn1) @ grad_l_wrt_Z1 + b1_bar
                )  # [B,nh,K,f] @ [B,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]

                last_coeff_chunk = coeff_chunk[:, :, -1, :, None]
                W1_last = (
                    W1_init - (last_coeff_chunk * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                )  # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                b1_last = b1_init - torch.sum(last_coeff_chunk * grad_l_wrt_Z1, dim=-2, keepdim=True)  # [B,nh,1,f]
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
            else:
                ilr_coeff_chunk = torch.broadcast_to(
                    ilr_coeff_chunk, (*ilr_coeff_chunk.shape[:2], inner_chunk_size, inner_chunk_size)
                )

                # [B, nh, K, f, f]
                grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ilr_coeff_chunk), grad_W1)
                grad_W1 = grad_W1 + params_dic["W1_grad"].unsqueeze(2)
                # [B, nh, K, f]
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ilr_coeff_chunk), grad_l_wrt_Z1)
                grad_b1 = grad_b1 + params_dic["b1_grad"]

                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_coeff_chunk.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_coeff_chunk

                # [B, nh, K, 1, f] @ [B, nh, K, f, f]
                Z1_bar = (XQ_chunk.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]

            Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)

            XQW_chunk = XQ_chunk + Z1_bar

            last_param_dic = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W1_grad": grad_W1_last,
                "b1_grad": grad_b1_last,
            }
            return last_param_dic, XQW_chunk

        if last_chunk_params_dic is not None:
            init_params_dic = last_chunk_params_dic
        else:
            init_params_dic = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
            }
            init_params_dic.update(W1_grad=torch.zeros_like(init_params_dic["W1_states"]))
            init_params_dic.update(b1_grad=torch.zeros_like(init_params_dic["b1_states"]))
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)  # [B,nh,NC,CS,f] -> [NC,B,nh,CS,f]

        # allocate output tensor
        XQW_batch = torch.empty(
            (num_chunks, B, self.num_heads, inner_chunk_size, self.head_dim), device=device, dtype=dtype
        )
        batch_params_dic, XQW_batch = scan(
            compute_chunk,
            init_params_dic,
            inputs,
            XQW_batch,
            self.config.scan_checkpoint_group if self.training else 0,
        )  # [NC,B,nh,CS,f]

        # [B, num_heads, L, C]
        if cache_params is not None:
            cache_params.update(batch_params_dic, self.layer_idx, L)

        XQW_batch = einops.rearrange(XQW_batch, "nc b nh cs f -> b (nc cs) (nh f)")  # [B,L,f]
        return XQW_batch, batch_params_dic


class TTTMLPModule(TTTBaseModule):
    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # inner loop weight matrix initialization for TTT-MLP
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def process_inner_loop(
        self, inputs, inner_chunk_size, last_chunk_params_dic, cache_params: Optional[TTTCache] = None
    ):
        # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size

        # in this case, we are decoding
        if last_chunk_params_dic is None and cache_params is not None:
            last_chunk_params_dic = cache_params.inner_params_to_dic(self.layer_idx)

        B = inputs["XV"].shape[0]  # [B, nh, NC, CS, f]
        num_chunks = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype
        # NOTE:
        # for prefilling, we will always use dual form for faster computation
        # we need to use primal form if inner_chunk_size is not a multiple of self.inner_chunk_size
        # since we need store the gradient for the next mini-batch computation
        use_dual_form = cache_params is None or inner_chunk_size % self.inner_chunk_size == 0

        def compute_chunk(params_dic, inputs):
            W1_init = params_dic["W1_states"]  # [B,nh,f,f]
            b1_init = params_dic["b1_states"]  # [B,nh,1,f]
            W2_init = params_dic["W2_states"]  # [B,nh,f,f]
            b2_init = params_dic["b2_states"]  # [B,nh,1,f]

            XV_chunk = inputs["XV"]  # [B,nh,K=1,f]
            XK_chunk = inputs["XK"]
            XQ_chunk = inputs["XQ"]
            coeff_chunk = inputs["coeff"]  # [B,nh,K=1,1]
            token_coeff_chunk = inputs["token_coeff"]
            ilr_coeff_chunk = inputs["ilr_coeff"]

            X1 = XK_chunk
            Z1 = X1 @ W1_init + b1_init  # [B,nh,K=1,f] @ [B,nh,f,f] -> [B,nh,K=1,f]
            X2 = F.gelu(Z1, approximate="tanh")
            Z2 = X2 @ W2_init + b2_init
            reconstruction_target = XV_chunk - XK_chunk

            ln_weight = self.ln_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ln_bias.reshape(self.num_heads, 1, self.head_dim)
            grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)  # [B,nh,K,f]
            # [B, nh, K, f]
            grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2, -1) * gelu_bwd(Z1)

            if use_dual_form:
                Attn1 = torch.tril(XQ_chunk @ X1.transpose(-2, -1))  # [B,nh,K,K]
                b1_bar = (
                    b1_init - torch.tril(coeff_chunk) @ grad_l_wrt_Z1
                )  # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
                Z1_bar = (
                    XQ_chunk @ W1_init - (coeff_chunk * Attn1) @ grad_l_wrt_Z1 + b1_bar
                )  # [B,nh,K,f] @ [B,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
                X2_bar = F.gelu(Z1_bar, approximate="tanh")

                Attn2 = torch.tril(X2_bar @ X2.transpose(-2, -1))  # [B,nh,K,K]
                b2_bar = (
                    b2_init - torch.tril(coeff_chunk) @ grad_l_wrt_Z2
                )  # [B,nh,1,f] - [B,nh,K,1] * [B,nh,K,f] -> [B,nh,K,f]
                Z2_bar = (
                    X2_bar @ W2_init - (coeff_chunk * Attn2) @ grad_l_wrt_Z2 + b2_bar
                )  # [B,nh,K,f] @ [1,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]

                last_coeff_chunk = coeff_chunk[:, :, -1, :, None]
                W1_last = (
                    W1_init - (last_coeff_chunk * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                )  # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                b1_last = b1_init - torch.sum(last_coeff_chunk * grad_l_wrt_Z1, dim=-2, keepdim=True)  # [B,nh,1,f]
                W2_last = (
                    W2_init - (last_coeff_chunk * X2).transpose(-1, -2) @ grad_l_wrt_Z2
                )  # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                b2_last = b2_init - torch.sum(last_coeff_chunk * grad_l_wrt_Z2, dim=-2, keepdim=True)  # [B,nh,1,f]
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
                grad_W2_last = torch.zeros_like(W2_last)
                grad_b2_last = torch.zeros_like(b2_last)

            else:
                ilr_coeff_chunk = torch.broadcast_to(
                    ilr_coeff_chunk, (*ilr_coeff_chunk.shape[:2], inner_chunk_size, inner_chunk_size)
                )

                # [B, nh, K, f, f]
                grad_W2 = torch.einsum("bhki,bhkj->bhkij", X2, grad_l_wrt_Z2)
                grad_W2 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ilr_coeff_chunk), grad_W2)
                grad_W2 = grad_W2 + params_dic["W2_grad"].unsqueeze(2)
                # [B, nh, K, f]
                grad_b2 = torch.einsum("bhnk,bhki->bhni", torch.tril(ilr_coeff_chunk), grad_l_wrt_Z2)
                grad_b2 = grad_b2 + params_dic["b2_grad"]

                # [B, nh, K, f, f]
                grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ilr_coeff_chunk), grad_W1)
                grad_W1 = grad_W1 + params_dic["W1_grad"].unsqueeze(2)
                # [B, nh, K, f]
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ilr_coeff_chunk), grad_l_wrt_Z1)
                grad_b1 = grad_b1 + params_dic["b1_grad"]

                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_coeff_chunk.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_coeff_chunk
                W2_bar = W2_init.unsqueeze(2) - grad_W2 * token_coeff_chunk.unsqueeze(-1)
                b2_bar = b2_init - grad_b2 * token_coeff_chunk

                # [B, nh, K, 1, f] @ [B, nh, K, f, f]
                Z1_bar = (XQ_chunk.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar
                X2_bar = F.gelu(Z1_bar, approximate="tanh")
                Z2_bar = (X2_bar.unsqueeze(3) @ W2_bar).squeeze(3) + b2_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                W2_last = W2_bar[:, :, -1]
                b2_last = b2_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]
                grad_W2_last = grad_W2[:, :, -1]
                grad_b2_last = grad_b2[:, :, -1:]

            Z2_bar = ln_fwd(Z2_bar, ln_weight, ln_bias)

            XQW_chunk = XQ_chunk + Z2_bar

            last_param_dic = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W2_states": W2_last,
                "b2_states": b2_last,
                "W1_grad": grad_W1_last,
                "b1_grad": grad_b1_last,
                "W2_grad": grad_W2_last,
                "b2_grad": grad_b2_last,
            }
            return last_param_dic, XQW_chunk

        if last_chunk_params_dic is not None:
            init_params_dic = last_chunk_params_dic
        else:
            init_params_dic = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "W2_states": torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b2_states": torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1)),
            }
            init_params_dic.update(W1_grad=torch.zeros_like(init_params_dic["W1_states"]))
            init_params_dic.update(b1_grad=torch.zeros_like(init_params_dic["b1_states"]))
            init_params_dic.update(W2_grad=torch.zeros_like(init_params_dic["W2_states"]))
            init_params_dic.update(b2_grad=torch.zeros_like(init_params_dic["b2_states"]))
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)  # [B,nh,NC,CS,f] -> [NC,B,nh,CS,f]
        # allocate output tensor
        XQW_batch = torch.empty(
            (num_chunks, B, self.num_heads, inner_chunk_size, self.head_dim), device=device, dtype=dtype
        )
        batch_params_dic, XQW_batch = scan(
            compute_chunk,
            init_params_dic,
            inputs,
            XQW_batch,
            self.config.scan_checkpoint_group if self.training else 0,
        )  # [NC,B,nh,CS,f]

        # [B, num_heads, L, C]
        if cache_params is not None:
            cache_params.update(batch_params_dic, self.layer_idx, L)

        XQW_batch = einops.rearrange(XQW_batch, "nc b nh cs f -> b (nc cs) (nh f)")  # [B,L,f]
        return XQW_batch, batch_params_dic


class TTTDecoderLayer(nn.Module):
    def __init__(self, config: TTTConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.conv_before_ttt = config.conv_before_ttt

        if config.inner_net_type == "linear":
            ttt_module = TTTLinearModule
        elif config.inner_net_type == "mlp":
            ttt_module = TTTMLPModule
        else:
            raise ValueError(f"Invalid inner_net_type: {config.inner_net_type}")

        self.seq_modeling = ttt_module(config=config, layer_idx=layer_idx)

        self.mlp = TTTMLP(config)
        if self.conv_before_ttt:
            self.conv = TTTConv(config, layer_idx)

        self.input_layernorm = TTTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TTTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TTTCache] = None,
    ):
        if self.conv_before_ttt:
            residual = hidden_states
            hidden_states = self.conv(hidden_states, cache_params=cache_params)
            hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # TTT
        hidden_states = self.seq_modeling(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TTTPreTrainedModel(PreTrainedModel):
    config_class = TTTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TTTDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@dataclass
class TTTOutput(ModelOutput):
    """
    Class for the TTT model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`TTTCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[TTTCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class TTTCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`TTTCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[TTTCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TTTModel(TTTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TTTDecoderLayer`]

    Args:
        config: TTTConfig
    """

    def __init__(self, config: TTTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TTTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = TTTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[TTTCache] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_params is None and use_cache:
            cache_params = TTTCache(self, inputs_embeds.size(0))

        seqlen_offset = 0
        if cache_params is not None:
            seqlen_offset = cache_params.seqlen_offset
        position_ids = torch.arange(
            seqlen_offset, seqlen_offset + inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device
        ).unsqueeze(0)

        hidden_states = inputs_embeds

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    cache_params,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_params=cache_params,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return TTTOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


class TTTForCausalLM(TTTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = TTTModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, cache_params: Optional[TTTCache] = None, inputs_embeds=None, **kwargs
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            attention_mask = attention_mask[:, -1].unsqueeze(-1) if attention_mask is not None else None

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )

        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[TTTCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        *,
        output_attentions: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert not output_attentions, "output_attentions is not available in TTTForCausalLM"

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return TTTCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states,
        )

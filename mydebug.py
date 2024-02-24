import torch
import torch.nn as nn
import xformers
import xformers.ops


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
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
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


device = torch.device('cuda:0')
vocab_size = 32000
batch_size = 2
sequence_length = 128
hidden_size = 4096
intermediate_size = 11008
num_hidden_layers = 32
num_attention_heads = 32
num_key_value_heads = 32
attention_bias = False
max_position_embeddings = 2048
rope_theta = 10000.0

num_heads = num_attention_heads
head_dim = hidden_size // num_heads
num_key_value_groups = num_heads // num_key_value_heads

input_ids = torch.randint(
    0,
    vocab_size,
    (batch_size, sequence_length),
    dtype=torch.long,
    device=device
)
input_mask = torch.ones(
    batch_size,
    sequence_length,
    dtype=torch.bool,
    device=device
)

token_embed_op = nn.Embedding(vocab_size, hidden_size, device=device)

input_ids = input_ids.transpose(0, 1)
input_embeds = token_embed_op(input_ids)
hidden_encoder_states = {
    "hidden_states": input_embeds,
    "sequence_mask": input_mask,
}

q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias, device=device)
k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias, device=device)
v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias, device=device)
o_proj = nn.Linear(hidden_size, hidden_size, bias=attention_bias, device=device)

query_states = q_proj(input_embeds)
key_states = k_proj(input_embeds)
value_states = v_proj(input_embeds)


rotary_emb_op = LlamaRotaryEmbedding(
                head_dim,
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
                device=device
            )

query_states = query_states.view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)
key_states = key_states.view(batch_size, sequence_length, num_key_value_heads, head_dim).transpose(1, 2)
value_states = value_states.view(batch_size, sequence_length, num_key_value_heads, head_dim).transpose(1, 2)
kv_seq_len = key_states.shape[-2]

cos, sin = rotary_emb_op(value_states, seq_len=kv_seq_len)

query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids=None)
# [bsz, nh, t, hd]

query_states = query_states.transpose(1, 2)
key_states = key_states.transpose(1, 2)
value_states = value_states.transpose(1, 2)

# This is a nasty hack. We know attention_mask in transformers is either LowerTriangular or all Zeros.
# We therefore check if one element in the upper triangular portion is zero. If it is, then the mask is all zeros.
if False:# attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
    # input and output should be of form (bsz, q_len, num_heads, head_dim)
    attn_output = xformers.ops.memory_efficient_attention(
        query_states, key_states, value_states, attn_bias=None
    )
else:
    # input and output should be of form (bsz, q_len, num_heads, head_dim)
    attn_output = xformers.ops.memory_efficient_attention(
        query_states,
        key_states,
        value_states,
        attn_bias=xformers.ops.LowerTriangularMask(),
    )

attn_output = attn_output.reshape(batch_size, sequence_length, hidden_size)

from flash_attn import bert_padding
from flash_attn.flash_attn_interface import flash_attn_varlen_func


kv_length = key_states.shape[1]
# [batch_size, seq_length, num_heads, d_qk]
# Shaping for use in `flash-attn` version of flash-attn: `flash_attn_unpadded_func`
query_states = query_states.view(
    batch_size * sequence_length,  num_heads, head_dim
)  # [batch_size * q_length, self.n_heads, d_qk]

key_states = key_states.view(
    batch_size * sequence_length, num_heads, head_dim
)  # [batch_size * kv_length, self.n_heads, d_qk]
value_states = value_states.view(
    batch_size * sequence_length, num_heads, head_dim
)  # [batch_size * kv_length, self.n_heads, d_v]

q_sequence_mask = input_mask
kv_sequence_mask = input_mask

# TODO @thomasw21: Compute once, instead of computing for each layers.
cu_seqlens_q = torch.zeros((q_sequence_mask.shape[0] + 1), dtype=torch.int32, device=query_states.device)
cu_seqlens_k = torch.zeros((kv_sequence_mask.shape[0] + 1), dtype=torch.int32, device=query_states.device)
torch.cumsum(q_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32, out=cu_seqlens_q[1:])
torch.cumsum(kv_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32, out=cu_seqlens_k[1:])

# TODO(kunhao): flash attn's causal means that the query can only attend to the keys before it. This is not
# what we want if we are using kv cache. This is a hack as we always have q_length == 1 when using kv cache.
causal = False if q_sequence_mask.shape[1] == 1 else True
torch_dtype = torch.bfloat16
attention_output = flash_attn_varlen_func(
    q=query_states.to(torch_dtype),
    k=key_states.to(torch_dtype),
    v=value_states.to(torch_dtype),
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=q_sequence_mask.shape[1],
    max_seqlen_k=kv_sequence_mask.shape[1],
    dropout_p=0.0,
    softmax_scale=None,  # This already defaults to the scale I'm interested in
    causal=causal,
    return_attn_probs=False,
)

attention_output = attention_output.contiguous().view(batch_size, sequence_length, num_heads * head_dim)





# torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml > log.txt 2>&1 &
# for jobid in `ps -u | grep python | awk '{ print $2 }'`; do kill $jobid; done





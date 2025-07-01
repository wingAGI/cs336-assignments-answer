from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor

import numpy as np
import regex as re
import time

##### Helper functions #####

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def to_bytes_tuple(word: str) -> Tuple[bytes]:
    l = list(word.encode("utf-8"))
    l = [bytes([x]) for x in l]
    return tuple(l)

##### Helper functions #####

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    return torch.matmul(in_features, weights.T)
    


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    return weights[token_ids]


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    cond = run_silu(torch.matmul(in_features, w1_weight.T))
    x = torch.matmul(in_features, w3_weight.T)
    
    return torch.matmul(cond * x, w2_weight.T)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    attn = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.shape[-1]))
    if mask is not None:    
        attn = attn.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(attn, dim=-1)

    return torch.matmul(attn, V)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_v d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    d_k, d_in = q_proj_weight.shape
    d_v = v_proj_weight.shape[0]

    q = run_linear(d_in, d_k, q_proj_weight, in_features)
    k = run_linear(d_in, d_k, k_proj_weight, in_features)
    v = run_linear(d_in, d_v, v_proj_weight, in_features)   # B, T, d_v // nh

    q = q.view(q.shape[:-1] + (num_heads, -1))   # B, T, nh, d_k // nh
    k = k.view(k.shape[:-1] + (num_heads, -1))
    v = v.view(v.shape[:-1] + (num_heads, -1))   # B, T, nh, d_v // nh

    q = q.permute(0, 2, 1, 3)   # B, nh, T, d_k // nh
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)   # B, nh, T, d_v // nh

    # casual mask
    seq_len = q.shape[2]
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    attn_out = run_scaled_dot_product_attention(q, k, v, mask) # B, nh, T, d_v // nh

    attn_out = attn_out.permute(0, 2, 1, 3) # B, T, nh, d_v // nh
    attn_out = attn_out.contiguous()
    attn_out = attn_out.view(attn_out.shape[:-2] + (d_v,))    # B, T, d_v

    out = run_linear(d_v, d_model, o_proj_weight, attn_out)

    return out



def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    d_k, d_in = q_proj_weight.shape
    d_v = v_proj_weight.shape[0]
    seq_len = in_features.shape[-2]

    q = run_linear(d_in, d_k, q_proj_weight, in_features)
    k = run_linear(d_in, d_k, k_proj_weight, in_features)
    v = run_linear(d_in, d_v, v_proj_weight, in_features)   # B, T, d_v // nh

    q = q.view(q.shape[:-1] + (num_heads, -1))   # B, T, nh, d_k // nh
    k = k.view(k.shape[:-1] + (num_heads, -1))
    v = v.view(v.shape[:-1] + (num_heads, -1))   # B, T, nh, d_v // nh

    q = q.permute(0, 2, 1, 3)   # B, nh, T, d_k // nh
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)   # B, nh, T, d_v // nh

    # rope
    if token_positions is not None:
        q = run_rope(d_k // num_heads, theta, seq_len, q, token_positions)
        k = run_rope(d_k // num_heads, theta, seq_len, k, token_positions)

    # casual mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    attn_out = run_scaled_dot_product_attention(q, k, v, mask) # B, nh, T, d_v // nh

    attn_out = attn_out.permute(0, 2, 1, 3) # B, T, nh, d_v // nh
    attn_out = attn_out.contiguous()
    attn_out = attn_out.view(attn_out.shape[:-2] + (d_v,))    # B, T, d_v

    out = run_linear(d_v, d_model, o_proj_weight, attn_out)

    return out


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.
    
    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    seq_len = in_query_or_key.shape[-2]
    # 创建频率
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))    # (d_k // 2,)
    
    # 扩展 freqs 以匹配最大序列长度
    freqs = freqs.unsqueeze(0).expand(max_seq_len, -1)  # (max_seq_len, d_k // 2)
    
    # 创建位置索引
    t = torch.arange(max_seq_len).unsqueeze(1)          # (max_seq_len, 1)
    
    # 计算嵌入
    emb = freqs * t                                     # (max_seq_len, d_k // 2)
    
    # 创建旋转矩阵
    cos_emb = torch.cos(emb)                            # (max_seq_len, d_k // 2)
    sin_emb = torch.sin(emb)                              # (max_seq_len, d_k // 2)

    cos_emb, sin_emb = cos_emb[:seq_len], sin_emb[:seq_len]
    
    # 将 RoPE 应用于输入张量
    # 拆分输入为偶数和奇数维度
    x_even = in_query_or_key[..., ::2]                  # (..., seq_len, d_k // 2)
    x_odd = in_query_or_key[..., 1::2]                  # (..., seq_len, d_k // 2)
    
    # 计算旋转后的表示
    rotated_even = x_even * cos_emb - x_odd * sin_emb
    rotated_odd = x_even * sin_emb + x_odd * cos_emb
    
    # 重新组合维度
    result = torch.zeros_like(in_query_or_key)
    result[..., ::2] = rotated_even
    result[..., 1::2] = rotated_odd
    
    return result




def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    # 获取 token 位置
    batch_size, seq_length, _ = in_features.shape
    token_positions = torch.arange(seq_length).expand(batch_size, -1)
    # token_positions = torch.arange(seq_length)
    
    # 应用第一个 RMSNorm
    ln1_weight = weights['ln1.weight']
    normed_input = run_rmsnorm(d_model, 1e-5, ln1_weight, in_features)
    
    # 获取 Q、K、V 投影权重
    q_proj_weight = weights['attn.q_proj.weight']
    k_proj_weight = weights['attn.k_proj.weight']
    v_proj_weight = weights['attn.v_proj.weight']
    
    # 运行多头自注意力（带 RoPE）
    attn_output = run_multihead_self_attention_with_rope(
        d_model, num_heads, seq_length, theta,
        q_proj_weight, k_proj_weight, v_proj_weight,
        weights['attn.output_proj.weight'],
        normed_input,
        token_positions
    )
    
    # 第一个残差连接
    residual = in_features + attn_output
    
    # 第二个 RMSNorm
    ln2_weight = weights['ln2.weight']
    normed_attn_output = run_rmsnorm(d_model, 1e-5, ln2_weight, residual)
    
    # 运行 FFN（SwiGLU）
    ffn_output = run_swiglu(
        d_model, d_ff,
        weights['ffn.w1.weight'],
        weights['ffn.w2.weight'],
        weights['ffn.w3.weight'],
        normed_attn_output
    )
    
    # 第二个残差连接
    output = residual + ffn_output
    
    return output


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    batch_size, seq_length = in_indices.shape
    
    # 获取 token 嵌入
    token_embeddings = run_embedding(vocab_size, d_model, weights['token_embeddings.weight'], in_indices)
    
    # 处理每个Transformer层
    hidden_state = token_embeddings
    for i in range(num_layers):
        # 构建当前层的权重字典
        layer_weights = {}
        for k, v in weights.items():
            if k.startswith(f'layers.{i}.'):
                layer_weights[k] = v

        # remove layers.{i} in front of key
        layer_weights.update({
            k[len(f'layers.{i}.'):]: v for k, v in layer_weights.items()
        })
        
        # 应用当前Transformer层
        hidden_state = run_transformer_block(
            d_model, num_heads, d_ff, context_length, rope_theta,
            layer_weights, hidden_state
        )
    
    # 最终层归一化
    ln_final_weight = weights['ln_final.weight']
    hidden_state = run_rmsnorm(d_model, 1e-6, ln_final_weight, hidden_state)
    
    # 语言模型头部 - 将隐藏状态映射到词汇表大小
    lm_logits = run_linear(vocab_size, d_model, weights['lm_head.weight'], hidden_state)
    
    return lm_logits


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    # return in_features / torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps)
    x = in_features
    square = in_features ** 2
    mean = torch.mean(square, dim=-1, keepdim=True)
    rms = torch.sqrt(mean + eps)
    return x / rms * weights



def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features * torch.sigmoid(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # 从数据集中随机选择起始位置
    starts = np.random.randint(
        low=0,
        high=len(dataset) - context_length,
        size=(batch_size,)
    )
    
    # 根据起始位置获取输入序列和对应的标签
    inputs = np.stack([dataset[start:start+context_length] for start in starts])
    labels = np.stack([dataset[start+1:start+context_length+1] for start in starts])
    
    # 将numpy数组转换为torch张量，并移动到指定设备
    return (
        torch.from_numpy(inputs).long().to(device),
        torch.from_numpy(labels).long().to(device)
    )



def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return torch.softmax(in_features, dim)


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Log-softmax for numerical stability (log(softmax(inputs)))
    log_softmax = inputs - inputs.logsumexp(dim=1, keepdim=True)
    
    # Gather the log probabilities of the target classes
    # Equivalent to: log_softmax[range(N), targets]
    target_log_probs = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    # Negative mean of the target log probabilities
    loss = -target_log_probs.mean()
    
    return loss


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # Filter parameters with gradients
    parameters_with_grad = [p for p in parameters if p.grad is not None]
    
    if len(parameters_with_grad) == 0:
        return
    
    # Calculate total L2 norm of all gradients
    total_norm = torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in parameters_with_grad))
    
    # Calculate clipping coefficient
    clip_coef = max_l2_norm / (total_norm + 1e-6)  # Add small value to avoid division by zero
    
    # If total norm exceeds max_norm, scale down all gradients
    if clip_coef < 1.0:
        for p in parameters_with_grad:
            p.grad.mul_(clip_coef)

def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    class AdamW(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
            if not 0.0 <= lr:
                raise ValueError(f"Invalid learning rate: {lr}")
            if not 0.0 <= eps:
                raise ValueError(f"Invalid epsilon value: {eps}")
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            
            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            super(AdamW, self).__init__(params, defaults)

        def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    # Perform stepweight decay
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                    # Get parameter-specific state
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']
                    step_size = group['lr']
                    eps = group['eps']

                    # Update state
                    state['step'] += 1
                    grad = p.grad.data

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    denom = exp_avg_sq.sqrt().add_(eps)
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

            return loss
    
    return AdamW

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        # Warm-up 阶段：线性增加学习率
        lr = (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        # Cosine Annealing 阶段：余弦函数衰减
        t = it - warmup_iters
        T = cosine_cycle_iters - warmup_iters
        cos_value = np.cos(np.pi * t / T)
        lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + cos_value)
    else:
        # Post-annealing 阶段：学习率保持最小值
        lr = min_learning_rate

    return lr

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    if isinstance(out, (str, os.PathLike)):
        with open(out, 'wb') as f:
            torch.save(checkpoint, f)
    else:
        torch.save(checkpoint, out)

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    if isinstance(src, (str, os.PathLike)):
        with open(src, 'rb') as f:
            checkpoint = torch.load(f)
    else:
        checkpoint = torch.load(src)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']


import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Union, Iterable, Optional

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)



class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a BPE tokenizer from a given vocabulary, list of merges, and (optionally) special tokens.
        
        Args:
            vocab: A dictionary mapping token IDs to their byte representations.
            merges: A list of tuples representing BPE merge operations.
            special_tokens: Optional list of strings that should be treated as unbreakable tokens.
        """
        self.vocab = vocab
        self.byte_to_token_id = {v: k for k, v in vocab.items()}
        self.merges = merges

        self.bpe_ranks = dict(zip(merges, range(len(merges))))
            
        # Handle special tokens
        self.special_tokens = special_tokens or []
        self.special_token_bytes = [token.encode("utf-8") for token in self.special_tokens]
        
        # Ensure special tokens are in the vocabulary
        for token_bytes in self.special_token_bytes:
            if token_bytes not in self.byte_to_token_id:
                # Add to vocab if not already present
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.byte_to_token_id[token_bytes] = new_id

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text string into a sequence of token IDs.
        
        Args:
            text: The input text to encode.
            
        Returns:
            A list of integer token IDs representing the encoded text.
        """
        tokens = []

        # Sort special tokens by length (longest first) to avoid partial matches
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_special_tokens))
        if pattern:
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                # If it's a special token, add its ID directly
                tokens.append(self.byte_to_token_id[part.encode("utf-8")])
            else:
                # Otherwise, tokenize normally using BPE
                tokens.extend(self._tokenize_normal(part))

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> iter:
        """
        Given an iterable of strings (e.g., a file handle), yield token IDs lazily.
        
        Args:
            iterable: An iterable source of text chunks.
            
        Yields:
            Token IDs generated by processing the input iterable.
        """
        for chunk in iterable:
            yield from self.encode(chunk)


    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into a human-readable string.
        
        Args:
            ids: A list of integer token IDs.
            
        Returns:
            The decoded string representation of the input token IDs.
        """
        # Concatenate all token bytes
        full_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        
        # Decode bytes to string, replacing invalid sequences
        return full_bytes.decode("utf-8", errors="replace")

    def _tokenize_normal(self, text: str) -> list[int]:
        """
        Tokenize a normal piece of text (not a special token) into token IDs.
        
        Args:
            text: A string to tokenize.
            
        Returns:
            A list of token IDs representing the tokenized text.
        """
        # Pre-tokenization
        pre_tokens = []
        for m in re.finditer(PAT, text):
            word = m.group(0)
            pre_tokens.append(word)

        token_ids = []
        for token in pre_tokens:
            # Convert token to bytes tuple
            byte_tuple = to_bytes_tuple(token)
            
            # Apply BPE merges
            merged = self._apply_merges(byte_tuple)
            
            # Get token IDs
            token_ids.extend(self.byte_to_token_id[b] for b in merged)
        
        return token_ids

    def _apply_merges(self, byte_tuple: tuple[bytes, ...]) -> list[bytes]:
        """
        Apply BPE merges to a sequence of bytes.
        
        Args:
            byte_tuple: A tuple of single-byte tokens.
            
        Returns:
            A list of merged byte tokens after applying all applicable merges.
        """
        word: list[bytes] = list(byte_tuple)

        def get_pairs(word: list[bytes]):
            pairs = set()
            prev_char = word[0]
            for char in word[1:]:
                pairs.add((prev_char, char))
                prev_char = char
            return pairs
        
        pairs = get_pairs(word)

        if not pairs:
            return word

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        return word


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # Step 1: Initialize Vocabulary
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    for token_bytes in special_token_bytes:
        if token_bytes not in vocab.values():
            vocab[next_id] = token_bytes
            next_id += 1

    # Step 2: Pre-tokenization
    pre_tokens_cnt = defaultdict(int)

    def to_bytes_tuple(word: str) -> Tuple[bytes]:
        l = list(tuple(word.encode("utf-8")))
        l = [bytes([x]) for x in l]
        return tuple(l)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = re.split("|".join(map(re.escape, special_tokens)), text)
    
    for chunk in chunks:
        for m in re.finditer(PAT, chunk):
            word = m.group(0)
            pre_tokens_cnt[to_bytes_tuple(word)] += 1   # key of pre_tokens_cnt e.g. (b'H', b'e', b'l', b'l', b'o')

    # Step 3: Compute BPE Merges
    merges = []

    while len(vocab) < vocab_size:
        pair_counts = defaultdict(int)

        # Count all adjacent byte pairs
        for token, cnt in pre_tokens_cnt.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counts[pair] += cnt

        if not pair_counts:
            break  # No more pairs to merge

        # Find the most frequent pair(s)
        max_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        best_pair = max(candidates)

        a, b = best_pair

        # Create new token
        new_token = a + b
        vocab[next_id] = new_token
        next_id += 1

        # Apply the merge to all pre-tokenized sequences
        # 收集变更
        changes = []
        for token, cnt in pre_tokens_cnt.items():
            # Find all occurrences of the `best_pair` in `token`
            indices = [i for i in range(len(token) - 1) if token[i:i + 2] == best_pair]
            if indices:
                # Replace each occurrence with `new_token`
                new_pre_token = []
                i = 0
                while i < len(token):
                    if i in indices:
                        new_pre_token.append(new_token)
                        i += 2
                    else:
                        new_pre_token.append(token[i])
                        i += 1
                new_pre_token = tuple(new_pre_token)
                changes.append((token, new_pre_token, cnt))

        # 应用变更
        for old_token, new_pre_token, cnt in changes:
            pre_tokens_cnt[new_pre_token] = pre_tokens_cnt.get(new_pre_token, 0) + cnt
            del pre_tokens_cnt[old_token]

        # Record the merge
        merges.append((a, b))

    return vocab, merges
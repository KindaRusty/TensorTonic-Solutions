import torch

def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns scores of shape (batch, heads, query_length, key_length).
    """
    B, S_q, D = q.shape 
    _, S_k, _ = k.shape
    H = num_heads
    # Head size
    d_h = D // H 
    # 1. Split D into (H, d_h)
    q_reshaped = q.reshape(B, S_q, H, d_h)
    k_reshaped = k.reshape(B, S_k, H, d_h)
    # 2. Permutate to put H onward S
    # Input: (Batch, Sequence, Head, d_h)
    # Output: (Batch, Head, Sequence, d_h)
    q_heads = q_reshaped.transpose(1, 2)
    k_heads = k_reshaped.transpose(1, 2)
    # 3. Compute dot product by Einsum
    # 'd' axis appear in input but not in output -> get summarization
    dot_product = torch.einsum('bhid,bhjd->bhij', q_heads, k_heads)
    # 4. Divide by square root of d_h for normalization
    scores = dot_product / math.sqrt(d_h)
    return scores

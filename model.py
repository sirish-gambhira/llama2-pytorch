import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        xrms = torch.sqrt(torch.pow(x, 2).mean(dim=-1) + self.eps).unsqueeze(-1) # (batch, seq len, 1)
        return (x / xrms) * self.weight # (b, seq len, dim) * (dim)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32
    vocab_size: int = -1
    multiple_of: int = 256
    ff_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    # KV cache settings
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None

def precompute_theta_pos_freq(d_head : int, seq_len: int, device: str = None, theta: float = 10000.0):
    assert d_head % 2 == 0, "Head dim must be divisible by 2"
    theta_exp = torch.arange(0, d_head, 2).float()
    theta_val = 1 / (theta ** (theta_exp / d_head)).to(device) # [1, d / 2]
    m = torch.arange(seq_len, device=device).float() # [1, m]
    freq = torch.outer(m, theta_val) # [m, d / 2]
    freq_complex = torch.polar(torch.ones_like(freq), freq) # [m, d / 2]
    return freq_complex
    
def apply_rotary_embeddings(x : torch.Tensor, freqs_complex: torch.Tensor, device: str):
    batch_size, seq_len, n_heads, d_head = x.shape
    x_grp = x.view(batch_size, seq_len, n_heads, -1, 2) # [b, seq len, h, d] => [b, seq len, h, d / 2, 2]
    x_complex = torch.complex(x_grp[:, :, :, :, 0], x_grp[:, :, :, :, 1]).to(device) # [b, seq len, h, d / 2]
    prod = x_complex * freqs_complex.unsqueeze(0).unsqueeze(2) # [b, seq len, h, d / 2] * [1, seq len, 1, d / 2]
    out = torch.view_as_real(prod) # [b, seq len, h, d / 2, 2]
    out = out.reshape(*x.shape).type(torch.float16) # [b, seq len, h, d]
    return out

class FeedForwardBlock(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ff_dim_multiplier is not None:
            hidden_dim = int(args.ff_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
    
    def forward(self, x):
        swish = F.silu(self.w1(x))
        x3 = self.w3(x)
        x = swish * x3
        return self.w2(x)
        
class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.num_rep = args.n_heads // args.n_kv_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False) # [b, seq len, num kv heads, self.head_dim]
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
    
    def forward(self, x, start_pos, freq_complex):
        batch_size, seq_len, dim = x.shape # (b, 1, dim)
        assert seq_len == 1, "Sequence length must be 1"
        
        xq = self.wq(x) # (B, 1, num q heads * head dim)
        xk = self.wk(x) # (B, 1, num kv heads * head dim)
        xv = self.wv(x) # (B, 1, num kv heads * head dim)
        
        xq = xq.view(x.shape[0], x.shape[1], self.n_heads, self.head_dim)
        xk = xk.view(x.shape[0], x.shape[1], self.n_kv_heads, self.head_dim)
        xv = xv.view(x.shape[0], x.shape[1], self.n_kv_heads, self.head_dim)
        
        xq = apply_rotary_embeddings(xq, freq_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freq_complex, device=x.device)
        
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv
        
        # retrieve all tokens from KV cache
        keys = self.cache_k[:batch_size, :start_pos+seq_len] # [batch size, seq len, num kv heads, head dim]
        values = self.cache_v[:batch_size, :start_pos+seq_len]
        
        keys = repeat_kv(keys, self.num_rep) # [batch size, seq len, num kv heads * num repeat, head_dim]
        values = repeat_kv(keys, self.num_rep) # [batch size, seq len, num kv heads * num repeat, head_dim]

        # remove head dim - each head will observe all sequence but only part of the embedding
        xq = xq.transpose(1, 2) # [batch size, n_heads, seq len, head_dim]
        keys = keys.transpose(1, 2) #[batch size, n_heads, seq len, head dim]
        values = values.transpose(1, 2)
        
        scores = xq @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = scores @ values # [batch, num q heads, seq len, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # [batch, seq len, n_heads * head_dim]
        output = self.wo(output) # [batch, seq len, self.dim]
        return output

def repeat_kv(x, num_repeat):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if num_repeat == 1:
        return x
    else:
        return (
            # [B, seq len, num kv heads, head dim]
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, num_repeat, head_dim)
            .reshape(*x.shape[:2], -1, x.shape[-1])
            # [B, seq len, n_kv_heads * num_repeast, head_dim]
        )

class EncoderBlock(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.n_heads = args.n_heads
        assert args.dim % args.n_heads == 0, "Dimension should be divisible by num heads"
        self.d_head = self.dim // self.n_heads
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForwardBlock(args)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
    
    def forward(self, x : torch.Tensor, start_pos: int, freq_complex: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), start_pos, freq_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
        
class Transformer(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.args = args
        self.dim = args.dim
        assert args.vocab_size != -1, "Vocab size should be set"
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(dim=args.dim, eps = args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.freqs_complex = precompute_theta_pos_freq(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)

    
    def forward(self, tokens, start_pos):
        # input is one token at a time
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token processed at a time"
        
        h = self.tok_embeddings(tokens)
        
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]
        
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h)
        out = self.output(h).float()
        return out


         
    

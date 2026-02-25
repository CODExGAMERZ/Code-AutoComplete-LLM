import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, past_kv=None):
        B, T, C = x.size()

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        seq_len_q = att.size(-2)
        seq_len_k = att.size(-1)

        mask = torch.tril(
            torch.ones(seq_len_q, seq_len_k, device=x.device)
        ).bool()

        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)

        return out, (k, v)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_heads)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x, past_kv=None):
        attn_out, kv = self.attn(self.ln1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, kv


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layers=8, n_heads=8, n_embd=512):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_heads)
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        self.block_size = block_size

    def forward(self, x, past_kvs=None):
        B, T = x.shape

        if past_kvs is not None:
            past_length = past_kvs[0][0].size(2)
        else:
            past_length = 0

        pos = torch.arange(
            past_length,
            past_length + T,
            device=x.device
        ).unsqueeze(0)

        x = self.token_emb(x) + self.pos_emb(pos)

        new_kvs = []

        for i, block in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            x, kv = block(x, past)
            new_kvs.append(kv)

        x = self.ln(x)
        logits = self.head(x)

        return logits, new_kvs
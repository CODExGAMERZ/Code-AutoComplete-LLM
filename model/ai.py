import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layers=8, n_heads=8, n_embd=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_heads,
                dim_feedforward=4 * n_embd,
                activation="gelu",
                batch_first=True
            )
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device)

        x = self.token_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)

        for block in self.blocks:
            x = block(x, src_mask=mask)

        x = self.ln(x)
        return self.head(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== Hyperparameters ======
EMB_DIM = 128      # piece embedding dim
N_HEAD = 8
N_LAYERS = 4
VOCAB_SIZE = 4672  # total possible moves
BOARD_SIZE = 8


# ====== 2D Positional Embedding ======
class PositionalEmbedding2D(nn.Module):
    def __init__(self, emb_dim, height=8, width=8):
        super().__init__()
        self.row_embed = nn.Parameter(torch.randn(height, emb_dim))
        self.col_embed = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, x):
        """
        x: (batch, 64, emb_dim)
        returns: (batch, 64, emb_dim) with added 2D positional info
        """
        B, N, D = x.shape
        assert N == 64, "Input must have 64 tokens (8x8 board)"
        # Create (8x8, emb_dim) positional grid
        pos = torch.zeros(BOARD_SIZE, BOARD_SIZE, D, device=x.device)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                pos[i, j] = self.row_embed[i] + self.col_embed[j]
        pos = pos.view(1, 64, D)  # (1, 64, emb_dim)
        return x + pos


# ====== Transformer Block ======
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, int(mlp_ratio * emb_dim)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * emb_dim), emb_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


# ====== GPT-like Chess Model ======
class ChessGPT(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, n_layers=N_LAYERS, n_heads=N_HEAD, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.pos_embed = PositionalEmbedding2D(emb_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(emb_dim, n_heads) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(emb_dim)
        self.policy_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        """
        x: (batch, 8, 8, emb_dim)
        returns: (batch, vocab_size) logits
        """
        B = x.size(0)
        x = x.view(B, 64, -1)  # flatten board
        x = self.pos_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)

        # Pool board representation (mean pooling)
        x = x.mean(dim=1)  # (batch, emb_dim)

        logits = self.policy_head(x)
        return logits

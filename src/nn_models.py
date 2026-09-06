"""PyTorch Tabular Neural Network Architectures (Tabular ResNet).

Based on Gorishniy et al. (NeurIPS 2021) "Revisiting Deep Learning Models for Tabular Data".
Combines continuous numerical features with learned categorical entity embeddings
and multi-layer residual blocks with LayerNorm, GELU, Dropout, and skip connections.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    """Residual block for tabular data: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout -> Skip."""
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.norm(x)
        z = self.linear1(z)
        z = self.act(z)
        z = self.dropout1(z)
        z = self.linear2(z)
        z = self.dropout2(z)
        return residual + z


class TabularResNet(nn.Module):
    """Tabular ResNet with categorical entity embeddings and residual blocks."""
    def __init__(
        self,
        num_features: int,
        cat_cardinalities: list[int],
        embedding_dims: list[int] = None,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        if embedding_dims is None:
            embedding_dims = [
                min(16, max(4, int(1.6 * (card ** 0.56))))
                for card in cat_cardinalities
            ]
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, dim)  # +1 for unknown/unseen categories
            for card, dim in zip(cat_cardinalities, embedding_dims)
        ])
        
        total_emb_dim = sum(embedding_dims)
        in_dim = num_features + total_emb_dim
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x_num: Float tensor of shape (batch, num_features)
            x_cat: Long tensor of shape (batch, num_cats)
        
        Returns:
            Logits of shape (batch,)
        """
        if len(self.embeddings) > 0 and x_cat is not None:
            emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat([x_num] + emb_outs, dim=1)
        else:
            x = x_num
            
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.head_norm(h)
        return self.head(h).squeeze(-1)

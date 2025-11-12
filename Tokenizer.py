import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

class ChessBoardTokenizer(nn.Module):
    """
    A tokenizer module that converts a single FEN string representation of a 
    chess board into a continuous embedded tensor, ready for an LLM input 
    sequence (without the batch dimension).
    """
    
    PIECE_TO_INDEX: Dict[str, int] = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11, # Black pieces
        '.': 12  # Empty square
    }
    
    def __init__(self, emb_dim: int = 128):
        """Initializes the tokenizer and its embedding layer."""
        super().__init__()
        self.emb_dim = emb_dim
        self.num_piece_types = len(self.PIECE_TO_INDEX)
        
        # Maps a piece index (0-12) to a high-dimensional vector (128)
        self.piece_embedding = nn.Embedding(self.num_piece_types, self.emb_dim)

    def _fen_to_index_matrix(self, fen: str) -> torch.Tensor:
        """
        Convert FEN string to an 8x8 integer matrix of piece indices.
        Returns tensor shape (8, 8).
        """
        piece_placement = fen.split()[0]
        rows = piece_placement.split('/')
        board_idx = np.zeros((8, 8), dtype=int)
        empty_index = self.PIECE_TO_INDEX['.']

        for r, row in enumerate(rows):
            file = 0
            for ch in row:
                if ch.isdigit():
                    file += int(ch)
                else:
                    board_idx[r, file] = self.PIECE_TO_INDEX.get(ch, empty_index)
                    file += 1
                
            while file < 8:
                board_idx[r, file] = empty_index
                file += 1
                
        return torch.tensor(board_idx, dtype=torch.long)

    def flatten_board(self, board_emb: torch.Tensor) -> torch.Tensor:
        """
        Flattens the 8x8 spatial dimensions into a single sequence dimension of 64.
        
        Input shape: (R, C, D) -> (8, 8, 128)
        Output shape: (R*C, D) -> (64, 128)
        """
        # Get the Rank (R), Column (C), and Embedding dimension (D)
        R, C, D = board_emb.shape
        
        # Reshape the tensor to (64, D). The DataLoader will stack these to (B, 64, D).
        return board_emb.view(R * C, D)
        
    def forward(self, fen: str, flatten: bool = True) -> torch.Tensor:
        """
        Convert FEN string to the embedded board tensor.
        
        Args:
            fen: The FEN string of the current board state.
            flatten: If True, returns a (64, emb_dim) sequence.
                     If False, returns a (8, 8, emb_dim) spatial grid.
            
        Returns:
            The embedded board tensor without the batch dimension.
        """
        
        # 1. Convert FEN to 8x8 matrix of indices. Shape: (8, 8)
        # Removed .unsqueeze(0)
        board_idx = self._fen_to_index_matrix(fen)
        
        # 2. Look up the learned embedding vector for each index. Shape: (8, 8, emb_dim)
        # No batch dimension is added here
        board_emb = self.piece_embedding(board_idx)
        
        if flatten:
            return self.flatten_board(board_emb)
        else:
            return board_emb

# === Example usage ===
if __name__ == "__main__":
    EMBEDDING_DIM = 128
    tokenizer = ChessBoardTokenizer(emb_dim=EMBEDDING_DIM)
    fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    # 1. Get the flattened sequence output (Ready for LLM)
    board_sequence = tokenizer(fen_start, flatten=True)
    
    print("--- Flattened Output (Ready for LLM) ---")
    # Corrected output shape printout: (Sequence Length, Embedding Dim)
    print(f"Shape: {board_sequence.shape} (Sequence Length, Embedding Dim)")
    
    # 2. Get the 3D spatial grid (if needed for CNNs)
    board_grid = tokenizer(fen_start, flatten=False)
    
    print("\n--- Spatial Grid Output (Before Flattening) ---")
    # Corrected output shape printout: (Rank, File, Embedding Dim)
    print(f"Shape: {board_grid.shape} (Rank, File, Embedding Dim)")
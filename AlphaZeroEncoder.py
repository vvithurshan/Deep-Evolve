import chess
import itertools

class AlphaZeroMoveEncoder:
    """
    Manages the encoding and decoding of chess moves using the 
    AlphaZero-style 64 squares x 73 move-type fixed action space.
    """
    def __init__(self):
        """Initializes the vocabulary upon class creation."""
        self.move_vocab = self._create_move_vocabulary()
        self.VOCAB_SIZE = len(self.move_vocab)

        # Build reverse lookup for faster encoding
        self._reverse_vocab = {v: k for k, v in self.move_vocab.items()}

    # --- Static Utility Methods ---
    @staticmethod
    def _square_to_coords(square_idx):
        """0–63 → (file, rank)"""
        # (file index 0-7, rank index 0-7)
        return square_idx % 8, square_idx // 8

    @staticmethod
    def _coords_to_square(file, rank):
        """(file, rank) → 0–63"""
        # Bounds check is omitted for simplicity but necessary in production code
        return rank * 8 + file

    # --- Core Vocabulary Builder ---
    def _create_move_vocabulary(self):
        """
        Builds the 4672-entry AlphaZero-style move vocabulary.
        Each move = (from_square_idx, move_type_tuple)
        """
        move_vocab = {}
        move_id = 0

        # 1. Sliding Directions (8)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), # Rooks/Queens
                      (-1, -1), (-1, 1), (1, -1), (1, 1)] # Bishops/Queens
        
        move_types = []
        # Sliding Moves (56 = 8 directions * 7 distances)
        for dx, dy in directions:
            for dist in range(1, 8):
                move_types.append((dx, dy, dist)) # (dx, dy, distance)

        # 2. Knight Moves (8)
        knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                        (-2, -1), (-1, -2), (1, -2), (2, -1)]
        for (dx, dy) in knight_moves:
            move_types.append((dx, dy, 1)) # (dx, dy, distance=1 for knight)

        # 3. Promotions (9 = 3 pieces × 3 directions)
        promotion_pieces = ['q', 'r', 'b']
        # The 3 promotion directions (relative to a pawn's movement)
        promotion_dirs = [(0, 1), (-1, 1), (1, 1)]
        for piece, (dx, dy) in itertools.product(promotion_pieces, promotion_dirs):
            move_types.append((dx, dy, piece)) # (dx, dy, piece_symbol)

        # Combine 64 squares with 73 move types
        for from_sq in range(64):
            for mtype in move_types:
                move_vocab[move_id] = (from_sq, mtype)
                move_id += 1

        assert move_id == 4672
        return move_vocab

    # --- Public Encoding/Decoding Methods ---
    def encode(self, move_uci: str) -> int | None:
        """
        Encodes a standard UCI move (e.g., 'e2e4', 'a7a8q') into its 
        corresponding policy head ID (0-4671).
        """
        move = chess.Move.from_uci(move_uci)
        from_sq, to_sq = move.from_square, move.to_square

        fx, fy = self._square_to_coords(from_sq)
        tx, ty = self._square_to_coords(to_sq)
        dx, dy = tx - fx, ty - fy

        # Determine the promotion piece symbol ('q', 'r', 'b', 'n') if applicable
        promo_sym = chess.Piece(move.promotion, chess.WHITE).symbol().lower() if move.promotion else None

        # Build the target move key (src, mtype)
        if promo_sym in ['q', 'r', 'b']: # Promotions
            # For promotions, we only care about the direction (dx, dy) and the piece
            target_key = (from_sq, (dx, dy, promo_sym))
        else: # Normal moves
            # Calculate distance using L-infinity norm (max of abs(dx), abs(dy))
            dist = max(abs(dx), abs(dy))
            
            # If dist is 0 (pass move) or the move is invalid (e.g. not straight/diagonal/knight), skip
            if dist == 0:
                return None
            
            # Normalize direction vector to find unit vector (ddx, ddy)
            # This is key for matching the vocabulary structure!
            ddx = dx // dist
            ddy = dy // dist
            
            target_key = (from_sq, (ddx, ddy, dist))

        # Look up the ID using the reverse vocabulary
        return self._reverse_vocab.get(target_key)


    def decode(self, move_id: int) -> str:
        """
        Decodes a policy head ID (0-4671) back into a UCI move string.
        """
        from_sq, mtype = self.move_vocab[move_id]
        fx, fy = self._square_to_coords(from_sq)
        dx, dy, step = mtype

        # Handle promotions
        if isinstance(step, str): # The step is a promotion piece symbol ('q', 'r', 'b')
            promo_piece_sym = step
            # dx and dy already encode the final delta (e.g., (0, 1) for straight)
            to_sq = self._coords_to_square(fx + dx, fy + dy)
            promotion_type = chess.Piece.from_symbol(promo_piece_sym).piece_type
        else: # Normal moves (Sliding or Knight)
            promo_piece_sym = None
            # Multiply the unit vector (dx, dy) by the distance (step)
            final_dx, final_dy = dx * step, dy * step
            to_sq = self._coords_to_square(fx + final_dx, fy + final_dy)
            promotion_type = None
        
        # Create the chess.Move object
        move = chess.Move(from_sq, to_sq, promotion=promotion_type)
        return move.uci()

# === Example usage ===
if __name__ == "__main__":
    encoder = AlphaZeroMoveEncoder()
    print(f"Encoder initialized. Vocabulary Size: {encoder.VOCAB_SIZE}")
    print("-" * 35)

    # 1. Encode Example (Normal Move)
    move_uci_1 = "e2e4"
    move_id_1 = encoder.encode(move_uci_1)
    print(f"Encoding {move_uci_1} → ID {move_id_1}")

    decoded_move_1 = encoder.decode(move_id_1)
    print(f"Decoding ID {move_id_1} → Move {decoded_move_1}")
    print("-" * 35)

    # 2. Encode Example (Capture/Sliding Move)
    move_uci_2 = "a1a8" # Rook move 7 squares North
    move_id_2 = encoder.encode(move_uci_2)
    print(f"Encoding {move_uci_2} → ID {move_id_2}")

    decoded_move_2 = encoder.decode(move_id_2)
    print(f"Decoding ID {move_id_2} → Move {decoded_move_2}")
    print("-" * 35)

    # 3. Encode Example (Promotion Move)
    # The promotion piece symbol must be one of 'q', 'r', 'b' as defined in the vocab
    move_uci_3 = "a7b8q" 
    move_id_3 = encoder.encode(move_uci_3)
    print(f"Encoding {move_uci_3} → ID {move_id_3}")
    
    decoded_move_3 = encoder.decode(move_id_3)
    print(f"Decoding ID {move_id_3} → Move {decoded_move_3}")
    print("-" * 35)
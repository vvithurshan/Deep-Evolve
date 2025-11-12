from Tokenizer import ChessBoardTokenizer
from AlphaZeroEncoder import AlphaZeroMoveEncoder
import chess
import random
import torch
from datasets import Dataset  
from tqdm import tqdm
from model import ChessGPT
from torch.utils.data import Dataset as TorchDataset, DataLoader

torch.manual_seed(42)
random.seed(42)
# ------------------------
# 1. Setup
# ------------------------
EMBEDDING_DIM = 512
tokenizer = ChessBoardTokenizer(emb_dim=EMBEDDING_DIM)
encoder = AlphaZeroMoveEncoder()

def algebraic_to_uci(board_fen, algebraic_move):
    board = chess.Board(board_fen)
    move = board.parse_san(algebraic_move)
    return move.uci()

def random_fen():
    board = chess.Board()
    for _ in range(random.randint(0, 20)):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    return board.fen()

def legal_moves(fen):
    board = chess.Board(fen)
    return [board.san(m) for m in board.legal_moves]

def create_random_dataset(n=5000):
    data = []
    for _ in tqdm(range(n), desc="Generating FENs"):
        fen = random_fen()
        moves = legal_moves(fen)
        for move in moves:
            data.append({"input": fen, "output": move})
            break
    # ✅ FIX: Use the class method, not instance call
    return Dataset.from_list(data)

dataset = create_random_dataset()

# ------------------------
# 2. Tokenization + Encoding
# ------------------------
Inputs, Outputs = [], []
for i in range(len(dataset)):
    input_tensor = tokenizer(dataset[i]['input'])
    output_tensor = encoder.encode(algebraic_to_uci(dataset[i]['input'], dataset[i]['output']))

    # Ensure clean tensors detached from any prior graph
    if not torch.is_tensor(input_tensor):
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    else:
        input_tensor = input_tensor.detach().clone().float()

    if not torch.is_tensor(output_tensor):
        output_tensor = torch.tensor(output_tensor, dtype=torch.long)
    else:
        output_tensor = output_tensor.detach().clone().long()

    Inputs.append(input_tensor)
    Outputs.append(output_tensor)

# ------------------------
# 3. PyTorch Dataset + Dataloader
# ------------------------
class MyDataset(TorchDataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

my_dataset = MyDataset(Inputs, Outputs)
dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

# ------------------------
# 4. Model, Optimizer, Loss
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessGPT(emb_dim=EMBEDDING_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# ------------------------
# 5. Training Loop
# ------------------------
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx+1}/{len(dataloader)}] | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] finished. Avg Loss: {avg_loss:.4f}")

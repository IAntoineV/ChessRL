import torch
import chess
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import os
import wandb
from datetime import datetime
from models.seq_model import SeqModel, SeqModelConfig
from data_process.vocab import policy_index, get_hash_table_vocab_to_index
from data_process.parse import dir_iterator_fen_move, ParsingConfigFenMove, encode_fens
from data_process.fen_encoder import fen_to_tensor
from data_process.legal_mask import get_legal_move_mask
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup


load_dotenv()
# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
seq_len = 64
input_dim = 19

#Wandb key
wandb_key = "YOUR KEY"

# Model configuration
config_model = SeqModelConfig(output_dim=len(policy_index), input_dim=input_dim, n_embd=1200, seq_len=seq_len, n_layer=6)
encoding_model = SeqModel(config_model).to(device)

lr = 2e-4
weight_decay = 0.01
num_training_steps = 100_000
num_warmup_steps = int(0.01 * num_training_steps)  # 1% warm-up

opt = AdamW(encoding_model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps, num_training_steps)

# Mixed precision scaler
scaler = torch.amp.GradScaler(device)

# Training parameters
batch_size = 320
grad_accumulation_steps = 4
num_to_sample_per_game = 10
num_it = 10000
dir_pgn_path = "./pgn_data"
metrics_freq = 100
max_grad_norm = 1.0  # Gradient clipping
training_config = dict(batch_size = batch_size,
grad_accumulation_steps = grad_accumulation_steps,
num_to_sample_per_game = num_to_sample_per_game,
num_it = num_it,
dir_pgn_path = dir_pgn_path,
metrics_freq = metrics_freq,
max_grad_norm = max_grad_norm  # Gradient clipping
)

# Logging setup
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./logs/{time_stamp}"
os.makedirs(log_dir, exist_ok=True)

print(f"Device: {device}")
num_params_learnable = sum(p.numel() for p in encoding_model.parameters() if p.requires_grad)
num_params = sum(p.numel() for p in encoding_model.parameters())
print(f"Total Params: {num_params} | Learnable Params: {num_params_learnable}")

# Data setup
hash_table_vocab_to_index = get_hash_table_vocab_to_index()
parsing_config = ParsingConfigFenMove(batch_size=batch_size, num_to_sample=num_to_sample_per_game)

config = {**config_model.__dict__, **parsing_config.__dict__, **training_config}

# Initialize Weights & Biases (wandb)
wandb.login(key=wandb_key)
wandb.init(project="chessRL", name=f"run_{time_stamp}", config=config)

generator = dir_iterator_fen_move(dir_pgn_path, config=parsing_config)
progress_bar = tqdm(range(num_it))

# Training loop
for it in progress_bar:
    fens, moves = next(generator)
    targets = torch.tensor([hash_table_vocab_to_index[move] for move in moves], device=device)
    inputs = encode_fens(fens).to(device).view(batch_size, seq_len, input_dim)

    # Mixed precision forward pass
    with torch.amp.autocast(device):
        logits = encoding_model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)

    # Backpropagation with mixed precision
    scaler.scale(loss).backward()

    if (it + 1) % grad_accumulation_steps == 0:  # Gradient accumulation
        scaler.unscale_(opt)  # Unscales before clipping
        torch.nn.utils.clip_grad_norm_(encoding_model.parameters(), max_grad_norm)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        scheduler.step()

    acc = (torch.argmax(logits, dim=-1) == targets).float().mean().item()
    progress_bar.set_description(f"Loss: {loss.item():.4f} Accuracy: {acc:.4f}")

    # Log metrics to wandb
    wandb.log({"Loss": loss.item(), "Accuracy": acc, "lr" : scheduler.get_lr()}, step=it)

    # Evaluation every `metrics_freq` steps
    if it % metrics_freq == 0:
        fens, moves = next(generator)
        targets = torch.tensor([hash_table_vocab_to_index[move] for move in moves], device=device)
        inputs = encode_fens(fens).to(device).view(batch_size, seq_len, input_dim)

        with torch.no_grad(), torch.amp.autocast(device):
            logits = encoding_model(inputs)
            probs = torch.nn.functional.softmax(logits, dim=-1)

        # Compute legal move probability
        board = chess.Board()
        def get_board(fen):
            board.set_board_fen(fen.split(" ")[0])
            return board
        masks = [get_legal_move_mask(get_board(fen), vocab_to_index=hash_table_vocab_to_index, dtype=logits.dtype, device=device) for fen in fens]
        masks = torch.stack(masks)
        legal_probs_sum = (probs * masks).sum(dim=-1).mean().item()

        print(f"\nLegal Probability at Iteration {it}: {legal_probs_sum}")
        torch.save(encoding_model.state_dict(), os.path.join(log_dir, f"model_{it}.pth"))
        wandb.log({"Legal probability metric": legal_probs_sum}, step=it)

# Save model
torch.save(encoding_model.state_dict(), os.path.join(log_dir, "model_last.pth"))
wandb.save(os.path.join(log_dir, "model.pth"))

print(f"Training completed. Model saved in {log_dir}")

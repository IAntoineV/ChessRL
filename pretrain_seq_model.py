


import torch
from tqdm import tqdm
import numpy as np
from models.seq_model import SeqModel, SeqModelConfig
from data.vocab import policy_index, get_hash_table_vocab_to_index
from data.parse import dir_iterator_fen_move, ParsingConfigFenMove, encode_fens
from data.fen_encoder import fen_to_tensor

device= "cuda" if torch.cuda.is_available() else "cpu"
seq_len=64
input_dim=19

config = SeqModelConfig(output_dim=len(policy_index), input_dim=input_dim, n_embd=480, seq_len=seq_len)
encoding_model = SeqModel(config).to(device)
opt = torch.optim.Adam(encoding_model.parameters(), lr=1e-3)
batch_size = 256
num_to_sample_per_game=15
num_it = 10000
dir_pgn_path = "./pgn_data"




print(f"device : {device}")
num_params_learnable = sum(p.numel() for p in encoding_model.parameters() if p.requires_grad)
num_params = sum(p.numel() for p in encoding_model.parameters())
print(f"params : {num_params} \n learnable params : {num_params_learnable}")



hash_table_vocab_to_index = get_hash_table_vocab_to_index()
parsing_config = ParsingConfigFenMove(batch_size=batch_size, num_to_sample=num_to_sample_per_game)

generator = dir_iterator_fen_move(dir_pgn_path, config=parsing_config)
progress_bar = tqdm(range(num_it))

for it in progress_bar:
    fens,moves = next(generator)
    targets = torch.tensor([hash_table_vocab_to_index[move] for move in moves]).to(device)
    inputs = encode_fens(fens).to(device).view(batch_size, seq_len, input_dim)
    logits = encoding_model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    opt.step()
    opt.zero_grad()
    acc = (torch.argmax(logits, dim=-1) == targets).float().mean()

    progress_bar.set_description(f"Loss: {loss.item()} Accuracy: {acc.item()}")




import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.cuda.amp import GradScaler, autocast
import sys
import torch.nn.functional as F
import math
import os
sys.path.append(os.getcwd())
from src.data_process.decoder_generator import generator_decoder_dir, all_moves
from torch.utils.data import DataLoader
import torch
from torch.utils.data import IterableDataset

import math
device = "cuda" if torch.cuda.is_available() else "cpu"
class PositionalEncoding(nn.Module):
    def __init__(self, nhid, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, nhid)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, nhid, 2).float() * (-math.log(10000.0) / nhid)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class ChessDecoder(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, num_heads, vocab_size, dropout=0.1):

        super().__init__()
        self.model_type = "Transformer"
        self.embdedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model) 
        self.pos_encoder = PositionalEncoding(nhid=d_model, dropout = dropout) 
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward =d_ff, dropout = dropout) #  we assume nhid = d_model = dim_feedforward
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers = num_layers) 
        self.fc_out = nn.Linear(d_model, vocab_size)  # Predict next move
        self.d_model = d_model
        self.encode_infos = nn.Linear(d_model+3, d_model)
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src, infos):
        batch_size, seq_length = src.shape
        src = self.embdedding(src) * math.sqrt(self.d_model) 
        src = self.pos_encoder(src) # addition src + position done in pos_encoder
        src = torch.cat((src, infos.unsqueeze(1).tile(1, seq_length,1)), dim=-1)
        src = self.encode_infos(src)
        src_mask = self.generate_square_subsequent_mask(batch_size).to(device) # autoregressive (decoder)
        output = self.transformer_encoder(src, mask=src_mask)
        output = self.fc_out(output)
        return output


class ChessTokenizer:

    def __init__(self, pad_token ="<PAD>", end_token="<EOS>", all_moves=all_moves):

        import copy
        self.vocab = copy.deepcopy(all_moves)
        self.vocab.extend([pad_token, end_token])
        self.pad_id = len(self.vocab) -2
        self.eos_id = len(self.vocab) -1
        self.dictionnary = { move: i for i,move in enumerate(self.vocab)}
        self.pad_token = pad_token
        self.end_token = end_token

    def tokenize(self, str_game):
        return [ self.dictionnary[el] for el in str_game] + [self.dictionnary[self.end_token]]

    def decode(self, encoded_sequences):
        
        return [[ self.vocab[index]   for index in seq ] for seq in encoded_sequences]

        


def collate_fn(batch, tokenizer=None,  pad_token="<PAD>"):
    """
    Collate function to apply padding to batches.
    
    Args:
        batch (list): List of sequences (lists of move strings).
        pad_token (str): Token used for padding.
    
    Returns:
        torch.Tensor: Padded batch of sequences.
        torch.Tensor: Lengths of original sequences before padding.
    """
    # Find the longest sequence in the batch
    max_length = max([len(el[0]) for el in batch])
    # Pad all sequences to the same length
    padded_batch = [moves[0] + [pad_token] * (max_length - len(moves[0])) for moves in batch]
    # Convert to tensor (assuming moves are tokenized into indices elsewhere)
    
    tokenized_batch = [tokenizer.tokenize(game) for game in padded_batch ]
    batch_tensor = torch.tensor(tokenized_batch, dtype=torch.long)
    # Store original lengths before padding
    lengths = torch.tensor([len(moves[0]) for moves in batch], dtype=torch.long)
    
    infos = torch.tensor([el[1] for el in batch]).long()
    return batch_tensor, lengths, infos

class ChessDataset(IterableDataset):
    def __init__(self, generator_decoder_dir, dir_path):
        self.generator_decoder_dir = generator_decoder_dir
        self.dir_path = dir_path

    def __iter__(self):
        return self.generator_decoder_dir(self.dir_path)

import torch.optim as optim
import torch.nn as nn
import chess

def compute_legal_moves_acc(inputs, outputs, tokenizer):
    b,seq_len = inputs.shape
    inputs = inputs[:, 1:]
    outputs = outputs[:, :-1]
    input_tokens = tokenizer.decode(inputs)
    output_tokens = tokenizer.decode(outputs)
    total_legal_moves = 0
    total_predicted_moves = 0
    # Compute legal move ratio
    for i in range(b):  # Iterate over batch
        board = chess.Board()  # Create a fresh chess board
        for k in range(seq_len):
            move = input_tokens[i][k]
            next_move_pred = output_tokens[i][k]
            if move == tokenizer.end_token or move == tokenizer.pad_token:
                break
            move = chess.Move.from_uci(move)  
            board.push(move)
            # if next_move_pred in list(map(lambda x : x.uci(), board.legal_moves)):
            if next_move_pred in [m.uci() for m in board.legal_moves]:
                total_legal_moves += 1
            total_predicted_moves += 1
    legal_acc = total_legal_moves / total_predicted_moves
    return legal_acc

def compute_legal_prob(inputs, outputs, tokenizer):
    """"""
    b,seq_len = inputs.shape
    outputs = outputs[:, :, :-2] #remove proba associated to eos and pad

    probs = nn.functional.softmax(outputs, dim=-1)
    total_predicted_moves = 0
    total_legal_prob = 0
    # Compute legal move ratio
    for i in range(b):  # Iterate over batch
        board = chess.Board()  # Create a fresh chess board
        for k in range(seq_len):
            move = inputs[i][k]
            move = tokenizer.decode([[move]])[0][0]
   
            if move == tokenizer.end_token or move == tokenizer.pad_token:
                break
        
            move = chess.Move.from_uci(move)  
            board.push(move)
            legal_mask = torch.zeros((vocab_size -2), device = device)
            for x in board.legal_moves:
                legal_mask[tokenizer.tokenize([x.uci()])[0]] = 1
    
            total_legal_prob += torch.sum(probs[i,k] * legal_mask, dim=-1)
            total_predicted_moves += 1
    legal_acc = total_legal_prob / total_predicted_moves
    return legal_acc


def train(model, dataloader, lr, device, num_steps=500):
    from datetime import datetime
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"{date=}")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    best_accuracy = 0.0
    best_loss = float('inf')
    total_loss_per_nums_steps = 0
    total_correct_per_num_steps = 0
    total_tokens_per_num_steps = 0

    total_correct_per_epoch = 0
    total_tokens_per_epoch = 0
    dataloader = iter(dataloader)

    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler()

    for step in range(num_steps):

        try:
            batch = next(dataloader)
        except StopIteration:
            dataloader = iter(dataloader)
            batch = next(dataloader)

        inputs, lengths, infos = batch  # Assuming inputs are tokenized indices
        inputs = inputs.to(device)
        infos = infos.to(device)

        optimizer.zero_grad()

        with autocast():

            outputs = model(inputs, infos)
            b,seq_length,vocab_size = outputs.shape 
            
            loss = criterion(outputs[:, :-1].reshape(-1, vocab_size), inputs[:, 1:].flatten())
     
        scaler.scale(loss).backward()

        # Unscale the gradients and call the optimizer step
        scaler.step(optimizer)

        # Update the scaler
        scaler.update()

        
        total_loss_per_nums_steps += loss.item()
        wandb.log({"batch_loss": loss.item()}, step=step)

        # Compute accuracy
        predictions = torch.argmax(outputs, dim=-1)  # Get predicted token indices
        correct = ((predictions[:, :-1] == inputs[:,1:])  & (predictions[:, :-1] != tokenizer.pad_id)).sum().item()  # Count correct predictions

        total_correct_per_num_steps += correct
        total_tokens_per_num_steps += inputs.numel()  # Total number of tokens
        total_correct_per_epoch += correct
        total_tokens_per_epoch +=  inputs.numel()  # Total number of tokens

        if step % 10 ==0:
            avg_loss = total_loss_per_nums_steps / 10
            accuracy = total_correct_per_num_steps / total_tokens_per_num_steps
            wandb.log({ "nums_steps_loss": avg_loss, "num_steps_accuracy": accuracy}, step=step)
            
            if step % 100==0:
                epoch_accuracy = total_correct_per_epoch/total_tokens_per_epoch
                print(f"{step=} : {epoch_accuracy=}")
                legal_acc = compute_legal_prob(inputs, outputs, tokenizer)
                wandb.log({"legal_prob": legal_acc}, step=step)
                if accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    torch.save(model.state_dict(), f"models_saves/decoder_{date}_best_acc.pth")
                    print("Best model saved!")
                # reinitialize values for epoch
                total_correct_per_epoch = 0
                total_tokens_per_epoch = 0
            # reinitialize values for the next 10 steps
            total_loss = 0
            total_correct = 0
            total_tokens = 0
            total_loss_per_nums_steps = 0
    wandb.finish()


if __name__=="__main__":
    num_layers = 6
    d_model = 960
    d_ff = 960
    num_heads = 32
    batch_size= 256
    lr = 4e-5
    num_steps = 500000
    tokenizer = ChessTokenizer()
    vocab_size = len(tokenizer.vocab)
    print(f"{vocab_size=}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessDecoder(num_layers, d_model, d_ff, num_heads, vocab_size).to(device)
    print((sum(p.numel() for p in model.parameters())))
    dir_path = os.environ.get("PGN_DIR")
    train_dataset = ChessDataset(generator_decoder_dir, dir_path = dir_path)
 
    train_dataloader =  DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, tokenizer=tokenizer), drop_last=True)

    import wandb
    wandb.init(project="chess-llm", config={
        "num_layers": num_layers,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_heads": num_heads,
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_steps": num_steps
    })   

    train(model, train_dataloader, num_steps=num_steps, lr=lr, device="cuda")
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
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src):
        src = self.embdedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src) # addition src + position done in pos_encoder
        src_mask = self.generate_square_subsequent_mask(src) # autoregressive (decoder)
        output = self.transformer_encoder(src, mask=src_mask)
        output = self.fc_out(output)
        return output


class ChessTokenizer:

    def __init__(self, pad_token ="<PAD>", end_token="<EOS>", all_moves=all_moves):


        self.vocab = all_moves
        self.vocab.extend([pad_token, end_token])
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
    max_length = max([len(moves) for moves in batch])
    # Pad all sequences to the same length
    padded_batch = [moves + [pad_token] * (max_length - len(moves)) for moves in batch]
    # Convert to tensor (assuming moves are tokenized into indices elsewhere)
    
    tokenized_batch = [tokenizer.tokenize(game) for game in padded_batch ]
    batch_tensor = torch.tensor(tokenized_batch, dtype=torch.long)
    # Store original lengths before padding
    lengths = torch.tensor([len(moves) for moves in batch], dtype=torch.long)
    
    return batch_tensor, lengths

class ChessDataset(IterableDataset):
    def __init__(self, generator_decoder_dir, dir_path):
        self.generator_decoder_dir = generator_decoder_dir
        self.dir_path = dir_path

    def __iter__(self):
        return self.generator_decoder_dir(self.dir_path)

import torch.optim as optim
import torch.nn as nn

def compute_legal_moves_acc(inputs, outputs, tokenizer):
    input_tokens = tokenizer.decode(inputs)
    output_tokens = tokenizer.decode(outputs)
    total_legal_moves = 0
    total_predicted_moves = 0
    # Compute legal move ratio
    for i in range(inputs.shape[0]):  # Iterate over batch
        board = chess.Board()  # Create a fresh chess board
        for k in range(inputs.shape[1]):
            
            next_move = output_tokens[i,k,]   
            board.push(real_move)
            if next_move
        # predicted_move = policy_index[predictions[i].item()]  # Convert prediction to UCI string
        legal_moves = [move.uci() for move in board.legal_moves]  # Get all legal moves

        if predicted_move in legal_moves:
            total_legal_moves += 1  # Count legal moves

        total_predicted_moves += 1  # Count total moves
    *
    return total_legal_moves/total_predicted_moves

def train(model, dataloader, epochs, lr, device, num_steps_per_epoch=500):
    from datetime import datetime
    date = datetime.now().strftime("%Y%m%d_%H_%_%S")
    print(f"{date=}")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    best_accuracy = 0.0
    best_loss = float('inf')
    total_loss = 0
    total_correct_per_num_steps = 0
    total_tokens_per_num_steps = 0

    total_correct_per_epoch = 0
    total_tokens_per_epoch = 0
    dataloader = iter(dataloader)
    for step in range(num_steps_per_epoch):
        try:
            batch = next(dataloader)
        except:
            batch = next(iter(dataloader)) #new_iteration

        inputs, lengths = batch  # Assuming inputs are tokenized indices
        inputs = inputs.to(device)
        
        optimizer.zero_grad()
                    
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), inputs.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        wandb.log({"batch_loss": loss.item()})

        # Compute accuracy
        predictions = torch.argmax(outputs, dim=-1)  # Get predicted token indices
        correct = (predictions == inputs).sum().item()  # Count correct predictions
        total_correct_per_num_steps += correct
        total_tokens_per_num_steps += inputs.numel()  # Total number of tokens

        if step % 10 ==0:
            avg_loss = total_loss / num_steps_per_epoch
            accuracy = total_correct_per_num_steps / total_tokens_per_num_steps
            wandb.log({"step": step+1, "epoch_loss": avg_loss, "epoch_accuracy": accuracy})
            
            if step % 1000:
                epoch_accuracy = total_correct_per_epoch/total_tokens_per_epoch
                print(f"{step=} : {epoch_accuracy=}")

                


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
    wandb.finish()


if __name__=="__main__":
    num_layers = 6
    d_model = 640
    d_ff = 640
    num_heads = 32
 
    batch_size=16
    lr = 1e-3
    epochs = 10
    tokenizer = ChessTokenizer()
    vocab_size = len(tokenizer.vocab)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessDecoder(num_layers, d_model, d_ff, num_heads, vocab_size).to(device)

    dir_path = "/raid/home/detectionfeuxdeforet/caillaud_gab/llm_project/ChessRL/data/"
    train_dataset = ChessDataset(generator_decoder_dir, dir_path = dir_path)

 
    train_dataloader =  DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, tokenizer=tokenizer))

    import wandb
    wandb.init(project="chess-llm", config={
        "num_layers": num_layers,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_heads": num_heads,
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": epochs
    })   

    train(model, train_dataloader, epochs=epochs, lr=lr, device="cuda")
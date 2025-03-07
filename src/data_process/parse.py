import os
from symtable import Function

import chess
import chess.pgn
import numpy as np
import torch
from attr import dataclass

from data_process.fen_encoder import fen_to_tensor
from data_process.vocab import policy_index

from game_sampler import linear_augmentation_sampler, order_and_compute_deltas
@dataclass
class ParsingConfig:
    batch_size = 32
    min_length = 20
    num_moves = 10
    block_size = 256
    padding_idx = len(policy_index)
    @property
    def clip_length(self):
        return self.block_size-64

class ParsingConfigFenMove:
    def __init__(self, sampler=None, batch_size = 32, min_length = 20, num_to_sample=2):
        if sampler is None:
            sampler = linear_augmentation_sampler
        self.sampler = sampler
        self.batch_size = batch_size
        self.min_length = min_length
        self.num_to_sample = num_to_sample

    def sample(self, length_game):
        return order_and_compute_deltas( self.sampler(length_game, self.num_to_sample))






def get_batch(pgn_path, config:ParsingConfig, return_fen = False):
    with open(pgn_path, "r") as f:
        fen_array = []
        moves_array = []
        while True:
            pgn = chess.pgn.read_game(f)
            if pgn.next()==None:
                continue
            elo = min(int(pgn.headers["WhiteElo"]),int(pgn.headers["BlackElo"]))
            if elo <= 2200 or 'FEN' in pgn.headers.keys() or '960' in pgn.headers['Event'] or 'Odds' in pgn.headers['Event'] or 'house' in pgn.headers['Event']:
                continue
            moves = [move for move in pgn.mainline_moves()]
            if len(moves) < config.min_length:
                continue
            #start index is a random int in 0, len(moves)- num_moves
            start_index = np.random.randint(0,len(moves)-config.num_moves)
            board = chess.Board()
            for move in moves[:start_index]:
                board.push(move)
            fen = board.fen()
            moves = moves[start_index:]
            fen_array.append(fen)
            moves_array.append(encode_moves(moves))
            if len(fen_array) == config.batch_size:
                if return_fen:
                    yield (fen_array)
                else:
                    yield (encode_fens(fen_array).to("cuda"),clip_and_batch(moves_array,batch_size=config.batch_size, clip=config.clip_length, padding_idx=config.padding_idx).to("cuda"))
                fen_array = []
                moves_array = []

def dir_iterator(dir_path,config:ParsingConfig=None, return_fen = True):
    if config is None:
        import warnings
        warnings.warn("PGN processing without config given, basic one used...")
        config = ParsingConfig()
    for pgn in os.listdir(dir_path):
        print(pgn)
        pgn_path = os.path.join(dir_path,pgn)
        gen = get_batch(pgn_path,config, return_fen = return_fen)
        while True:
            try:
                yield next(gen)
            except:
                break
            
def encode_fens(fen_array):
    #encode in pytorch tensor
    #print(fen_array[0])
    fens = torch.from_numpy(np.array([fen_to_tensor(fen) for fen in fen_array]))
    return fens

def encode_moves(moves_array):
    moves = []
    #print(moves_array)
    for move in moves_array:
        if move.uci()[-1]!='n':
            move_id = policy_index.index(move.uci())
        else:
            move_id = policy_index.index(move.uci()[:-1])
        moves.append(move_id)
    return torch.from_numpy(np.array(moves))

def clip_and_batch(moves_array,batch_size,clip, padding_idx):
    #clip and batch moves
    moves = torch.full((batch_size,clip),padding_idx,dtype = torch.int64)
    for i in range(batch_size):
        if moves_array[i].shape[0] > clip:
            moves[i] = moves_array[i][:clip]
        else:
            moves[i,:moves_array[i].shape[0]] = moves_array[i]
    return moves




def fen_move_tuple_generator(pgn_path, config:ParsingConfigFenMove):
    with open(pgn_path, "r") as f:
        fen_array = []
        moves_array = []
        while True:
            pgn = chess.pgn.read_game(f)
            if pgn.next()==None:
                continue
            elo = min(int(pgn.headers["WhiteElo"]),int(pgn.headers["BlackElo"]))
            if elo <= 2200 or 'FEN' in pgn.headers.keys() or '960' in pgn.headers['Event'] or 'Odds' in pgn.headers['Event'] or 'house' in pgn.headers['Event']:
                continue
            moves = [move for move in pgn.mainline_moves()]
            if len(moves) < config.min_length:
                continue
            deltas = config.sample(len(moves)-1)
            board = chess.Board()
            it=0
            for delta in deltas:
                for _ in range(delta):
                    board.push(moves[it])
                    it+=1
                fen = board.fen()
                fen_array.append(fen)
                moves_array.append(str(moves[it]))
                if len(fen_array) == config.batch_size:
                    yield fen_array,moves_array
                    fen_array = []
                    moves_array = []



def dir_iterator_fen_move(dir_path,config:ParsingConfigFenMove=None):
    if config is None:
        import warnings
        warnings.warn("PGN processing without config given, basic one used...")
        config = ParsingConfigFenMove()
    all_files = os.listdir(dir_path)
    print("pgns : ", all_files)
    for pgn in all_files:
        print(pgn)
        pgn_path = os.path.join(dir_path,pgn)
        gen = fen_move_tuple_generator(pgn_path,config)
        while True:
            try:
                yield next(gen)
            except:
                break
# if __name__ == "__main__":
#     dir_path = "../pgn_data_example"
#     config = ParsingConfig()
#     config.batch_size=2

#     gen = dir_iterator(dir_path, config, return_fen = False)
#     for _ in range(1):
#         fens,moves = next(gen)
#         print(fens.shape,moves.dtype)

#     config = ParsingConfigFenMove()
#     config.batch_size = 2

#     gen = dir_iterator_fen_move(dir_path, config)
#     for _ in range(1):
#         fens, moves = next(gen)
#         print(fens.shape, moves)

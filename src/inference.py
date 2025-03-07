from data_process.data_gen import swap_side, get_x_from_board
import json
import chess
import torch
from models.bt4 import BT4
from vocab import PolicyIndex

if __name__ == '__main__':
    dir = "../models_saves/model_1/"
    weights_path = dir + "model.pth"
    config_path = dir + "config.json"

    config = json.load(open(config_path, "r"))
    model =  BT4(**config)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    fen = "r1bq1rk1/2pp1ppp/p1n2n2/2b1p3/1pP1P3/1B1P1N2/PP3PPP/RNBQR1K1 b - c3 0 9"
    board = chess.Board()
    board.set_board_fen(fen)
    policy_manager = PolicyIndex()






import numpy as np
import chess.pgn
import chess
import json
import torch
from src.models.bt4 import BT4
from src.data_process.data_gen import get_x_from_board, swap_side, update_repr
from src.data_process.vocab import PolicyIndex
class AIHandler:

    @staticmethod
    def get_init_repr(L_moves_from_start, elo, TC, history=7):
        """

        :param L_moves_from_start:
        :param elo:
        :param TC:
        :param history:
        :return:
        """
        board = chess.Board()
        x_start = []
        boad_turn = len(L_moves_from_start) # odd == black turn at the end, even == white turn at the end
        for i, move in enumerate(L_moves_from_start):
            xs = get_x_from_board(elo, board, TC, with_lm_mask=False)  # get white repr of board
            board.push(move) # play the move
            if (boad_turn - i) % 2 == 0:
                x_start.append(xs)
            else:
                x_start.append(swap_side(xs))  # We flip side to retransform black from white repr to black repr



        x_start = x_start[-(history):]  # Keep only the last "history" board representation.
        x_start = np.concatenate([x[:, :, :12] for x in x_start], axis=-1)  # We only keep board representation and
        # take away (castling_rights, en_passant_right, color, TC, elo) from each history board

        return x_start

if __name__ == '__main__':
    device= "cuda" if torch.cuda.is_available() else "cpu"
    print("device : ", device)
    elo=2700
    TC="600"
    history = 7
    pgn_path = "../../pgn_data_example/pgn_example.pgn"

    dir = "../../models_saves/model_1/"
    weights_path = dir + "model.pth"
    config_path = dir + "config.json"

    config = json.load(open(config_path, "r"))
    model = BT4(**config).to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    policy_manager = PolicyIndex()

    with open(pgn_path, encoding="utf8") as f:
        # load game
        pgn = chess.pgn.read_game(f)
        pgn.next()
        moves = [move for move in pgn.mainline_moves()]
        start_index = 29
        moves = moves[:start_index]
        x_start = AIHandler.get_init_repr(moves, elo, TC, history=history)
        board = chess.Board()
        for move in moves:
            board.push(move)
        i=0
        while True:
            print(board.unicode())
            print("Model to play :")
            xs, lm_mask = get_x_from_board(elo, board, TC)

            if i==0:
                x_repr =np.concatenate((x_start, xs), axis=-1)[np.newaxis]
            else:
                x_repr = update_repr(x_repr[0], xs,history=history)[np.newaxis]
            Xs = torch.from_numpy(x_repr[:, :, :, :-2]).to(device)
            Es = torch.from_numpy(x_repr[:, :, :, -2:]).to(device)
            y_pred,_= model(Xs,Es)
            move_index = y_pred.argmax(dim=-1).item()
            token = policy_manager.policy_index[move_index]
            print(token)
            uci_ai = policy_manager.token_to_uci(board,token)
            print("token :", token, " move : ", uci_ai)
            uci_ai = chess.Move.from_uci(uci_ai)
            board.push(uci_ai)
            xs, _ = get_x_from_board(elo, board, TC)
            x_repr = update_repr(x_repr[0], xs, history=history)[np.newaxis]
            print(board.unicode())
            uci_player = input("Play a move :")
            uci_player = chess.Move.from_uci(uci_player)
            board.push(uci_player)
            xs, _ = get_x_from_board(elo, board, TC)
            x_repr = update_repr(x_repr[0], xs, history=history)[np.newaxis]

            i+=1





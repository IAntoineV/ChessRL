import numpy as np
import chess
import torch
import json
from src.models.bt4 import BT4
from src.data_process.data_gen import get_x_from_board, update_repr
from src.data_process.vocab import PolicyIndex
from src.model_evaluation.ChessBot import ChessBot
from src.data_process.play_against_AI import AIHandler


class BT4Bot(ChessBot):
    def __init__(self, model_path: str, config_path: str, elo: int = 2700, time_control: str = "600", history: int = 7):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.elo = elo
        self.time_control = time_control
        self.history = history
        self.board = chess.Board()

        config = json.load(open(config_path, "r"))
        self.model = BT4(**config).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.policy_manager = PolicyIndex()
        self.x_repr = None
        self.i=0
        self.current_log = []
        self.log_history = []
        self.num_moves =0
    def initialize_pos(self, list_moves: list):
        self.log_history.append(self.current_log)
        self.current_log = []
        self.board = chess.Board()
        for move in list_moves:
            self.board.push_uci(move)
        self.x_repr = AIHandler.get_init_repr(list_moves, self.elo, self.time_control, history=self.history)
        self.i=0
        self.num_moves= len(list_moves)

    def get_bot_move(self, opponent_move: str) -> str:
        self.board.push_uci(opponent_move)
        self.num_moves+=1
        xs, lm_mask = get_x_from_board(self.elo, self.board, self.time_control)
        if self.i == 0:
            self.x_repr = np.concatenate((self.x_repr, xs), axis=-1)[np.newaxis]
        else:
            self.x_repr = update_repr(self.x_repr[0], xs, history=self.history)[np.newaxis]
        Xs = torch.from_numpy(self.x_repr[:, :, :, :-2]).to(self.device)
        Es = torch.from_numpy(self.x_repr[:, :, :, -2:]).to(self.device)
        y_pred, _ = self.model(Xs, Es)
        proba = torch.nn.functional.softmax(y_pred, dim=-1).detach().cpu().numpy()
        entropy = -np.sum(proba * np.log(proba + 1e-9))
        self.current_log.append((self.num_moves, entropy))
        move_index = np.argmax(lm_mask*proba)
        #masked_proba = (lm_mask * proba)[0]  # Apply the mask
        #masked_proba /= masked_proba.sum()  # Normalize to make it a valid probability distribution
        #move_index = np.random.choice(len(proba[0]), p=masked_proba)  # Sample based on probability
        legal_prob = (lm_mask * proba).sum()
        if  legal_prob < 1e-1:
            print("model playing many illegal moves :", legal_prob, "board : ", not self.board.turn)
        token = self.policy_manager.policy_index[move_index]
        uci_ai = self.policy_manager.token_to_uci(self.board, token)
        self.board.push_uci(uci_ai)
        self.num_moves+=1
        xs, _ = get_x_from_board(self.elo, self.board, self.time_control)
        self.x_repr = update_repr(self.x_repr[0], xs, history=self.history)[np.newaxis]
        self.i+=1
        return uci_ai



if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    pgn_dir = os.environ.get("PGN_DIR")
    pgn_path = os.path.join(pgn_dir, os.listdir(pgn_dir)[0])
    print(f"pgn file : {pgn_path}")
    model_dir = os.environ.get("MODEL_DIR")
    print(f"model dir : {model_dir}")

    import chess.pgn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device : ", device)
    elo = 2700
    TC = "600"
    history = 7

    weights_path = os.path.join(model_dir, "model.pth")
    config_path = os.path.join(model_dir,"config.json")
    bot = BT4Bot(weights_path,config_path, elo=elo, time_control=TC, history=history )
    with open(pgn_path, encoding="utf8") as f:
        # load game
        pgn = chess.pgn.read_game(f)
        pgn.next()
        moves = [move for move in pgn.mainline_moves()]
        start_index = 29
        moves = list(map(lambda x : x.uci(), moves[:start_index]))
        start_moves = moves[:-1]
        bot.initialize_pos(start_moves)
        opponent_move = moves[-1]
        print("First Move", opponent_move)
        while True:
            print("Model to play :")
            ai_move = bot.get_bot_move(opponent_move)
            print("AI Move :", ai_move)
            print(bot.board.unicode())
            opponent_move = input("Play a move :")

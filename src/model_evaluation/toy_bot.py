from src.model_evaluation.ChessBot import ChessBot
import chess
import random


class RandomBot(ChessBot):
    def __init__(self):
        self.board=  chess.Board()


    def initialize_pos(self, list_moves):
        self.board=chess.Board()
        for move in list_moves:
            self.board.push_uci(move)

    def get_bot_move(self, move):
        self.board.push_uci(move)
        all_legal_moves = list(self.board.legal_moves)
        move = random.choice(all_legal_moves).uci()
        self.board.push_uci(move)
        return move






import chess
from typing import List
import abc
import sys
import os
sys.path.append(os.getcwd())
from src.models.mcts import MCTS
from src.models.alphabeta_agent import alpha_beta
import random



class ChessBot(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_bot_move(self, move: str) -> str:
        """
        Update board with the opponent move (if any), compute your best move,
        update the board, and return your move in UCI string format.
        """
        pass

    @abc.abstractmethod
    def initialize_pos(self, list_moves: List[str]) -> None:
        """
        Initialize the internal board by playing a list of moves in UCI format.
        """
        pass


class MCTSChessBot(ChessBot):
    def __init__(self, critic=None, exploration_constant: float = 1.414, iterations: int = 1000):
        self.board = chess.Board()
        self.mcts = MCTS(
            critic=critic, exploration_constant=exploration_constant)
        self.iterations = iterations

    def initialize_pos(self, list_moves: list):
        for move in list_moves:
            self.board.push_uci(move)

    def get_bot_move(self, opponent_move: str) -> str:
        if opponent_move:
            self.board.push_uci(opponent_move)
        best_move, metrics = self.mcts.search(
            self.board, iterations=self.iterations)
        self.board.push(best_move)
        print("MCTS metrics:", metrics)
        print("best move:", best_move)
        return best_move.uci()


class AlphaBetaChessBot(ChessBot):
    def __init__(self, search_depth: int = 3):
        self.board = chess.Board()
        self.search_depth = search_depth

    def initialize_pos(self, list_moves: list):
        for move in list_moves:
            self.board.push_uci(move)

    def get_bot_move(self, opponent_move: str) -> str:
        if opponent_move:
            self.board.push_uci(opponent_move)
        _, best_move = alpha_beta(
            self.board, self.search_depth, -float('inf'), float('inf'), True)
        if best_move is None:
            # In rare cases when no best move is found, choose randomly.
            best_move = random.choice(list(self.board.legal_moves))
        self.board.push(best_move)
        return best_move.uci()


if __name__ == "__main__":
    # Initialize bots with desired parameters.
    mcts_bot = MCTSChessBot(iterations=1000)
    alphabeta_bot = AlphaBetaChessBot(search_depth=3)

    # Initialize board with an opening sequence, e.g.,
    moves = ["e2e4", "e7e5", "g1f3"]
    mcts_bot.initialize_pos(moves)
    alphabeta_bot.initialize_pos(moves)

    # For demonstration, let the two bots play against each other:
    board = chess.Board()
    for move in moves:
        board.push_uci(move)

    print("Starting game between MCTS (White) and AlphaBeta (Black)")
    move = ''
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = mcts_bot.get_bot_move(move)
            print("MCTS (White) chooses:", move)
            print(mcts_bot.board, '\n')
        else:
            move = alphabeta_bot.get_bot_move(move)
            print("AlphaBeta (Black) chooses:", move)
            print(alphabeta_bot.board, '\n')
        board.push_uci(move)
        

    print("Game over. Result:", board.result())

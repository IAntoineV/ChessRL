


from src.model_evaluation.ChessBot import ChessBot, MCTSChessBot, AlphaBetaChessBot
from src.model_evaluation.StockfishBot import StockfishBot
import chess
import chess.pgn

import chess
import chess.pgn
import random
import numpy as np
from collections import defaultdict
from typing import List, Type


class ChessTournament:
    def __init__(self, names, bots: List[ChessBot], starting_position_generator, games_per_match=10):
        self.bots = bots
        self.names = names
        self.starting_position_generator = starting_position_generator
        self.games_per_match = games_per_match
        self.scores = { name : 0 for name in self.names }
        self.win_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})
        self.results_matrix = -np.ones((len(bots), len(bots), 2))  # Store results for both white and black

    def play_game(self, bot1: ChessBot, bot2: ChessBot, name1: str, name2: str):
        board = chess.Board()
        list_moves = next(self.starting_position_generator)
        num_moves = len(list_moves)
        for move in list_moves[:-1]:
            board.push_uci(move)
        turn = {True: bot1, False: bot2}
        first_to_play = not board.turn
        turn[first_to_play].initialize_pos(list_moves[:-1])
        turn[1- first_to_play].initialize_pos(list_moves)
        last_move = list_moves[-1]
        board.push_uci(last_move)
        while not board.is_game_over():
            current_bot = turn[board.turn]

            move = current_bot.get_bot_move(last_move)
            if move not in [elt.uci() for elt in board.legal_moves]:
                print(f"Illegal move by {name1 if current_bot == bot1 else name2}")
                return bot2 if board.turn else bot1  # The opponent wins
            last_move = move
            board.push_uci(last_move)

        result = board.result()
        if result == "1-0":
            return bot1
        elif result == "0-1":
            return bot2
        else:
            return None  # Draw

    def run_1v1_matches(self):
        num_bots = len(self.bots)
        num_games=0
        total_games = self.games_per_match*num_bots*(num_bots-1)
        for i, (bot1, name1) in enumerate(zip(self.bots, self.names)):
            for j, (bot2, name2) in enumerate(zip(self.bots, self.names)):

                if i == j:
                    continue
                for k in range(self.games_per_match):
                    print(f"Num match to play {num_games}/{total_games} ")
                    num_games += 1
                    white_bot, black_bot = (bot1, bot2) if k % 2 == 0 else (bot2, bot1)
                    white_name, black_name = (name1, name2) if k % 2 == 0 else (name2, name1)
                    winner = self.play_game(white_bot, black_bot, white_name, black_name)

                    if winner is not None:
                        winner_name = white_name if winner == white_bot else black_name
                        self.scores[winner_name] += 1
                        self.win_stats[winner_name]['wins'] += 1
                        self.results_matrix[i][j][k % 2] = 1 if winner == white_bot else 0
                    else:
                        self.win_stats[name1]['draws'] += 1
                        self.win_stats[name2]['draws'] += 1
                        self.results_matrix[i][j][k % 2] = 0.5  # Draw


    def display_results(self):
        print("Final Rankings:")
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (bot, score) in enumerate(sorted_scores, start=1):
            print(f"{rank}. {bot}: {score} points")
        print("Win Statistics:")
        for bot, stats in self.win_stats.items():
            print(f"{bot} - Wins: {stats['wins']}, Draws: {stats['draws']}")

        print("\nResults Matrix:")
        for i, bot1 in enumerate(self.names):
            for j, bot2 in enumerate(self.names):
                if i != j:
                    print(
                        f"{bot1} vs {bot2} - White Win Rate: {self.results_matrix[i][j][0]}, Black Win Rate: {self.results_matrix[i][j][1]}")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    pgn_dir = os.environ.get("PGN_DIR")
    from src.model_evaluation.toy_bot import  RandomBot
    from src.data_process.parse import dir_decorator, list_move_generator
    generator = dir_decorator(list_move_generator, pgn_dir)
    def gen():
        for list_move in generator:
            yield [move.uci() for move in list_move]


    generator_listmove = gen()
    # bot_names = ["RandomBot1", "RandomBot2"]
    # bots = [ RandomBot(), RandomBot()]
    import os
    stockfish_path = os.environ.get("STOCKFISH_PATH")

    bot_names = ["RandomBot", "Stockfish"]
    bots = [ RandomBot(), StockfishBot(stockfish_path)]

    # bot_names = ["RandomBot", "AlphaBetaBot"]
    # bots = [ RandomBot(), AlphaBetaChessBot(search_depth=4)]

    tournament = ChessTournament(bot_names, bots, generator_listmove, games_per_match=2)
    tournament.run_1v1_matches()
    tournament.display_results()

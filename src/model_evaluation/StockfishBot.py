from src.model_evaluation.ChessBot import ChessBot
import chess
import chess.engine


class StockfishBot(ChessBot):
    def __init__(self, engine_path: str, elo=1500, time=1.0):
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.engine.configure(
            {"UCI_LimitStrength": True, "UCI_Elo": elo}
        )
        self.time=time

    def initialize_pos(self, list_moves: list):
        self.board = chess.Board()
        for move in list_moves:
            self.board.push_uci(move)

    def get_bot_move(self, opponent_move: str) -> str:
        if opponent_move:
            self.board.push_uci(opponent_move)

        result = self.engine.play(self.board, chess.engine.Limit(time=self.time))
        best_move = result.move

        self.board.push(best_move)
        return best_move.uci()

    def close_engine(self):
        self.engine.quit()

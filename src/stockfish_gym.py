import gymnasium as gym
import numpy as np
import torch
import chess
import chess.engine
import random as rd
import os
from gymnasium import spaces
from data_process.vocab import policy_index, get_hash_table_vocab_to_index
from data_process.fen_encoder import fen_to_tensor


class ChessStockfishEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, stockfish_path, stockfish_elo=1350, reward_eval_elo=1500):
        super(ChessStockfishEnv, self).__init__()

        # Initialize chess board
        self.board = chess.Board()

        # Setup Stockfish engines
        self.stockfish_path = stockfish_path
        self.stockfish_elo = stockfish_elo
        self.reward_eval_elo = reward_eval_elo

        # Create opponent engine
        self.opponent_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.opponent_engine.configure(
            {"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo}
        )

        # Create evaluation engine for rewards
        self.reward_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.reward_engine.configure(
            {"UCI_LimitStrength": True, "UCI_Elo": reward_eval_elo}
        )

        # Define action and observation spaces
        self.action_space = spaces.Discrete(
            len(policy_index)
        )  # Use move vocabulary size
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=fen_to_tensor(self.board.fen()).shape,  # Use FEN encoder output shape
            dtype=np.float32,
        )

        # Store move vocabulary
        self.move_vocab = policy_index
        self.vocab_to_index = get_hash_table_vocab_to_index(vocab_list=self.move_vocab)
        self.current_legal_moves = []

    def get_legal_move_mask(self, dtype=torch.float32, device="cpu"):
        mask = torch.zeros(len(policy_index), dtype=dtype, device=device)
        indices = torch.tensor(
            [self.vocab_to_index[move] for move in self.current_legal_moves],
            device=device,
        )
        mask.scatter_(0, indices, 1)
        return mask

    def _action_to_move(self, action):
        """Convert action index to a chess move using the move vocabulary."""
        move_token = self.move_vocab[action]
        try:
            move = chess.Move.from_uci(move_token)
            return move
        except ValueError:
            return None  # Invalid move

    def reset(self, fen=None, seed=None, options=None):
        """Reset the environment to the initial state."""
        if fen is None:
            self.board.reset()
        else:
            self.board.set_board_fen(fen.split(" ")[0])
        self.current_legal_moves = list(self.board.legal_moves)
        return fen_to_tensor(self.board.fen()), {}  # Use FEN encoder for observation

    def step(self, action):
        """Execute one step in the environment."""
        infos = {}
        truncated = False

        # Convert action to chess move
        move = self._action_to_move(action)
        if move is None or move not in self.current_legal_moves:
            infos["error"] = "Invalid move"
            return fen_to_tensor(self.board.fen()), -10, True, True, infos

        # Apply player move
        self.board.push(move)

        # Check if game ended after player move
        if self.board.is_game_over():
            result = self._game_result_reward(
                player_caused=True
            )  # Player caused the game to end
            infos["result"] = result[1]["result"]
            return fen_to_tensor(self.board.fen()), result[0], True, False, infos

        # Get Stockfish move
        try:
            result = self.opponent_engine.play(self.board, chess.engine.Limit(time=0.1))
            self.board.push(result.move)
        except chess.engine.EngineTerminatedError:
            infos["error"] = "Engine crashed"
            return fen_to_tensor(self.board.fen()), 0, True, True, infos

        # Check if game ended after Stockfish move
        if self.board.is_game_over():
            result = self._game_result_reward(
                player_caused=False
            )  # Stockfish caused the game to end
            infos["result"] = result[1]["result"]
            return fen_to_tensor(self.board.fen()), result[0], True, False, infos

        # Calculate reward using evaluation engine
        try:
            analysis = self.reward_engine.analyse(
                self.board, chess.engine.Limit(depth=10)
            )
            score = analysis["score"].white().score(mate_score=10000)
            reward = np.tanh(score / 1000)  # Normalize to [-1, 1]
        except chess.engine.EngineTerminatedError:
            infos["error"] = "Engine reward crashed"
            reward = 0

        # Update legal moves
        self.current_legal_moves = list(self.board.legal_moves)

        # Check if the game is over (redundant, but ensures correctness)
        done = self.board.is_game_over()
        if done:
            result = self._game_result_reward(
                player_caused=False
            )  # Stockfish caused the game to end
            infos["result"] = result[1]["result"]
            reward = result[0]

        return fen_to_tensor(self.board.fen()), reward, done, False, infos

    def _game_result_reward(self, player_caused):
        """Calculate terminal rewards based on game result and who caused it."""
        if self.board.is_checkmate():
            if player_caused:
                # Player delivered checkmate (player wins)
                return (1.0, {"result": "win"})
            else:
                # Stockfish delivered checkmate (player loses)
                return (-1.0, {"result": "loss"})
        elif (
            self.board.is_stalemate()
            or self.board.is_insufficient_material()
            or self.board.is_seventyfive_moves()
            or self.board.is_fivefold_repetition()
        ):
            # Draw
            return (0.0, {"result": "draw"})
        else:
            # Other terminal states (e.g. timeout)
            return (0.0, {"result": "unknown"})

    def render(self, mode="human"):
        """Render the current board state."""
        print(self.board.unicode())
        return

    def close(self):
        """Clean up resources."""
        print("Quit engines")
        self.opponent_engine.quit()
        self.reward_engine.quit()

    def __del__(self):
        self.close()

    def samples_legal_action(self):

        return rd.sample([elt.__str__() for elt in self.board.legal_moves], 1)[0]


if __name__ == "__main__":
    load_dotenv()
    env = ChessStockfishEnv(
        stockfish_path=os.environ.get("STOCKFISH_PATH"),
        stockfish_elo=1320,
        reward_eval_elo=1500,
    )

    obs = env.reset(
        fen="r1bq1rk1/2pp1ppp/p1n2n2/2b1p3/1pP1P3/1B1P1N2/PP3PPP/RNBQR1K1 w - c3 0 9"
    )
    done = False
    total_reward = 0
    while not done:
        action = env.samples_legal_action()

        obs, reward, terminated, truncated, info = env.step(env.vocab_to_index[action])
        total_reward += reward
        print("=" * 20)
        env.render()

        if terminated:
            print(f"Game ended. Total reward: {total_reward}")
            print("Final result:", info)
            break

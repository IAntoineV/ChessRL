import chess
import chess.engine
import chess.pgn
import torch
import math
import random
import time
import sys
import os

sys.path.append(os.getcwd())
from src.data_process.fen_encoder import fen_to_tensor
from src.reward_train.custom_board_reward import evaluate_board



def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a chess.Board into a torch tensor with shape (19, 8, 8)
    """
    return torch.tensor(fen_to_tensor(board.fen()), dtype=torch.float32).permute(2, 0, 1)


class Node:
    def __init__(self, board: chess.Board, parent: "Node" = None, move: chess.Move = None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: list[Node] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.untried_moves = list(board.legal_moves)

    def is_fully_expanded(self) -> bool:
        """Return True if there are no more moves to try."""
        return len(self.untried_moves) == 0

    def best_child(self, exploration_constant: float) -> "Node":
        """
        Select the child with the highest UCT (Upper Confidence Bound for Trees) score.
        """
        best_score = -float("inf")
        best_child = None
        for child in self.children:
            # If the child hasn't been visited, set score to infinity.
            score = float("inf") if child.visits == 0 else child.value + \
                exploration_constant * \
                math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self) -> "Node":
        """
        Expand the node by selecting one untried move at random and adding the resulting child.
        """
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        new_board = self.board.copy()
        new_board.push(move)
        child = Node(new_board, parent=self, move=move)
        self.children.append(child)
        return child

    def update(self, value: float) -> None:
        """Update the node with the simulation result."""
        self.visits += 1
        self.value += value

    def __str__(self) -> str:
        return f"Node(move={self.move}, visits={self.visits}, value={self.value:.2f})"


class MCTS:
    def __init__(self, critic=None, exploration_constant: float = 1.414, max_depth: int = 10, epsilon: float = 0.2):
        # Neural network for board evaluation (if available).
        self.critic = critic
        self.exploration_constant = exploration_constant
        self.nodes_expanded = 0
        self.epsilon = epsilon
        self.max_depth = max_depth

    def search(self, board: chess.Board, iterations: int = 1000):
        """
        Run MCTS for a given number of iterations and return the selected move
        """
        root = Node(board.copy())
        start_time = time.time()

        for _ in range(iterations):
            node = root
            # Selection: Traverse tree until reaching a node that is not fully expanded.
            while node.is_fully_expanded() and len(node.children)>0:
                node = node.best_child(self.exploration_constant)


            # Expansion: Expand node if it's not terminal.
            if not node.is_fully_expanded():
                node = node.expand()
                self.nodes_expanded += 1

            # Simulation: Evaluate the board state.
            value = self.simulate(node.board)

            # Backpropagation: Update nodes value and number of visits from the expanded node up to the root.
            while node is not None:
                node.update(value)
                node = node.parent

        total_time = time.time() - start_time
        metrics = {
            "iterations": iterations,
            "time_seconds": total_time,
            "nodes_expanded": self.nodes_expanded,
        }
        # Select move from the child of the root with the highest visit count. => we use the number of visit as confidence in the move value (maybe try other selection)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move, metrics

    def simulate(self, board: chess.Board) -> float:
        """
        Simulate a game from the current board state using the critic network for evaluation if available or by performing random rollouts (until max depth or game over).
        """
        if self.critic is not None:
            x = board_to_tensor(board)
            with torch.no_grad():
                return self.critic(x.unsqueeze(0)).item()
        else:
            rollout_board = board.copy()
            depth = 0
            while not rollout_board.is_game_over() and depth < self.max_depth:
                if rollout_board.is_checkmate():
                    return -10000.0 if rollout_board.turn == chess.WHITE else 10000.0  # Loss for White, Win for Black

                moves = list(rollout_board.legal_moves)
                if not moves:
                    break
                if random.random() < self.epsilon:
                    move = random.choice(moves)
                else:
                    best_reward = -float("inf")
                    best_moves = []
                    for move in moves:
                        test_board = rollout_board.copy()
                        test_board.push(move)
                        reward = evaluate_board(test_board)
                        if reward > best_reward:
                            best_reward = reward
                            best_moves = [move]
                        elif reward == best_reward:
                            best_moves.append(move)
                    move = random.choice(best_moves)
                
                rollout_board.push(move)
                depth += 1
            return evaluate_board(rollout_board)

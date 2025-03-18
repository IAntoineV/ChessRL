import os
import sys
sys.path.append(os.getcwd())


from src.models.mcts_critic_model import CriticNet
from src.models.mcts import Node, MCTS
from dotenv import load_dotenv
from src.data_process.fen_encoder import fen_to_tensor
import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random



def board_to_tensor(board: chess.Board):
    # shape: (19,8,8)
    return torch.tensor(fen_to_tensor(board.fen()), dtype=torch.float32).permute(2, 0, 1)

# Supervised Training for the Critic Network


def train_critic(critic, dataset, epochs=10, batch_size=32, lr=1e-3):
    optimizer = optim.Adam(critic.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    critic.train()
    for epoch in range(epochs):
        random.shuffle(dataset)
        losses = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            boards = [board_to_tensor(item[0]) for item in batch]
            evaluations = [item[1] for item in batch]
            # create a tensor of shape (batchsize, 19, 8, 8)
            boards_tensor = torch.stack(boards)
            eval_tensor = torch.tensor(evaluations, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = critic(boards_tensor)
            loss = loss_fn(outputs, eval_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}: Loss = {np.mean(losses):.4f}")

# Generate Training Data using Stockfish Evaluations => TO IMPROVE


def generate_dataset(engine, num_positions=100):
    dataset = []
    board = chess.Board()
    for _ in range(num_positions):
        if _ % 10 == 0:
            print(f"Generating position {_}")
        moves_to_play = random.randint(5, 15)
        for _ in range(moves_to_play):
            if board.is_game_over():
                break
            move = random.choice(list(board.legal_moves))
            board.push(move)
        # Use Stockfish for evaluation.
        info = engine.analyse(board, chess.engine.Limit(time=0.1))
        score = info["score"].white().score(mate_score=10000)
        if score is None:
            score = 0
        eval_normalized = np.tanh(score / 400.0)
        dataset.append((board.copy(), eval_normalized))
        board.reset()  # Restart for next sample.
    return dataset

# Display board and some metrics for debugging


def debug_display(board, metrics=None):
    print(board)
    if metrics:
        print("Metrics:", metrics)
    print("\n" + "="*50 + "\n")

# Testing on Lichess Puzzles


def test_on_lichess_puzzles(mcts, puzzles):
    results = []
    for fen in puzzles:
        board = chess.Board(fen)
        move, metrics = mcts.search(board, iterations=1000)
        results.append({'fen': fen, 'move': move, 'metrics': metrics})
        print("Puzzle FEN:", fen)
        print(board)
        print("MCTS selected move:", move)
        print("Search Metrics:", metrics)
        print("\n" + "-"*40 + "\n")
    return results


if __name__ == "__main__":

    # Load Stockfish Engine
    load_dotenv()
    engine_path = os.getenv("STOCKFISH_PATH")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path,)

    # # Generate dataset using Stockfish evaluations.
    # print("Generating dataset from Stockfish evaluations...")
    # dataset = generate_dataset(engine, num_positions=50)

    # critic = CriticNet()
    # print("Training critic network on generated dataset...")
    # train_critic(critic, dataset, epochs=50)

    # mcts = MCTS(critic=critic, exploration_constant=1.414)
    mcts = MCTS(critic=None, exploration_constant=1.414)
    # Play a game: MCTS (White) vs Stockfish (Black)
    board = chess.Board()
    print("Starting game: MCTS (White) vs Stockfish (Black)")
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move, metrics = mcts.search(board, iterations=1000)
            print("MCTS (White) chooses:", move)
        else:
            result = engine.play(board, chess.engine.Limit(depth=0))
            move = result.move
            print("Stockfish (Black) chooses:", move)
        board.push(move)
        debug_display(board, metrics)
    print("Game over. Result:", board.result())

    engine.quit()

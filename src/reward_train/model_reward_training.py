
import tqdm
from copy import deepcopy
import sys
import os
import numpy as np
import chess.pgn
import chess
import json
import random
import torch
from src.models.bt4 import BT4
from src.data_process.vocab import PolicyIndex
from src.reward_train.reward_loss import MoveTraining
import chess.engine
import ray

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getcwd())
dir = os.environ.get("MODEL_DIR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = os.path.join(dir,"model.pth")
config_path = os.path.join(dir,"config.json")
config = json.load(open(config_path, "r"))
model = BT4(**config).to(device)
model.load_state_dict(torch.load(weights_path))
model.eval()
model_ref = deepcopy(model)
for param in model_ref.parameters():
    param.requires_grad = False
policy_manager = PolicyIndex()

class RewardModelBasic:
    def __init__(self, stockfish_path, depth, time):
        self.stockfish_path = stockfish_path
        self.reward_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.depth = depth
        self.time = time
    def get_reward(self,board:chess.Board):
        """ Evaluate the quality of the last move pushed to the board"""
        info = {}
        try:
            analysis = self.reward_engine.analyse(
                board, chess.engine.Limit(depth=self.depth, time=self.time)
            )
            score = analysis["score"].white().score(mate_score=10000)
            reward = np.tanh(score / 1000)  # Normalize to [-1, 1]
            if board.turn==chess.WHITE: # The move evaluated was played as black so we need black reward.
                reward=  -reward
        except chess.engine.EngineTerminatedError:
            reward = -1
            info["error"] = "EngineTerminatedError"
        return reward, info



    def get_rewards(self,move_indexes,boards, policy_manager):
        b,G = move_indexes.shape
        rewards = []
        for batch_idx in range(b):
            board:chess.Board = boards[batch_idx]
            reward_batch = []
            for sample_idx in range(G):
                board_for_move = board.copy()
                token = policy_manager.policy_index[move_indexes[batch_idx,sample_idx].item()]
                uci_ai = policy_manager.token_to_uci(board, token)
                # print("token :", token, " move : ", uci_ai)

                legal_moves = list(map( lambda x : x.uci(), board_for_move.legal_moves))
                if str(uci_ai) not in legal_moves:
                    uci_ai = legal_moves[random.randint(0,len(legal_moves)-1)]
                uci_ai = chess.Move.from_uci(uci_ai)
                board_for_move.push(uci_ai)
                reward, infos = self.get_reward(board_for_move)
                reward_batch.append(reward)
            rewards.append(reward_batch)
        rewards = np.array(rewards)
        return rewards


@ray.remote
class StockfishWorker:
    def __init__(self, stockfish_path, depth, time):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.depth = depth
        self.time = time

    def evaluate_position(self, board_fen):
        """ Evaluate a single position using Stockfish."""
        board = chess.Board(board_fen)
        try:
            analysis = self.engine.analyse(board, chess.engine.Limit(depth=self.depth, time=self.time))
            score = analysis["score"].white().score(mate_score=10000)
            reward = np.tanh(score / 1000)  # Normalize to [-1, 1]
            if board.turn == chess.WHITE:  # The move evaluated was played as black, so we need black reward.
                reward = -reward
        except chess.engine.EngineTerminatedError:
            reward = -1
        return reward

    def close(self):
        self.engine.quit()


class RewardModel:
    def __init__(self, stockfish_path, depth, time, num_engines=4):
        self.num_engines = num_engines
        self.workers = [StockfishWorker.remote(stockfish_path, depth, time) for _ in range(num_engines)]

    def get_rewards(self, move_indexes, boards, policy_manager):
        """ Get rewards from Stockfish in parallel using Ray actors."""
        b, G = move_indexes.shape
        tasks = []
        board_task= []
        for batch_idx, board in enumerate(boards):
            board_to_eval = board.copy()
            worker = self.workers[batch_idx % self.num_engines]
            board_task.append(worker.evaluate_position.remote(board_to_eval.fen()))
        eval_board = np.array(ray.get(board_task)).reshape(b, 1)
        for batch_idx in range(b):
            board = boards[batch_idx]
            for sample_idx in range(G):
                board_for_move = board.copy()
                token = policy_manager.policy_index[move_indexes[batch_idx, sample_idx].item()]
                uci_ai = policy_manager.token_to_uci(board, token)

                legal_moves = list(map(lambda x: x.uci(), board_for_move.legal_moves))
                if str(uci_ai) not in legal_moves:
                    uci_ai = legal_moves[random.randint(0, len(legal_moves) - 1)]
                uci_ai = chess.Move.from_uci(uci_ai)
                board_for_move.push(uci_ai)

                worker = self.workers[(batch_idx * G + sample_idx) % self.num_engines]
                tasks.append(worker.evaluate_position.remote(board_for_move.fen()))

        evals = ray.get(tasks)
        evals = np.array(evals).reshape(b, G)
        rewards = evals - eval_board
        return rewards


################### PARAMETERS
num_steps = 10
num_epochs=1000
G = 16
batch_size=128
stockfish_path = os.environ.get("STOCKFISH_PATH")
depth = 15
time = 1
epsilon_grpo = 0.2
num_engines = 16
kl_coef = 1
###################

model.train()
num_params = sum([p.numel() for p in model.parameters()])

print("nb params : ", num_params)

from src.data_process.data_gen import data_gen

ds = data_gen({'batch_size': batch_size, 'path_pgn': os.environ.get("PGN_DIR"), "with_fen": True})
opt = torch.optim.NAdam(model.parameters(), lr=5e-5)
reward_model = RewardModel(stockfish_path, depth, time, num_engines=num_engines)

import wandb
wandb.login(key=os.environ.get("WANDB_KEY"))
id = wandb.util.generate_id()
wandb.init(project='ChessRL_tuning', id=id, resume='allow')

from torch.amp import GradScaler, autocast

scaler = GradScaler("cuda")

best_accuracy = 0.0
best_loss = float('inf')
best_reward = -float('inf')
for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_reward = 0.0
    for step in range(num_steps):
        batch = ds.get_batch()
        x, y_true,L_fen = batch

        x1, x2 = x[0], x[1]
        # convert to pytorch tensors
        x1, x2, y_true = torch.from_numpy(x1).to("cuda"), torch.from_numpy(x2).to("cuda"), torch.from_numpy(y_true).to(
            "cuda")
        legal_mask = y_true >= 0.
        y_true = torch.nn.functional.relu(y_true)

        # Enable autocast for mixed precision
        with autocast('cuda'):
            logits, aux_loss = model(x1, x2)
            logits_ref, _ = model_ref(x1.clone().detach(), x2.clone().detach())
            proba_ref = torch.softmax(logits_ref, dim=-1)
            masked_proba = proba_ref * legal_mask  # Zero out illegal moves
            masked_proba /= masked_proba.sum(dim=-1, keepdim=True)
            indexes_evaluated = torch.multinomial(masked_proba, num_samples=G, replacement=True)
            boards = list(map(lambda x : chess.Board(x), L_fen))
            rewards= reward_model.get_rewards(indexes_evaluated, boards, policy_manager)
            loss,infos = MoveTraining.compute_loss_grpo(logits, logits_ref, indexes_evaluated, rewards,
                                                        epsilon=epsilon_grpo, kl_coef=kl_coef)

        opt.zero_grad()

        # Scale the loss and call backward
        scaler.scale(loss).backward()

        # Unscale the gradients and call the optimizer step
        scaler.step(opt)

        # Update the scaler
        scaler.update()
        with torch.no_grad():
            acc = torch.mean(
                torch.eq(torch.argmax(logits, dim=-1), torch.argmax(y_true, dim=-1)).to(
                    torch.float32)).data.cpu().numpy()

            legal_prob = ((legal_mask * torch.nn.functional.softmax(logits, dim=-1)).sum(dim=-1)).mean().item()
            # print(f"step : {step}, accuracy : {accuracy}, loss : {total_loss}", end="\r")

            epoch_loss += loss.item()
            epoch_accuracy += acc.item()
        mean_reward_gain = infos["reward_bar_model"] - infos["reward_bar_model_ref"]

        epoch_reward+= mean_reward_gain
        if step % 1 == 0:
            logs = {"accuracy": acc.item(), "loss": loss.item(), "legal_prob": legal_prob}
            logs.update(infos)
            total_step = num_steps * epoch + step
            wandb.log(
                logs,
                step=total_step
            )
            print(total_step, logs)

    # Calculate average loss and accuracy for the epoch
    epoch_loss /= num_steps
    epoch_accuracy /= num_steps
    epoch_reward /= num_steps
    print(f"epoch : {epoch}, loss: {epoch_loss}, accuracy: {epoch_accuracy}, reward: {epoch_reward}")

    # Save the model if it has the best accuracy or loss so far
    if epoch_reward > 0 :
        best_reward = epoch_reward
        best_loss = epoch_loss
        torch.save(model.state_dict(), f'grpo_best_model_{epoch}.pth')
        print(f"Model saved at epoch {epoch} with epoch reward gain {epoch_reward}")
        model_ref = deepcopy(model)
        for param in model_ref.parameters():
            param.requires_grad = False


from src.model_evaluation.bot_arena import ChessTournament
from src.model_evaluation.toy_bot import  RandomBot
from src.model_evaluation.StockfishBot import StockfishBot
from src.data_process.parse import dir_decorator, list_move_generator
generator = dir_decorator(list_move_generator, "../../pgn_data")
def gen(max_len = 10):
    for list_move in generator:
        L = [list_move[i].uci() for i in range(min(max_len, len(list_move)))]
        yield L


generator_listmove = gen()
# bot_names = ["RandomBot1", "RandomBot2"]
# bots = [ RandomBot(), RandomBot()]
import os

import torch
from src.model_evaluation.BT4Bot import BT4Bot
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device : ", device)
elo = 3000
TC = "600"
history = 7


from dotenv import load_dotenv
import os
load_dotenv()
pgn_dir = os.environ.get("PGN_DIR")
pgn_path = os.path.join(pgn_dir, os.listdir(pgn_dir)[0])
print(f"pgn file : {pgn_path}")
model_dir = os.environ.get("MODEL_DIR")
model_distil_w = os.environ.get("MODEL_DISTIL_WEIGHTS")
model_distil_c = os.environ.get("MODEL_DISTIL_CONFIG")
model_grpo_w = os.environ.get("MODEL_GRPO_WEIGHTS")
model_grpo_c = os.environ.get("MODEL_GRPO_CONFIG")
print(f"model dir : {model_dir}")


config_path_model = os.path.join(model_dir,"config.json")
weights_path_s = os.path.join(model_dir, "model.pth")
weights_paths = [weights_path_s, model_distil_w, model_grpo_w]
config_paths = [config_path_model, model_distil_c, model_grpo_c]
bt4_bots = [BT4Bot(weights_path,config_path, elo=elo, time_control=TC, history=history ) for weights_path,config_path in zip(weights_paths,config_paths)]
names = ["Supervised", "Distil", "GRPO"]

stockfish_path = os.environ.get("STOCKFISH_PATH")

bot_names = ["BT4", "Stockfish"]
elo=2000
stockfishbot = StockfishBot(stockfish_path, elo=elo, time=0.1)

L_bots = [[bt4, stockfishbot] for bt4 in bt4_bots]
# bot_names = ["RandomBot", "AlphaBetaBot"]
# bots = [ RandomBot(), AlphaBetaChessBot(search_depth=4)]
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
for bots,name in zip(L_bots,names) :
    tournament = ChessTournament(bot_names, bots, generator_listmove, games_per_match=20)
    tournament.run_1v1_matches()
    tournament.display_results()

    bot_log_history = bots[0].log_history
    from collections import defaultdict

    all_stock = defaultdict(list)

    for trace in bot_log_history:
        for i, entropy in trace:
            all_stock[i].append(entropy)  # Collect entropy values per index

    mean_entropy = {i: torch.tensor(values).mean().item() for i, values in all_stock.items()}

    sorted_indices = sorted(mean_entropy.keys())
    sorted_values = [mean_entropy[i] for i in sorted_indices]


    plt.scatter(sorted_indices, sorted_values, marker='o', linestyle='-', label=f"{name}")

plt.xlabel("game length")
plt.ylabel("Mean Entropy")
plt.title(f"Mean Entropy Across Traces")
plt.grid(True)
plt.legend()
plt.show()


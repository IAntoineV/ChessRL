# ChessRL ♟️

**ChessRL** is a deep reinforcement learning experiment designed to train an AI to play chess. The project follows a two-phase approach:

1. **Supervised Learning** – Train on strong player games to play "as a human".
2. **Reinforcement Learning** – Improve the model through self-play using GRPO, guided by distillation from Stockfish evaluations.

---

## 📥 Data Sources

To train the model, you'll need strong human player games:

- [NikoNoel’s Chess Game Database](https://database.nikonoel.fr/)
- [Lichess PGN exports](https://lichess.org/games/export)

---

## 🛠️ Installation

1. Clone the repo:

```bash
git clone https://github.com/IAntoineV/ChessRL
cd ChessRL
```

Configure your environment:

Install a stockfish compiled script from https://stockfishchess.org/download/

Copy "./.env copy" to "./.env"

Replace the value inside with your own values.

Install dependencies:
```bash
pip install -r requirements.txt
```


## 🧠 Tutorials

The `tutorials/` folder contains step-by-step Jupyter notebooks to help you understand and use the project effectively:

| Notebook           | Description                                                                         |
|--------------------|-------------------------------------------------------------------------------------|
| `tutorial1.ipynb`  | 🧩 **Token Overview**: Explains the list of playable tokens used in the model.      |
| `tutorial2.ipynb`  | 📘 **Supervised Learning**: Shows how to train the model on human games.            |
| `tutorial3.ipynb`  | 🔁 **Reinforcement Learning**: Demonstrates GRPO training + Stockfish distillation. |
| `tutorial4.ipynb`  | 📈 **Evaluation**: Evaluates the model's Elo using Stockfish.                       |

To run a tutorial, make sure your environment is properly set up and all dependencies are installed. You’ll also need a valid path to the Stockfish engine set in the `.env` file.

---

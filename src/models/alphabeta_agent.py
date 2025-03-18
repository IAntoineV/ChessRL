import sys
import os
sys.path.append(os.getcwd())
from src.reward_train.custom_board_reward import evaluate_board, order_moves
import chess



def alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool):
    """
    Recursive Alpha-beta pruning algorithm.
    Returns a tuple of (evaluation, best_move).
    """
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None

    best_move = None
    ## we order the lesgal moves for faster pruning (TO DO: IMPROVE order function)
    moves = order_moves(board)
    if maximizing_player:
        max_eval = -float('inf')
        for move in moves:
            board.push(move)
            eval_score, _ = alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in moves:
            board.push(move)
            eval_score, _ = alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval, best_move
import chess
import chess.pgn
import numpy as np
import torch
def fen_to_tensor(fen: str):
    board = chess.Board(fen)
    P = 19  # 12 planes for pieces + 1 for side to play + 1 for en passant + 4 for castling + 1 for 50-move rule
    tensor = np.zeros((8, 8, P), dtype=np.float32)
    
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
    }
    
    # Populate piece planes
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        plane = piece_map[piece.symbol()]
        tensor[7 - rank, file, plane] = 1.0  # Flip rank to align with standard board representation
    
    # Side to play plane
    tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0

    # En passant plane
    if board.ep_square is not None:
        rank, file = divmod(board.ep_square, 8)
        tensor[7 - rank, file, 13] = 1.0
    
    # Castling rights planes (4 total: white kingside, white queenside, black kingside, black queenside)
    tensor[:, :, 14] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[:, :, 15] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[:, :, 16] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[:, :, 17] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    
    # 50-move rule plane (normalized to [0,1])
    tensor[:, :, 18] = min(board.halfmove_clock / 100.0, 1.0)
    
    return tensor


if __name__ == "__main__":

    fen = "r1bq1rk1/2pp1ppp/p1n2n2/2b1p3/1pP1P3/1B1P1N2/PP3PPP/RNBQR1K1 b - c3 0 9"
    tensor = fen_to_tensor(fen)
    print(tensor.shape)  # Should output (8, 8, 19)
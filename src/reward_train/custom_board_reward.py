import chess

"""
See https://www.chessprogramming.org/Simplified_Evaluation_Function
 Tomasz Michniewski  evaluation function"
"""

# Piece values (in centipawns)
piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000  # Arbitrary high value
}

# Piece-square tables for positional evaluation
pawnEvalWhite = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, -20, -20, 10, 10,  5,
    5, -5, -10,  0,  0, -10, -5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0
]
pawnEvalBlack = list(reversed(pawnEvalWhite))

knightEvalWhite = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]

knightEvalBlack = list(reversed(knightEvalWhite))

bishopEvalWhite = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]
bishopEvalBlack = list(reversed(bishopEvalWhite))

rookEvalWhite = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0
]
rookEvalBlack = list(reversed(rookEvalWhite))

queenEvalWhite = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
]

queenEvalBlack = list(reversed(queenEvalWhite))

kingEvalWhite = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30
]
kingEvalBlack = list(reversed(kingEvalWhite))

kingEvalEndGameWhite = [
    50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30,  0,  0,  0,  0, -30, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -20, -10,  0,  0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50
]
kingEvalEndGameBlack = list(reversed(kingEvalEndGameWhite))
# Additional piece-square tables can be added similarly...

piece_square_tables = {
    chess.PAWN: (pawnEvalWhite, pawnEvalBlack),
    chess.KNIGHT: (knightEvalWhite, knightEvalBlack),
    chess.BISHOP: (bishopEvalWhite, bishopEvalBlack),
    chess.ROOK: (rookEvalWhite, rookEvalBlack),
    chess.QUEEN: (queenEvalWhite, queenEvalBlack),
}

import chess

def is_near_end_game(board: chess.Board) -> bool:
    """
    Determines if the game is near the endgame phase.
    - Both sides have no queens, or
    - A side with a queen has at most one minor piece (bishop/knight).
    """
    queens = {chess.WHITE: 0, chess.BLACK: 0}
    minor_pieces = {chess.WHITE: 0, chess.BLACK: 0}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.piece_type == chess.QUEEN:
                queens[piece.color] += 1
            elif piece.piece_type in {chess.BISHOP, chess.KNIGHT}:
                minor_pieces[piece.color] += 1

    # No queens on the board
    if queens[chess.WHITE] == 0 and queens[chess.BLACK] == 0:
        return True

    # Each side with a queen has at most one minor piece
    for color in [chess.WHITE, chess.BLACK]:
        if queens[color] > 0 and minor_pieces[color] > 1:
            return False

    return True


def evaluate_piece(piece: chess.Piece, square: chess.Square, endgame: bool) -> int:
    piece_type = piece.piece_type
    table = piece_square_tables.get(piece_type, ([], []))

    if piece_type == chess.KING:
        if endgame:
            table = (kingEvalEndGameWhite, kingEvalEndGameBlack)
        else:
            table = (kingEvalWhite, kingEvalBlack)
    
    if piece.color == chess.WHITE:
        return piece_values[piece_type] + table[0][square]
    else:
        return - (piece_values[piece_type] + table[1][square])


def evaluate_board(board: chess.Board) -> float:
    if board.is_checkmate():
        return -10000.0 if board.turn else 10000.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    endgame = is_near_end_game(board)
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            score += evaluate_piece(piece, square, endgame)
    
    return score if board.turn == chess.WHITE else -score


def order_moves(board: chess.Board) -> list[chess.Move]:
    ## TO CHANGE
    def move_score(move: chess.Move) -> float:
        if move.promotion:
            return float('inf') if board.turn == chess.WHITE else -float('inf')
        
        board.push(move)
        value = evaluate_board(board)
        board.pop()
        return value
    
    return sorted(board.legal_moves, key=move_score, reverse=board.turn == chess.WHITE)

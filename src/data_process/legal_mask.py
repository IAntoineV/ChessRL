import torch

def get_legal_move_mask(board,vocab_to_index, dtype=torch.float32, device="cpu"):
    mask = torch.zeros(len(vocab_to_index), dtype=dtype, device=device)
    indices = torch.tensor([vocab_to_index[str(move)] for move in board.legal_moves], device=device)
    mask.scatter_(0, indices, 1)
    return mask
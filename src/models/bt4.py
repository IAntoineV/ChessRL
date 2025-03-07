import torch
from mixture_of_experts import MoE
import sys
import os
import numpy as np
import tqdm
sys.path.append(os.getcwd())
import attentions as attn



move = np.arange(1, 8)

diag = np.array([
    move + move * 8,
    move - move * 8,
    move * -1 - move * 8,
    move * -1 + move * 8
])

orthog = np.array([
    move,
    move * -8,
    move * -1,
    move * 8
])

knight = np.array([
    [2 + 1 * 8],
    [2 - 1 * 8],
    [1 - 2 * 8],
    [-1 - 2 * 8],
    [-2 - 1 * 8],
    [-2 + 1 * 8],
    [-1 + 2 * 8],
    [1 + 2 * 8]
])

promos = np.array([2 * 8, 3 * 8, 4 * 8])
pawn_promotion = np.array([
    -1 + promos,
    0 + promos,
    1 + promos
])


def make_map():
    """theoretically possible put-down squares (numpy array) for each pick-up square (list element).
    squares are [0, 1, ..., 63] for [a1, b1, ..., h8]. squares after 63 are promotion squares.
    each successive "row" beyond 63 (ie. 64:72, 72:80, 80:88) are for over-promotions to queen, rook, and bishop;
    respectively. a pawn traverse to row 56:64 signifies a "default" promotion to a knight."""
    traversable = []
    for i in range(8):
        for j in range(8):
            sq = (8 * i + j)
            traversable.append(
                sq +
                np.sort(
                    np.int32(
                        np.concatenate((
                            orthog[0][:7 - j], orthog[2][:j], orthog[1][:i], orthog[3][:7 - i],
                            diag[0][:np.min((7 - i, 7 - j))], diag[3][:np.min((7 - i, j))],
                            diag[1][:np.min((i, 7 - j))], diag[2][:np.min((i, j))],
                            knight[0] if i < 7 and j < 6 else [], knight[1] if i > 0 and j < 6 else [],
                            knight[2] if i > 1 and j < 7 else [], knight[3] if i > 1 and j > 0 else [],
                            knight[4] if i > 0 and j > 1 else [], knight[5] if i < 7 and j > 1 else [],
                            knight[6] if i < 6 and j > 0 else [], knight[7] if i < 6 and j < 7 else [],
                            pawn_promotion[0] if i == 6 and j > 0 else [],
                            pawn_promotion[1] if i == 6 else [],
                            pawn_promotion[2] if i == 6 and j < 7 else [],
                        ))
                    )
                )
            )
    z = np.zeros((64 * 64 + 8 * 24, 1858), dtype=np.int32)
    # first loop for standard moves (for i in 0:1858, stride by 1)
    i = 0
    for pickup_index, putdown_indices in enumerate(traversable):
        for putdown_index in putdown_indices:
            if putdown_index < 64:
                z[putdown_index + (64 * pickup_index), i] = 1
                i += 1
    # second loop for promotions (for i in 1792:1858, stride by ls[j])
    j = 0
    j1 = np.array([3, -2, 3, -2, 3])
    j2 = np.array([3, 3, -5, 3, 3, -5, 3, 3, 1])
    ls = np.append(j1, 1)
    for k in range(6):
        ls = np.append(ls, j2)
    ls = np.append(ls, j1)
    ls = np.append(ls, 0)
    for pickup_index, putdown_indices in enumerate(traversable):
        for putdown_index in putdown_indices:
            if putdown_index >= 64:
                pickup_file = pickup_index % 8
                promotion_file = putdown_index % 8
                promotion_rank = (putdown_index // 8) - 8
                z[4096 + pickup_file * 24 + (promotion_file * 3 + promotion_rank), i] = 1
                i += ls[j]
                j += 1

    return z


class ApplyAttentionPolicyMap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.from_numpy(make_map()).to(torch.float32).to("cuda")

    def forward(self, logits, pp_logits):
        logits = torch.cat([torch.reshape(logits, [-1, 64 * 64]),
                            torch.reshape(pp_logits, [-1, 8 * 24])],
                           dim=1)
        return torch.matmul(logits, self.fc1)


class MaGating(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(64, hidden_dim))
        self.b = torch.nn.Parameter(torch.ones(64, hidden_dim))

    def forward(self, x):
        return x * torch.exp(self.a) + self.b


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.attention = attn.RelativeMultiHeadAttention2(d_model, num_heads, 0).to("cuda")
        self.norm1 = torch.nn.LayerNorm(d_model).to("cuda")
        self.norm2 = torch.nn.LayerNorm(d_model).to("cuda")

        # self.ff1 = torch.nn.Linear(d_model,d_ff).to("cuda")
        # self.ff2 = torch.nn.Linear(d_ff,d_model).to("cuda")
        self.moe = MoE(
            dim=d_model,
            num_experts=4,
            hidden_dim=d_ff,
            second_policy_train='random',  # in top_2 gating, policy for whether to use a second-place expert
            second_policy_eval='random',
            # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
            second_threshold_train=0.2,
            second_threshold_eval=0.2,
            capacity_factor_train=1.25,
            # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            capacity_factor_eval=2.,  # capacity_factor_* should be set to a value >=1
            loss_coef=1e-2
        )

        self.gelu = torch.nn.GELU().to("cuda")

    def forward(self, x, pos_enc):
        attn_out = self.attention(x, x, x, pos_enc)
        x = attn_out + x

        x = self.norm1(x)

        # y = self.ff1(x)
        # y = self.ff2(y)
        # y = self.gelu(y)
        y, loss = self.moe(x)

        y = y + x

        y = self.norm2(y)

        return y, loss


class AbsolutePositionalEncoder(torch.nn.Module):
    def __init__(self, d_model):
        super(AbsolutePositionalEncoder, self).__init__()
        self.position = torch.arange(64).unsqueeze(1)

        self.positional_encoding = torch.zeros(256, 64, d_model).to("cuda")

        _2i = torch.arange(0, d_model, step=2).float()

        self.positional_encoding[:, :, 0::2] = torch.sin(self.position / (10000 ** (_2i / d_model)))

        self.positional_encoding[:, :, 1::2] = torch.cos(self.position / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, _, _ = x.size()

        return self.positional_encoding[:batch_size, :, :]


class BT4(torch.nn.Module):
    def __init__(self, num_layers, d_model, d_ff, num_heads):
        super().__init__()

        self.d_model = d_model

        self.num_layers = num_layers

        self.layers = torch.nn.ModuleList([EncoderLayer(d_model, d_ff, num_heads) for _ in range(num_layers)])
        # self.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model,num_heads,d_ff,0,"gelu",batch_first=True),num_layers)

        self.linear1 = torch.nn.Linear(104, d_model)

        self.layernorm1 = torch.nn.LayerNorm(d_model)

        self.policy_tokens_lin = torch.nn.Linear(d_model, d_model)

        self.queries_pol = torch.nn.Linear(d_model, d_model)

        self.keys_pol = torch.nn.Linear(d_model, d_model)

        self.promo_offset = torch.nn.Linear(d_model, 4)

        self.positional = AbsolutePositionalEncoder(d_model)

        self.applyattn = ApplyAttentionPolicyMap()

        self.ma_gating = MaGating(d_model)

    def forward(self, x1, x2):
        aux_loss = 0
        x = torch.cat((x1, x2), dim=-1)
        # (B,8,8,104)

        # reshape
        x = torch.reshape(x, (-1, 64, 104))

        x = self.linear1(x)
        # add gelu
        x = torch.nn.GELU()(x)

        x = self.layernorm1(x)

        # add ma gating
        x = self.ma_gating(x)
        pos_enc = self.positional(x)
        for i in range(self.num_layers):
            x, loss_exp = self.layers[i](x, pos_enc)
            aux_loss += loss_exp
        # x = self.encoder(x)
        # policy tokens embedding
        policy_tokens = self.policy_tokens_lin(x)
        policy_tokens = torch.nn.GELU()(policy_tokens)

        queries = self.queries_pol(policy_tokens)

        keys = self.keys_pol(policy_tokens)

        matmul_qk = torch.matmul(queries, torch.transpose(keys, -2, -1))

        dk = torch.sqrt(torch.tensor(self.d_model))

        promotion_keys = keys[:, -8:, :]

        promotion_offsets = self.promo_offset(promotion_keys)

        promotion_offsets = torch.transpose(promotion_offsets, -2, -1) * dk

        promotion_offsets = promotion_offsets[:,
                            :3, :] + promotion_offsets[:, 3:4, :]

        n_promo_logits = matmul_qk[:, -16:-8, -8:]

        q_promo_logits = torch.unsqueeze(
            n_promo_logits + promotion_offsets[:, 0:1, :], dim=3)  # Bx8x8x1
        r_promo_logits = torch.unsqueeze(
            n_promo_logits + promotion_offsets[:, 1:2, :], dim=3)
        b_promo_logits = torch.unsqueeze(
            n_promo_logits + promotion_offsets[:, 2:3, :], dim=3)
        promotion_logits = torch.cat(
            [q_promo_logits, r_promo_logits, b_promo_logits], dim=3)  # Bx8x8x3
        # logits now alternate a7a8q,a7a8r,a7a8b,...,
        promotion_logits = torch.reshape(promotion_logits, [-1, 8, 24])

        # scale the logits by dividing them by sqrt(d_model) to stabilize gradients
        # Bx8x24 (8 from-squares, 3x8 promotions)
        promotion_logits = promotion_logits / dk
        # Bx64x64 (64 from-squares, 64 to-squares)
        policy_attn_logits = matmul_qk / dk

        h_fc1 = self.applyattn(policy_attn_logits, promotion_logits)

        return h_fc1, loss_exp

if __name__=="__main__":
    num_layers = 6
    d_model = 640
    d_ff = 640
    num_heads = 32

    bt4 = BT4(num_layers=num_layers, d_model=d_model, d_ff=d_ff, num_heads=num_heads).to("cuda")


    num_params = sum([ p.numel() for p in bt4.parameters()])

    print("nb params : ", num_params)

    from src.data_process.data_gen import data_gen

    ds = data_gen({'batch_size': 256, 'path_pgn': '/home/antoine/Bureau/3A/3A_RL/ChessRL/pgn_data/'})

    opt = torch.optim.NAdam(bt4.parameters(), lr=5e-5)

    import wandb

    id = wandb.util.generate_id()
    wandb.init(project='ChessRL_pretrain', id=id, resume='allow')


    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler()

    best_accuracy = 0.0
    best_loss = float('inf')

    for epoch in range(1000):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for step in range(1000):
            batch = ds.get_batch()
            x, y_true = batch

            x1, x2 = x[0], x[1]
            # convert to pytorch tensors
            x1, x2, y_true = torch.from_numpy(x1).to("cuda"), torch.from_numpy(x2).to("cuda"), torch.from_numpy(y_true).to(
                "cuda")
            legal_mask = y_true>=0.
            y_true = torch.nn.functional.relu(y_true)

            # Enable autocast for mixed precision
            with autocast():
                logits, aux_loss = bt4(x1, x2)
                loss = torch.nn.functional.cross_entropy(logits, y_true) + aux_loss
            opt.zero_grad()

            # Scale the loss and call backward
            scaler.scale(loss).backward()

            # Unscale the gradients and call the optimizer step
            scaler.step(opt)

            # Update the scaler
            scaler.update()

            with torch.no_grad():
                acc = torch.mean(
                    torch.eq(torch.argmax(logits, dim=-1), torch.argmax(y_true, dim=-1)).to(torch.float32)).data.cpu().numpy()

                legal_prob = ((legal_mask * torch.nn.functional.softmax(logits, dim=-1)).sum(dim=-1)).mean().item()
                #print(f"step : {step}, accuracy : {accuracy}, loss : {total_loss}", end="\r")

                epoch_loss += loss.item()
                epoch_accuracy += acc

            if step % 10==0:
                wandb.log({"accuracy": acc, "loss": loss.item(), "legal_prob": legal_prob}, step=1000* epoch+ step)

        # Calculate average loss and accuracy for the epoch
        epoch_loss /= 1000
        epoch_accuracy /= 1000

        print(f"epoch : {epoch}, loss: {epoch_loss}, accuracy: {epoch_accuracy}")

        # Save the model if it has the best accuracy or loss so far
        if epoch_accuracy > best_accuracy or epoch_loss < best_loss:
            best_accuracy = epoch_accuracy
            best_loss = epoch_loss
            torch.save(bt4.state_dict(), f'best_model.pth')
            print(f"Model saved at epoch {epoch} with accuracy {best_accuracy} and loss {best_loss}")

from typing_extensions import override
from src.models.bt4 import EncoderLayer
from src.models.bt4 import BT4
import torch

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.layer1(x))
        x = self.activation_fn(self.layer2(x))
        x = self.layer3(x)
        return x


class BT4WrapperWEncodedLayer(BT4):
    def __init__(self, num_layers, d_model, d_ff, num_heads):
        super().__init__(num_layers, d_model, d_ff, num_heads)


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

        return h_fc1, policy_tokens, pos_enc


class BT4PolicyValue(nn.Module):
    def __init__(self, bt4_model: BT4WrapperWEncodedLayer, d_model_value, d_ff_value, num_layers_value, num_heads_value, encoded_dim):
        super().__init__()
        activation_fn = torch.nn.GELU()
        self.bt4model = bt4_model
        self.d_model_value = d_model_value
        self.d_ff_value = d_ff_value
        self.num_heads_value = num_heads_value
        self.num_layers_value = num_layers_value

        # Replace linear layers with MLP
        self.encoded_board_first_layer = MLP(self.bt4model.d_model, d_model_value, d_model_value, activation_fn)
        self.encoded_pos_encoding_layer = MLP(self.bt4model.d_model, d_model_value, d_model_value, activation_fn)

        self.layers_value = nn.ModuleList([EncoderLayer(self.d_model_value, self.d_ff_value, self.num_layers_value)
                                           for _ in range(self.num_layers_value)])

        self.last_linear = MLP(d_model_value, d_ff_value, encoded_dim, activation_fn)
        self.value_linear = MLP(encoded_dim, d_model_value, 1, activation_fn)
    def forward(self, x_input):
        Xs = x_input[:, :, :, :-2]
        Es = x_input[:, :, :, -2:]
        log_prob, encoded_board, pos_enc = self.bt4model(Xs, Es) # log_prob (1858,) log prob, encoded baord (b,64,bt4model.d_model)
        x = self.encoded_board_first_layer(encoded_board)
        pos_enc = self.encoded_pos_encoding_layer(pos_enc)
        for i in range(self.num_layers_value):
            x, _ = self.layers_value[i](x, pos_enc)
        encoded4value = self.last_linear(x)
        encoded4value = torch.nn.GELU()(encoded4value) # (b,64,encoded_dim)
        pulled_value = encoded4value.mean(dim=1)
        value = self.value_linear(pulled_value)

        return log_prob, value


if __name__=="__main__":
    # Show how to load the wrapped version.
    import json
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device : ", device)
    elo = 2700
    TC = "600"
    history = 7
    pgn_path = "../../pgn_data_example/pgn_example.pgn"

    dir = "../../models_saves/model_1/"
    weights_path = dir + "model.pth"
    config_path = dir + "config.json"

    config = json.load(open(config_path, "r"))
    bt4_model = BT4WrapperWEncodedLayer(**config).to(device)
    bt4_model.load_state_dict(torch.load(weights_path))
    bt4_model.eval()
    model = BT4PolicyValue(bt4_model, 200, 400, 2, 20, 200).to(device)
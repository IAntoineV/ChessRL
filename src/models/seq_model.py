import torch
import torch.nn as nn
from models.layers import SeqModelConfig, Block, LayerNorm, AttentionPooling, FenEncoder
class SeqModel(nn.Module):
    def __init__(self, config: SeqModelConfig):
        super().__init__()
        self.config = config

        # Projection layer to transform input_dim -> n_embd
        self.input_proj = FenEncoder(d_model= config.n_embd,num_planes = config.input_dim)

        # Learnable positional embeddings (matches seq_len)
        self.position_embedding = nn.Parameter(torch.randn(1, config.seq_len, config.n_embd))

        # Transformer Blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # Final LayerNorm and Output Projection
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.head = nn.Sequential(AttentionPooling(config.n_embd),
                                  nn.Linear(config.n_embd, config.output_dim, bias=False),
                                  nn.LogSoftmax(dim=-1))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0.0, std=0.02)  # Initialize positional embeddings

    def forward(self, x):
        B, T, C = x.shape
        assert C == self.config.input_dim, f"Input dim {C} must match config.input_dim {self.config.input_dim}"
        assert T == self.config.seq_len, f"Sequence length {T} must match config.seq_len {self.config.seq_len}!"

        # (B,seq_len,input_dim) -> (B,seq_len,n_embd))
        x = self.input_proj(x)

        # Add learnable positional embeddings
        x = x + self.position_embedding

        # Transformer layers
        x = self.blocks(x)
        x = self.ln_f(x)

        # Output projection
        return self.head(x)  # (B, seq_len, n_embd)

# Example usage
if __name__ == "__main__":
    config = SeqModelConfig(output_dim=200, input_dim=19, n_embd=48, seq_len=64)
    encoding_model = SeqModel(config)
    sample_input = torch.randn(2, 64, 19)  # Batch of 2 sequences, each of length 64 with 19 features
    output = encoding_model(sample_input)
    print(output.shape)  # Expected (2, 200)
    num_params_learnable = sum(p.numel() for p in encoding_model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in encoding_model.parameters())
    print(f"params : {num_params} \n learnable params : {num_params_learnable}")


import torch
import torch.nn as nn
from torch.nn import RMSNorm
import math
from torch.nn import TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[: x.size(0)]
        return x.transpose(0, 1)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000, base=10000):
        super(RotaryPositionalEmbedding, self).__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x):
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class TransformerEncoderLayerWithRoPE(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="gelu", 
                 use_rms_norm=False, norm_first=False, bias=True, layer_norm_eps=1e-5, 
                 device=None, dtype=None):
        super(TransformerEncoderLayerWithRoPE, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.d_model = d_model
        self.nhead = nhead
        self.norm_first = norm_first
        
        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        
        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = activation
        
        # Layer normalization
        if use_rms_norm:
            self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
            self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, cos=None, sin=None, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Self-attention block
        if self.norm_first:
            x = src + self._sa_block(
                self.norm1(src), cos, sin, src_mask, src_key_padding_mask, is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                src + self._sa_block(src, cos, sin, src_mask, src_key_padding_mask, is_causal)
            )
            x = self.norm2(x + self._ff_block(x))
        
        return x
    
    def _sa_block(self, x, cos=None, sin=None, attn_mask=None, key_padding_mask=None, is_causal=False):
        batch_size, seq_len, d_model = x.shape
        head_dim = d_model // self.nhead
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, head_dim).transpose(1, 2)
        
        # Apply RoPE if provided
        if cos is not None and sin is not None:
            # cos and sin have shape [seq_len, d_model] but we need [seq_len, head_dim//2]
            # Take only the first head_dim//2 dimensions
            cos_head = cos[:, :head_dim//2]  # [seq_len, head_dim//2]
            sin_head = sin[:, :head_dim//2]  # [seq_len, head_dim//2]
            
            # Expand to match the shape of q and k: [batch_size, nhead, seq_len, head_dim//2]
            cos_expanded = cos_head.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
            sin_expanded = sin_head.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
            
            # Apply RoPE to Q and K
            q_rotated = apply_rotary_pos_emb(q, cos_expanded, sin_expanded)
            k_rotated = apply_rotary_pos_emb(k, cos_expanded, sin_expanded)
        else:
            q_rotated = q
            k_rotated = k
        
        # Scaled dot-product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q_rotated, k_rotated, v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout1.p if self.training else 0.0,
            is_causal=is_causal
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_output = self.out_proj(attn_output)
        
        return self.dropout1(attn_output)
    
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        dropout=0.1,
        use_rms_norm=False,
        use_rotary_pos_emb=False,
    ):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.use_rotary_pos_emb = use_rotary_pos_emb
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = RotaryPositionalEmbedding(d_model) if use_rotary_pos_emb else PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(d_model, nhead, dim_feedforward, dropout, activation="gelu", use_rms_norm=use_rms_norm)
            for _ in range(num_encoder_layers)
        ])
        self.output_norm = RMSNorm(d_model) if use_rms_norm else nn.LayerNorm(d_model)
        self.output_linear = nn.Linear(d_model, input_dim)

    def forward(self, src):
        src = self.input_linear(src) * math.sqrt(self.d_model)
        
        if self.use_rotary_pos_emb:
            cos, sin = self.pos_encoder(src)
            for layer in self.encoder_layers:
                src = layer(src, cos, sin)
        else:
            src = self.pos_encoder(src)
            for layer in self.encoder_layers:
                src = layer(src, None, None)
        
        output = src[:, -1, :]
        output = self.output_norm(output)
        output = self.output_linear(output)
        return output

if __name__ == "__main__":
    model = TransformerModel(
        input_dim=32,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.15,
        use_rms_norm=True,
        use_rotary_pos_emb=True
    )
    print(model)
    print(f"Model size: {sum(p.numel() for p in model.parameters())}")
    print(f"Output shape: {model.forward(torch.randn(1, 100, 32)).shape}")
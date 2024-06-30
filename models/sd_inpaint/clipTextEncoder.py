import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import OrderedDict, Union
from . import clipTokenizer

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype)
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.in_proj_weight = nn.Parameter(torch.randn(3*d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.randn(3*d_model))

        self.qkv = nn.Linear(d_model, 3*d_model)
        self.qkv.weight = self.in_proj_weight
        self.qkv.bias = self.in_proj_bias

        self.out_proj = Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Tensor):
        bs, seq_len, embed_dim = x.shape

        qkv = self.qkv(x) # (b, num_tokens, 3*embed_dim)
        qkv = qkv.view(bs, seq_len, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, bs, n_head, num_tokens, head_dim)
        xq, xk, xv = qkv

        context_vec = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask)
        context_vec = context_vec.transpose(1, 2).contiguous().view(bs, seq_len, self.n_head*self.head_dim)
        context_vec = self.out_proj(context_vec)

        return context_vec

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, mask=self.attn_mask)
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.ln_final = LayerNorm(transformer_width)
    
    @property
    def dtype(self):
        return torch.float32

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) # [1, 77, 768]
        # [b, 77, 768] @ [768, 49408] = [b, 77, 49408]
        return x

def build_model(state_dict: dict):
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0] # 768
    transformer_heads = transformer_width // 64                # 12
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    
    # print(f"embed dim: {embed_dim}")
    # print(f"context_length: {context_length}")
    # print(f"vocab_size: {vocab_size}")
    # print(f"transformer width: {transformer_width}")
    # print(f"transformer heads: {transformer_heads}")
    # print(f"transformer layers: {transformer_layers}")
    
    text_model = CLIP(embed_dim, context_length, vocab_size,
                 transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    text_model.load_state_dict(state_dict, strict=False)
    return text_model

def load(model_path: str, device: Union[str, torch.device] = "cuda", jit: bool = False):
    """Load a CLIP model
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    with open(model_path, 'rb') as opened_file:
        # loading JIT archive
        model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
        state_dict = None
    
    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model
    else:
        print("[ERROR]: JIT must be False.")

if __name__ == "__main__":
    model = load(model_path="/home/shashank/Documents/AI-ML/audio-to-image/ViT-L-14.pt")
    # print(model)
    tokens = clipTokenizer.tokenize("red tshirt").to("cuda")
    print(model.encode_text(tokens))

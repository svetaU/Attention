import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import einops
from einops import rearrange
from einops import repeat

class NiptMultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim, heads=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.dim_head = (int(self.embed_dim / self.heads))
        _dim = self.dim_head * self.heads
        self.to_qvk = nn.Linear(self.embed_dim, _dim * 3, bias=False)
        self.last_linear = nn.Linear(_dim, self.embed_dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.to_qvk.weight)
        nn.init.xavier_uniform_(self.last_linear.weight)

    def forward(self, x, mask=None, return_attention=False):
        assert x.dim() == 3
        qkv = self.to_qvk(x)
        q, k, v = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d ', k=3, h=self.heads))
        scaled_dot_prod = torch.einsum('b h i d , b h j d -> b h i j', q, k) * self.scale_factor
        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[2:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)
        values = torch.einsum('b h i j , b h j d -> b h i d', attention, v)
        values = rearrange(values, 'b h t d -> b t (h d)')
        output = self.last_linear(values)
        if return_attention:
            return output, attention
        else:
            return output


class NiptEncoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads=2, dim_linear_block=1024, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_linear_block = dim_linear_block
        self.dropout = dropout
        self.attn_layer = NiptMultiHeadSelfAttention(embed_dim=self.embed_dim,
                                                     heads=self.num_heads)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.drop = nn.Dropout(self.dropout)
        self.linear_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.dim_linear_block),
            nn.Dropout(self.dropout),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_linear_block, self.embed_dim)
        )

    def forward(self, x, mask=None):
        y = self.norm1(self.drop(self.attn_layer(x, mask)) + x)
        return self.norm2(self.linear_net(y) + y)


class NiptAttentionEncoder(nn.Module):

    def __init__(self, embed_dim, num_heads, dim_linear_block, dropout, num_layers=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_linear_block = dim_linear_block
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList([NiptEncoderBlock(embed_dim=self.embed_dim,
                                                      num_heads=self.num_heads,
                                                      dim_linear_block=self.dim_linear_block,
                                                      dropout=self.dropout) for _ in range(self.num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention(self,x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.attn_layer(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x, mask=mask)
        return attention_maps

class AttnClassifier(nn.Module):
    def __init__(self, embed_dim, num_tokens, num_heads,
                 num_layers, model_dim, dropout=0.0, input_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.dropout = dropout
        self.input_dropout = input_dropout
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.input_dropout),
            nn.Linear(self.embed_dim, self.model_dim)
        )
        # Transformer
        self.transformer = NiptAttentionEncoder(
            embed_dim=self.model_dim,
            num_layers=self.num_layers,
            dim_linear_block=2*self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout)
        # Output classifier per sequence element
        self.output_net = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.model_dim, 1),
            nn.ReLU(inplace=True)
        )
        self.prob = nn.Sequential(
            nn.Linear(self.num_tokens,1)
        )

    def forward(self, x, mask=None, add_positional_encoding=False):
        x = self.input_net(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        x = rearrange(x, 'b c h  -> b (c h)')
        x = self.prob(x)
        x = torch.sigmoid(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=False):
        x = self.input_net(x)
        attention_maps = self.transformer.get_attention(x, mask=mask)
        return attention_maps


class TrainerWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def _calculate_loss(self, batch, mode="train"):
        # Fetch data (labels are floats 0. or 1.)
        inp_data, labels = batch

        # Perform prediction and calculate loss and accuracy
        preds = self.net(inp_data, add_positional_encoding=False)
        #loss = F.cross_entropy(preds.view(-1), labels.view(-1))
        loss = F.binary_cross_entropy(preds.view(-1), labels)
        acc = ((preds.view(-1) > 0.5) == (labels > 0.)).float().mean()
        score_class1 = preds[labels > 0.].mean()
        score_class2 = preds[labels < 1.].mean()

        return loss, acc, preds, score_class1, score_class2

    def get_attention_maps(self, x, mask=None, add_positional_encoding=False):
        attention_maps = self.net.get_attention_maps(x, mask, add_positional_encoding=False)
        return attention_maps

    @torch.no_grad()
    def acc(self, batch):
        inp_data, labels = batch
        preds = self.net(inp_data, add_positional_encoding=False)
        acc = ((preds.view(-1) > 0.5) == (labels > 0.)).float().mean()
        return acc, preds, labels

    def forward(self, x, **kwargs):
        loss, acc, preds, s1, s2 = self._calculate_loss(x, mode="train")
        return loss, acc, preds, s1, s2
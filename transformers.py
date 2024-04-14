
import torch
import torch.nn as nn

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=500, dropout_proba=0.1):
        super().__init__()
        self.max_seq_len=max_seq_len
        self.d_model=d_model

        pe_table=self.get_pe_table()
        self.register_buffer('pe_table' , pe_table)

        self.dropout=nn.Dropout(dropout_proba) 

    def get_pe_table(self):
        position_idxs=torch.arange(self.max_seq_len).unsqueeze(1) 
        embedding_idxs=torch.arange(self.d_model).unsqueeze(0)
        
        angle_rads = position_idxs / 10000 ** (2*(embedding_idxs//2)/self.d_model)

        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        pe_table = angle_rads.unsqueeze(0) # So we can apply it to a batch

        return pe_table     # (1, M, D)

    def forward(self, embeddings_batch):
        seq_len = embeddings_batch.size(1)
        pe_batch = self.pe_table[:, :seq_len].clone().detach()
        return self.dropout(embeddings_batch + pe_batch)

class ScaledDotProductAttn(nn.Module):
    def __init__(self, d_head) -> None:
        super().__init__()
        self.d_head = d_head

    def forward(self, q, k, v, mask=None):
        attention_weight = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_weights = attention_weight / math.sqrt(self.d_head)
        if mask:
            scaled_attention_weights = scaled_attention_weights.masked_fill(mask == 0, float('-inf'))
        scaled_attention_weights = nn.functional.softmax(scaled_attention_weights)

        res = torch.matmul(scaled_attention_weights, v)
        return res


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_heads) -> None:
        super().__init__()
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.dim = d_model // n_heads
        self.scaledattn = ScaledDotProductAttn(self.dim)
        self.q = nn.Linear(self.dim, self.dim)
        self.k = nn.Linear(self.dim, self.dim)
        self.v = nn.Linear(self.dim, self.dim)
        self.W = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # split
        q = q.view(q.size(0), q.size(1), self.n_heads, self.dim)
        k = k.view(k.size(0), k.size(1), self.n_heads, self.dim)
        v = v.view(v.size(0), v.size(1), self.n_heads, self.dim)

        # linear
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # attention
        attention_output = self.scaledattn(q, k, v, mask)

        # concat
        attention_output = attention_output.transpose(1,2).contiguous()
        attention_output = attention_output.view(attention_output.size(0), attention_output.size(1), -1)

        # attention
        res = self.W(attention_output)
        return res

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, d_model, bias=True)
        
    def forward(self, x):
        x = nn.ReLU(self.linear1(x))
        x = self.linear2(x)
        return x
    
class AddNorm(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, residual):
        return self.layernorm(x + residual)

class FeedForwardMoE(nn.Module):
    def __init__(self, d_model, hidden_size, num_experts) -> None:
        super().__init__()

        self.gate = nn.Linear(d_model, num_experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        gating_weights = nn.functional.softmax(self.gate(x), dim=-1)

        expert_outputs = None

        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            expert_contribution = gating_weights[:, :, i].unsqueeze(2) * expert_output
            if expert_outputs is None:
                expert_outputs = torch.zeros_like(expert_output)
            expert_outputs += expert_contribution
        
        return expert_outputs


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_size, MoE) -> None:
        super().__init__()
        self.attn = MultiHeadAttn(d_model, n_heads)
        self.addnorm1 = AddNorm(d_model)
        if MoE:
            self.ffn = FeedForwardMoE(d_model, hidden_size, MoE)
        else:
            self.ffn = FeedForward(d_model, hidden_size)
        self.addnorm2 = AddNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x, mask)
        x = self.addnorm1(x, attn_out)
        ffn_out = self.ffn(x)
        x = self.addnorm2(x, ffn_out)
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, hidden_size, MoE, n_blocks) -> None:
        super().__init__()
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, n_heads, hidden_size, MoE) for _ in range(n_blocks)
        ])

    def forward(self, x, mask=None):
        for block in self.encoder:
            x = block(x, mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_size, MoE) -> None:
        super().__init__()
        self.attn1 = MultiHeadAttn(d_model, n_heads)
        self.addnorm1 = AddNorm(d_model)
        self.attn2 = MultiHeadAttn(d_model, n_heads)
        self.addnorm2 = AddNorm(d_model)
        if MoE:
            self.ffn = FeedForwardMoE(d_model, hidden_size, MoE)
        else:
            self.ffn = FeedForward(d_model, hidden_size)
        self.addnorm3 = AddNorm(d_model)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        attn1 = self.attn1(x, x, x, trg_mask)
        x = self.addnorm1(x, attn1)
        attn2 = self.attn2(x, encoder_output, encoder_output, src_mask)
        x = self.addnorm2(x, attn2)
        ffn_out = self.ffn(x)
        x = self.addnorm3(x, ffn_out)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, hidden_size, MoE, n_blocks) -> None:
        super().__init__()
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, n_heads, hidden_size, MoE) for _ in range(n_blocks)
        ])

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for block in self.decoder:
            x = block(x, encoder_output, src_mask, trg_mask)
        return x

class TransformerMoE(nn.Module):
    def __init__(self, d_model, src_vocab_size, trg_vocab_size, n_heads, hidden_size, n_blocks, MoE=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.src_pos_embedding = PositionalEncoding(d_model)
        self.encoder = Encoder(d_model, n_heads, hidden_size, MoE, n_blocks)

        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.trg_pos_embedding = PositionalEncoding(d_model)
        self.decoder = Decoder(d_model, n_heads, hidden_size, MoE, n_blocks)

        self.linear = nn.Linear(d_model, trg_vocab_size)

        self.src_embedding.weight = self.trg_embedding.weight
        self.linear.weight = self.trg_embedding.weight

    def encode(self, src_token_ids, src_mask):
        src_embeddings = self.src_embedding(src_token_ids) * math.sqrt(self.d_model)
        src_embeddings = self.src_pos_embedding(src_embeddings)
        encoder_outputs = self.encoder(src_embeddings, src_mask)

        return encoder_outputs

    def decode(self, trg_token_ids, encoder_outputs, src_mask, trg_mask):
        trg_embeddings = self.trg_embedding(trg_token_ids) * math.sqrt(self.d_model)
        trg_embeddings = self.trg_pos_embedding(trg_embeddings) 
        decoder_outputs = self.decoder(trg_embeddings, encoder_outputs, src_mask, trg_mask)

        linear_out = self.linear(decoder_outputs) # (batch_size, trg_seq_len, trg_vocab_size)

        return linear_out

    def forward(self, src_token_ids, trg_token_ids, src_mask, trg_mask):

        encoder_outputs= self.encode(src_token_ids, src_mask) # (batch_size, src_seq_len, d_model)
        decoder_outputs= self.decode(trg_token_ids, encoder_outputs, src_mask, trg_mask) # (batch_size, trg_seq_len, d_model)

        return decoder_outputs 


d_model = 512
src_vocab_size = 1000
trg_vocab_size = 1000
n_heads = 8
hidden_size = 2048
n_blocks = 6
MoE = 4

transformer_moe = TransformerMoE(d_model, src_vocab_size, trg_vocab_size, n_heads, hidden_size, n_blocks, MoE)

batch_size = 5
src_seq_len = 20
trg_seq_len = 20

src_token_ids = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
trg_token_ids = torch.randint(0, trg_vocab_size, (batch_size, trg_seq_len))

src_mask = None
trg_mask = None

outputs = transformer_moe(src_token_ids, trg_token_ids, src_mask, trg_mask)

print("Output shape:", outputs.shape)

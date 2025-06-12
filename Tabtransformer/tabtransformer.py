import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model   = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        query = self.wq(query)
        key   = self.wk(key)
        value = self.wv(value)

        query = self.split_heads(query, batch_size)
        key   = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention_logits = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)

        output = torch.matmul(attention_weights, value)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.dense(output)

        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu    = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output    = self.dropout1(attn_output)
        out1           = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2       = self.layernorm2(out1 + ffn_output)

        return out2

class TabTransformer(nn.Module):
    def __init__(self, num_features, d_model, num_heads, num_layers, d_ff, num_classes, dropout_rate=0.1):
        super(TabTransformer, self).__init__()

        self.embedding      = nn.Linear(num_features, d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)])
        self.final_layer    = nn.Linear(d_model, num_classes)
        self.dropout        = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.final_layer(x)
        x = torch.sigmoid(x)

        return x
        
        
        
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Linear, ReLU
import itertools


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        # self.values = lora.Linear(self.head_dim, self.head_dim, bias=False, r=16)
        # self.keys = lora.Linear(self.head_dim, self.head_dim, bias=False, r=16)
        # self.queries = lora.Linear(self.head_dim, self.head_dim, bias=False, r=16)
        # self.fc_out = lora.Linear(heads * self.head_dim, embed_size, r=16)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.BatchNorm1d(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        

        #out = self.dropout(forward + x)
        return out + attention

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class Transformer(nn.Module):
    def __init__(
        self,
        out_size,
        src_vocab_size,
        src_pad_idx,
        embed_size=256,
        num_layers=3,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.device = device
        self.fc1 = nn.Linear(max_length*embed_size, out_size)
        self.out = nn.Linear(out_size, 1)
        
        self.bn = nn.BatchNorm1d(out_size)
        self.ln = nn.LayerNorm(out_size)
        self.fco = nn.Sequential(
            nn.Linear(out_size*2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )


    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)


    def forward(self, src):
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)
        enc_src = enc_src.reshape(enc_src.shape[0], -1)
        feature = self.fc1(enc_src)
        x = self.bn(feature)


        x = self.out(x)

        
        return x




class VirusCNN(nn.Module):
    def __init__(self, channel='both', rev_comp=False, share_weight=False) -> None:
        super().__init__()

        self.channel = channel
        self.rev_comp = rev_comp
        self.share_weight = share_weight

        if self.channel == 'both':
            fc_layer_input_dim = 256
        else:
            fc_layer_input_dim = 256
        self.fc = nn.Sequential(
            nn.Linear(fc_layer_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )

        self.codon_channel_num = 64

        self.base_channel1 = nn.Sequential(
            nn.Conv2d(1, 64, (6, 4)),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=2),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )



        self.codon_channel1 = nn.Sequential(
            nn.Conv2d(1, 64, (6, self.codon_channel_num)),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=2),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )



        self.dropout = nn.Dropout(p=0.2, inplace=True)

        self.codon_transformer = CodonTransformer()

        
    @staticmethod
    def build_rev_comp(x: torch.Tensor):
        return torch.flip(x, [-1, -2])

    def forward(self, x):
       
        x1 = self.base_channel1(x)
        # x = self.codon_transformer(x)
        # x = self.codon_channel1(x)

        z = x1
        # z = torch.cat((x, x1), dim=1)
        z = self.dropout(z)
        # z = torch.cat((z, y), dim=1)
        z = self.fc(z)

        return z


class CodonTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.codon_channel_num = 64
        self.codon_transformer = torch.zeros(64, 1, 3, 4)
        indicies = itertools.product(range(4), repeat=3)
        for i in range(self.codon_transformer.shape[0]):
            index = next(indicies)
            for j in range(3):
                self.codon_transformer[i, 0, j, index[j]] = 1
        self.codon_transformer = nn.Parameter(self.codon_transformer, requires_grad=False)
        self.padding_layers = [nn.ZeroPad2d((0, 0, 0, 2)), nn.ZeroPad2d((0, 0, 0, 1))]


    def forward(self, x):
        mod_len = int(x.shape[2] % 3)
        if mod_len != 2:
            x = self.padding_layers[mod_len](x)
        x = F.conv2d(x, self.codon_transformer) - 2
        x = F.relu(x)
        x = x.flatten(start_dim=2)

        x = x.view(-1, self.codon_channel_num, int(x.shape[2]//3), 3)
        x = x.transpose(2, 3)
        x = x.reshape(-1, 1, self.codon_channel_num, x.shape[-1]*3).transpose(2, 3)
        return x
    

class SiameseNetwork(nn.Module):
    def __init__(self, channel_num, nb_filter1, filter_len1, nb_dense, dropout_pool, dropout_dense):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channel_num, out_channels=nb_filter1, kernel_size=filter_len1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout_pool = nn.Dropout(dropout_pool)
        self.fc1 = nn.Linear(nb_filter1, nb_dense)
        self.dropout_dense = nn.Dropout(dropout_dense)
        self.fc2 = nn.Linear(nb_dense, 1)

    def forward_one(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout_pool(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_dense(x)
        x = torch.sigmoid(self.fc2(x))
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return (output1 + output2) / 2




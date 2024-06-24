import torch.nn.init as init
import copy
from modeling import units
from utils.layers import *
from modeling.graph_seq_emb import GraphEncoder, SequenceEncoder


class Embedding(torch.nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                                        max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                        sparse=sparse, _weight=_weight)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):

        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):

        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        pos = np.zeros([len(input_len), max_len])
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                pos[ind, pos_ind - 1] = pos_ind
        input_pos = tensor(pos)
        return self.position_encoding(input_pos), input_pos


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        # self.linear_k = nn.Linear(model_dim+44, self.dim_per_head * num_heads)
        # self.linear_v = nn.Linear(model_dim+44, self.dim_per_head * num_heads)
        # self.linear_q = nn.Linear(model_dim+44, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        # self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention  # attention: torch.Size([64, 21, 21])


def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def padding_mask_sand(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=1,
                 model_dim=768,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        self.pre_embedding = nn.Linear(vocab_size, model_dim)
        self.weight_layer = torch.nn.Linear(model_dim, 1)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.time_layer = torch.nn.Linear(64, 768)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, diagnosis_codes, mask, seq_time_step, input_len):
        diagnosis_codes = diagnosis_codes.permute(1, 0, 2)
        seq_time_step = torch.Tensor(seq_time_step).cuda().unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        mask = mask.permute(1, 0, 2)
        output = self.pre_embedding(diagnosis_codes)
        output += time_feature
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        attentions = []
        outputs = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
            outputs.append(output)
        weight = torch.softmax(self.weight_layer(outputs[-1]), dim=1)
        weight = weight * mask - 255 * (1 - mask)
        output = outputs[-1].permute(1, 0, 2)
        weight = weight.permute(1, 0, 2)
        return output, weight


from math import sqrt

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n

        att = torch.bmm(dist, v)
        return att




class EncoderNew(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=2,
                 model_dim=768,
                 gram_dim=768,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0,
                 num_attention_heads=1,
                 dropout_rate=0.2,
                 attention_dim=128):
        super(EncoderNew, self).__init__()
        self.use_gram_emb = True
        self.use_hita = True

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             # [EncoderLayer(model_dim+44, num_heads, ffn_dim, dropout) for _ in # 44
             range(num_layers)])
        # self.icd9_embedding = ICD9Embedding()
        # icd9_codes = [key for key, value in np.load('./data/icd2idx.npy', allow_pickle=True).item().items()]
        # self.icd9_embedding.build_pos1_mapping(icd9_codes)

        self.pre_embedding = Embedding(vocab_size, model_dim) 
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))

        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        # self.pos_embedding = PositionalEncoding(model_dim+44, max_seq_len) # 44
        # self.pos_embedding = PositionalEncoding(model_dim+44, max_seq_len) # 44
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)  # 44
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)  # 44
        self.time_layer = torch.nn.Linear(64, 768) 
        self.time_emb1 = torch.nn.Linear(1, model_dim, bias=True) 
        self.time_emb2 = torch.nn.Linear(model_dim, model_dim) 
        self.time_emb = MLP(input_size=1, hidden_size=model_dim, output_size=model_dim, num_layers=2,
                            dropout=dropout_rate) 

        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.filter = torch.nn.Linear(model_dim + gram_dim, model_dim)
        self.filter2 = torch.nn.Linear(model_dim, model_dim)
        self.self_layer = torch.nn.Linear(768, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.WQ1 = nn.Linear(model_dim, attention_dim, bias=False)  # attentio  dim = 128
        self.WK1 = nn.Linear(model_dim, attention_dim, bias=False)
        self.attention_dim = attention_dim
        self.softmax = nn.Softmax()
        self.softmax2 = nn.Softmax()
        self.decay = nn.Parameter(torch.FloatTensor([-0.1] * (vocab_size + 1)))
        self.initial = nn.Parameter(torch.FloatTensor([1.0] * (vocab_size + 1)))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        # self.mlp = MLP(input_size=1, hidden_size=200, output_size=39, num_layers=2, dropout=0.2)
        self.FT1 = nn.Linear(1, model_dim, bias=True)
        self.FT2 = nn.Linear(model_dim, 39, bias=True)
        self.tanh2 = nn.Tanh()
        # self.FT3 = nn.Linear(model_dim,39, bias=True)
        self.patient_time = True
        self.visit_time = True
        self.main_code_embedding = nn.Embedding(940, 744)
        # self.sub_codes1_embedding = nn.Embedding(12, 100)
        # self.sub_codes2_embedding = nn.Embedding(12, 100)
        self.concate_weight = nn.Linear(768, 1)
        self.weights = nn.Parameter(torch.Tensor([1.0, 1.0, 1.0]))
        self.gender_embedding = nn.Embedding(2, 768)
        self.age_embedding = nn.Embedding(101, 23)
        self.enthic_embedding = nn.Embedding(41, 768)
        # self.bert_pretrained = BertModel.from_pretrained('./data/my_bert_model')

    def get_self_attention(self, features, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill_(mask.bool(), -np.inf), dim=1)
        # attention = torch.softmax(self.self_layer(features).masked_fill(mask, -np.inf), dim=1)
        return attention

    def forward(self, ages, genders, ethnics,
                diagnosis_codes, mask, mask_code, seq_time_step, input_len, seq_time_step2):
        numbatch, numvisit, numcode = diagnosis_codes.shape

        # simPatents = torch.stack(simPatents).to(diagnosis_codes.device)
        # simPatents_emb = self.pre_embedding(simPatents.view(-1, numcode))
        # simPatents_emb = simPatents_emb.view(numbatch, 10, numvisit, numcode, -1) # 相似患者的表征
        # simPatents_emb = simPatents_emb.sum(dim=1)

        mask_code = mask_code.squeeze()
        mask_code = 1e+20 - mask_code * 1e+20

        v = self.pre_embedding(diagnosis_codes.view(-1, numcode))  
        v = v.view(-1, numcode, 768)

        v = self.dropout(v)
        myQ1 = self.WQ1(v)  # torch.Size([128, 39, 80])
        myK1 = self.WK1(v)
        dproduct1 = torch.bmm(myQ1, torch.transpose(myK1, 1, 2)).view(numbatch, numvisit, numcode,
                                                                      numcode) 
        dproduct1 = dproduct1 - mask_code.view(numbatch, numvisit, 1, numcode) - mask_code.view(numbatch, numvisit, numcode, 1)
        sproduct1 = self.softmax(dproduct1.view(-1, numcode) / np.sqrt(self.attention_dim)).view(-1, numcode, numcode)  # torch.Size([366, 39, 39])
        fembedding11 = torch.bmm(sproduct1, v)  # torch.Size([336, 39, 768])
        fembedding11 = (((mask_code - (1e+20)) / (-1e+20)).view(-1, numcode, 1) * fembedding11)

        deltaT = seq_time_step2.view(numbatch, numvisit, 1)  #
        # T_emb = self.FT2(self.FT1(deltaT)) # torch.Size([16, 21, 39])
        T_emb = self.FT2(self.sigmoid(self.FT1(deltaT)))  # torch.Size([16, 21, 39])
        if self.visit_time:
            # TGn = torch.bmm(T_emb.view(-1,1,39), fembedding11)
            TGn = torch.bmm(T_emb.view(-1, 1, numcode), fembedding11)
        else:
            TGn = fembedding11.sum(dim=2) + self.bias_embedding
        vv = TGn

        visit_emb = vv.view(numbatch, numvisit, -1)

        time_feature = self.time_emb2(self.softmax2(self.time_emb1(seq_time_step)))
        age_emb = ages / 100  # torch.Size([16, 1])
        gender_emb = torch.nn.functional.one_hot(genders, num_classes=2).squeeze()  # torch.Size([16, 2])
        ethnics_emb = torch.nn.functional.one_hot(ethnics, num_classes=41).squeeze()  # torch.Size([16, 41])
        if self.patient_time:
            output = visit_emb.to(diagnosis_codes.device) + time_feature
        else:
            output = visit_emb.to(diagnosis_codes.device)

        # static_feature = torch.cat([age_emb, gender_emb, ethnics_emb], dim=-1)  # torch.Size([16, 44])
        # static_feature = static_feature.unsqueeze(1).expand(-1, 21, -1)
        # output = torch.cat([static_feature, output], dim=-1)  # torch.Size([16, 22, 44+768])

        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos
        self_attention_mask = padding_mask(ind_pos, ind_pos)
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
        visit_att = torch.bmm(T_emb.view(-1, 1, 39), sproduct1).squeeze().view(-1, numvisit, numcode)
        return output, visit_att


class EncoderNew2(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=2,
                 model_dim=768,
                 gram_dim=768,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0,
                 num_attention_heads=1,
                 dropout_rate=0.2,
                 attention_dim=128):
        super(EncoderNew, self).__init__()
        self.use_gram_emb = True
        self.use_hita = True

        self.encoder_layers = nn.ModuleList(
            # [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        # self.icd9_embedding = ICD9Embedding()
        # icd9_codes = [key for key, value in np.load('./data/icd2idx.npy', allow_pickle=True).item().items()]
        # self.icd9_embedding.build_pos1_mapping(icd9_codes)

        self.pre_embedding = Embedding(vocab_size, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))

        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.time_layer = torch.nn.Linear(64, 768)  
        self.time_emb1 = torch.nn.Linear(1, model_dim, bias=True) 
        self.time_emb2 = torch.nn.Linear(model_dim, model_dim) 
        self.time_emb = MLP(input_size=1, hidden_size=model_dim, output_size=model_dim, num_layers=2,
                            dropout=dropout_rate) 

        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.filter = torch.nn.Linear(model_dim + gram_dim, model_dim)
        self.filter2 = torch.nn.Linear(model_dim, model_dim)
        self.self_layer = torch.nn.Linear(768, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.WQ1 = nn.Linear(model_dim, attention_dim, bias=False)  # attentio  dim = 128
        self.WK1 = nn.Linear(model_dim, attention_dim, bias=False)
        self.attention_dim = attention_dim
        self.softmax = nn.Softmax()
        self.softmax2 = nn.Softmax()
        self.decay = nn.Parameter(torch.FloatTensor([-0.1] * (vocab_size + 1)))
        self.initial = nn.Parameter(torch.FloatTensor([1.0] * (vocab_size + 1)))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        # self.mlp = MLP(input_size=1, hidden_size=200, output_size=39, num_layers=2, dropout=0.2)
        self.FT1 = nn.Linear(1, model_dim, bias=True)
        self.FT2 = nn.Linear(model_dim, 39, bias=True)
        self.tanh2 = nn.Tanh()
        # self.FT3 = nn.Linear(model_dim,39, bias=True)
        self.patient_time = True
        self.visit_time = True
        self.main_code_embedding = nn.Embedding(940, 744)
        # self.sub_codes1_embedding = nn.Embedding(12, 100)
        # self.sub_codes2_embedding = nn.Embedding(12, 100)
        self.concate_weight = nn.Linear(768, 1)
        self.weights = nn.Parameter(torch.Tensor([1.0, 1.0, 1.0]))
        self.gender_embedding = nn.Embedding(2, 768)
        self.age_embedding = nn.Embedding(101, 23)
        self.enthic_embedding = nn.Embedding(41, 768)
        # self.bert_pretrained = BertModel.from_pretrained('./data/my_bert_model')

    def get_self_attention(self, features, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill_(mask.bool(), -np.inf), dim=1)
        # attention = torch.softmax(self.self_layer(features).masked_fill(mask, -np.inf), dim=1)
        return attention

    def forward(self, ages, genders, ethnics,
                diagnosis_codes, mask, mask_code, seq_time_step, input_len, seq_time_step2):
        numbatch, numvisit, numcode = diagnosis_codes.shape

        # simPatents = torch.stack(simPatents).to(diagnosis_codes.device)
        # simPatents_emb = self.pre_embedding(simPatents.view(-1, numcode))
        # simPatents_emb = simPatents_emb.view(numbatch, 10, numvisit, numcode, -1) 
        # simPatents_emb = simPatents_emb.sum(dim=1)

        mask_code = mask_code.squeeze()
        mask_code = 1e+20 - mask_code * 1e+20

        # v_bert = self.bert_pretrained(diagnosis_codes.view(-1, numcode))
        # v_1 = weighted_emb.view(-1, numcode, 768)
        v = self.pre_embedding(diagnosis_codes.view(-1, numcode)) 
        # v = v + v_1
        # v = v_1
        # v = v.view(-1, numvisit * numcode, 768) + weights_normalized[0]*age_emb + \
        #     weights_normalized[1]*gender_emb + weights_normalized[2]*ethnics_emb
        # v = torch.cat([v.view(-1, numvisit*numcode, 768),age_emb, gender_emb, ethnics_emb], dim=-1)
        v = v.view(-1, numcode, 768)
        # simPatents_emb = simPatents_emb.view(-1, numcode, 768)
        # v = v + simPatents_emb

        v = self.dropout(v)
        myQ1 = self.WQ1(v)  # torch.Size([128, 39, 80])
        myK1 = self.WK1(v)
        dproduct1 = torch.bmm(myQ1, torch.transpose(myK1, 1, 2)).view(numbatch, numvisit, numcode,
                                                                      numcode)  
        dproduct1 = dproduct1 - mask_code.view(numbatch, numvisit, 1, numcode) - mask_code.view(numbatch, numvisit,
                                                                                                numcode, 1)
        sproduct1 = self.softmax(dproduct1.view(-1, numcode) / np.sqrt(self.attention_dim)).view(-1, numcode,
                                                                                                 numcode)  # torch.Size([366, 39, 39])
        fembedding11 = torch.bmm(sproduct1, v)  # torch.Size([336, 39, 768])
        fembedding11 = (((mask_code - (1e+20)) / (-1e+20)).view(-1, numcode, 1) * fembedding11)

        deltaT = seq_time_step2.view(numbatch, numvisit, 1)  #
        # T_emb = self.FT2(self.FT1(deltaT)) # torch.Size([16, 21, 39])
        T_emb = self.FT2(self.sigmoid(self.FT1(deltaT)))  # torch.Size([16, 21, 39])
        if self.visit_time:
            # TGn = torch.bmm(T_emb.view(-1,1,39), fembedding11)
            TGn = torch.bmm(T_emb.view(-1, 1, numcode), fembedding11)
        else:
            TGn = fembedding11.sum(dim=2) + self.bias_embedding
        vv = TGn

        visit_emb = vv.view(numbatch, numvisit, -1)

        time_feature = self.time_emb2(self.softmax2(self.time_emb1(seq_time_step)))
        age_emb = ages / 100  # torch.Size([16, 1])
        gender_emb = torch.nn.functional.one_hot(genders, num_classes=2).squeeze()  # torch.Size([16, 2])
        ethnics_emb = torch.nn.functional.one_hot(ethnics, num_classes=41).squeeze()  # torch.Size([16, 41])
        if self.patient_time:
            output = visit_emb.to(diagnosis_codes.device) + time_feature
        else:
            output = visit_emb.to(diagnosis_codes.device)
        # time_feature = self.tanh( self.time_emb( seq_time_step ) )
        # output = visit_emb.to(diagnosis_codes.device) + time_feature

        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos
        static_feature = torch.cat([age_emb, gender_emb, ethnics_emb], dim=-1)  # torch.Size([16, 44])
        static_feature = static_feature.unsqueeze(1).expand(-1, 21, -1)
        output = torch.cat([static_feature, output], dim=-1)  # torch.Size([16, 22, 44+768])
        self_attention_mask = padding_mask(ind_pos, ind_pos)
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
        visit_att = torch.bmm(T_emb.view(-1, 1, 39), sproduct1).squeeze().view(-1, numvisit, numcode)
        return output, visit_att


class EncoderPure(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=1,
                 model_dim=768,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0):
        super(EncoderPure, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        self.pre_embedding = Embedding(vocab_size, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, diagnosis_codes, mask, mask_code, seq_time_step, input_len):
        output = (self.pre_embedding(diagnosis_codes) * mask_code).sum(dim=2) + self.bias_embedding
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        attentions = []
        outputs = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
            outputs.append(output)

        return output


def adjust_input(batch_diagnosis_codes, batch_time_step, max_len, n_diagnosis_codes):
    batch_time_step = copy.deepcopy(batch_time_step)
    batch_diagnosis_codes = copy.deepcopy(batch_diagnosis_codes)
    for ind in range(len(batch_diagnosis_codes)):
        if len(batch_diagnosis_codes[ind]) > max_len:
            batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-(max_len):]
            batch_time_step[ind] = batch_time_step[ind][-(max_len):]
        batch_time_step[ind].append(0)
        batch_diagnosis_codes[ind].append([n_diagnosis_codes - 1])
    return batch_diagnosis_codes, batch_time_step


class TimeEncoder(nn.Module):
    def __init__(self, batch_size):
        super(TimeEncoder, self).__init__()
        self.batch_size = batch_size
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight_layer = torch.nn.Linear(64, 64)

    def forward(self, seq_time_step, final_queries, mask):
        selection_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))  # eq(6)--o
        selection_feature = self.relu(self.weight_layer(selection_feature))  # eq(6) -- k
        selection_feature = torch.sum(selection_feature * final_queries, 2, keepdim=True) / 8  # eq (7)
        selection_feature = selection_feature.masked_fill_(mask.bool(), -np.inf)
        return torch.softmax(selection_feature, 1)  # eq(8)


class TransformerTime(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, num_layers, dropout_rate):
        super(TransformerTime, self).__init__()
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = EncoderNew(n_diagnosis_codes + 1, 21, num_layers=num_layers, dropout_rate=dropout_rate)
        self.sim_feature_encoder = EncoderNew(n_diagnosis_codes + 1, 21, num_layers=num_layers,
                                              dropout_rate=dropout_rate)
        self.self_layer = torch.nn.Linear((768), 1)  # 44
        self.self_layer_sim = torch.nn.Linear((768), 1)  # 44
        self.quiry_layer = torch.nn.Linear((768), 64)  # 44
        self.quiry_layer_sim = torch.nn.Linear((768), 64)  # 44
        self.quiry_weight_layer = torch.nn.Linear((768), 2)  # 44
        self.relu = nn.ReLU(inplace=True)
        dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.weight_sum = nn.Parameter(torch.Tensor([0.7]))  # 定义可训练权重

        # self.lin = torch.nn.Linear(768 * 3+44, 768)
        self.lin = torch.nn.Linear(768 * 2, 768)

        self.graph_encoder = GraphEncoder(128, 768, 3)
        # self.graph_encoder = GraphEncoder(768, 768, 768, 3, 3)
        # self.graph_encoder = RelationalGraphConvolution(768, 768, 3, 3)
        self.sequence_encoder = SequenceEncoder(input_size=768, hidden_size=768, num_layers=2)
        self.drop_seq_graph = nn.Dropout(0.2) # 0.2

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill_(mask.bool(), -np.inf), dim=1)
        return attention

    def get_self_attention_sim(self, features, query, mask):
        attention = torch.softmax(self.self_layer_sim(features).masked_fill_(mask.bool(), -np.inf), dim=1)
        return attention

    def find_max_attention_edge(self, attention_weights, edge_index):
        # print('attention_weights.size(): ', attention_weights.size())

        all_attention_weights = torch.cat([weights[1].flatten() for weights in attention_weights if
                                           isinstance(weights, tuple) and weights[1] is not None])

        max_weight_index = all_attention_weights.argmax().item()

        max_attention_edge = edge_index[:, max_weight_index]

        return max_attention_edge

    def find_next_max_attention_edge(self, edge, attention_weights, edge_index):

        adjacent_edges = torch.nonzero((edge_index[0, :] == edge[0]) | (edge_index[1, :] == edge[1]),
                                       as_tuple=False).flatten()

        adjacent_weights = attention_weights[adjacent_edges]

        max_weight_index = adjacent_weights.argmax().item()

        max_attention_edge = edge_index[:, adjacent_edges[max_weight_index]]

        return max_attention_edge

        graph_embeddings = []
        for graph in patient_data:
            graph_embedding, attention_weights = graph_encoder(graph.x, graph.edge_index, graph.edge_type, graph.node_type)
            if graph_embedding.size(1) == 1:
                graph_embedding = graph_embedding.squeeze(1)
            graph_embeddings.append(graph_embedding)
        # try:
        patient_embedding = sequence_encoder(torch.stack(graph_embeddings).unsqueeze(0))
        # except:
        #     print('graph_embeddings: ', graph_embeddings)
        #     print('patient_data: ', patient_data)
        return patient_embedding

    def forward(self, simPatents, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, seq_time_step2,
                return_hita_attention=True, use_graph=False):
        bsz = diagnosis_codes.size(0)
        if return_hita_attention:
            num_simP = 7
            features, visit_att = self.feature_encoder(ages, genders, ethnics,
                                                       diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths,
                                                       seq_time_step2)  # torch.Size([16, 21, 768]), torch.Size([6000, 1, 22, 1])
            sim_feature, _ = self.sim_feature_encoder(simPatents[6].view(-1, 1), simPatents[7].view(-1, 1),
                                                      simPatents[8].view(-1, 1),
                                                      simPatents[0].view(-1, 21, 39), simPatents[1].view(-1, 21, 1),
                                                      simPatents[2].view(-1, 21, 39, 1),
                                                      simPatents[3].view(-1, 21, 1), simPatents[4].view(-1),
                                                      simPatents[5].view(-1, 21))

        else:
            features = self.sim_feature_encoder(simPatents, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                                                diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths,
                                                seq_time_step2)  # torch.Size([16, 21, 768]), torch.Size([6000, 1, 22, 1])
            # sim_feature--(160, 21, 812)
            sim_feature = self.sim_feature_encoder(simPatents[6].view(-1, 1), simPatents[7].view(-1, 1),
                                                   simPatents[8].view(-1, 1),
                                                   simPatents[0].view(-1, 21, 39), simPatents[1].view(-1, 21, 1),
                                                   simPatents[2].view(-1, 21, 39, 1),
                                                   simPatents[3].view(-1, 21, 1), simPatents[4].view(-1),
                                                   simPatents[5].view(-1, 21))
        final_statues = features * mask_final
        final_sim_statues = sim_feature * simPatents[9].view(-1, 21, 1)

        final_statues = final_statues.sum(1, keepdim=True)
        final_sim_statues = final_sim_statues.sum(1, keepdim=True)

        quiryes = self.relu(self.quiry_layer(final_statues))  # eq (5)
        sim_quiryes = self.relu(self.quiry_layer_sim(final_sim_statues))  # eq (5)

        self_weight = self.get_self_attention(features, quiryes, mask_mult)  # eq (4) alpha
        self_weight_sim = self.get_self_attention_sim(sim_feature, sim_quiryes,
                                                      simPatents[1].view(-1, 21, 1))  # eq (4) alpha

        weighted_features = features * self_weight  # torch.Size([16, 21, 768]) torch.Size([16, 21, 1])
        weighted_features_sim = sim_feature * self_weight_sim  # torch.Size([16, 21, 768]) torch.Size([16, 21, 1])

        averaged_features = torch.sum(weighted_features, 1)
        averaged_features_sim = torch.sum(weighted_features_sim, 1)

        averaged_features = self.dropout(averaged_features)  # 16, 812
        averaged_features_sim = self.dropout(averaged_features_sim)  # 160, 812
        averaged_features_sim = averaged_features_sim.view(bsz, num_simP, -1)  # 16, 10, 812

        att_weights = torch.bmm(averaged_features_sim, averaged_features.unsqueeze(2))
        att_weights = att_weights.squeeze(2)  # (batch_size, n_heads)
        att_weights = F.softmax(att_weights, dim=1)
        att_weights = att_weights.unsqueeze(2)  # (16, 10, 1)

        final_averaged_features = att_weights * averaged_features_sim
        final_averaged_features = torch.sum(final_averaged_features, dim=1)  # 16, 812
        # final_averaged_features = self.dropout(final_averaged_features)  # 160, 812

        # final_sum_features = torch.cat([averaged_features, final_averaged_features], dim=-1)
        # final_sum_features = self.dropout(final_sum_features)  # 16, 1624

        # final_sum_features = averaged_features + final_averaged_features # 16, 812

        age_emb = ages / 100  # torch.Size([16, 1])
        gender_emb = torch.nn.functional.one_hot(genders, num_classes=2).squeeze()  # torch.Size([16, 2])
        ethnics_emb = torch.nn.functional.one_hot(ethnics, num_classes=41).squeeze()  # torch.Size([16, 41])

        # sum 7
        final_sum_features = averaged_features # torch.Size([16, 768])

        # final_sum_features = torch.cat([averaged_features, age_emb, gender_emb, ethnics_emb], dim=-1) # torch.Size([16, 768+44])

        # sum 7 + age
        final_sum_features = torch.cat([final_sum_features, age_emb, gender_emb, ethnics_emb], dim=-1) # torch.Size([16, 768+44])
        final_sum_features = self.dropout(final_sum_features)  # 16, 1624


        # final_sum_features = self.weight_sum * averaged_features + (1 - self.weight_sum) * final_averaged_features
        # final_sum_features = torch.cat([final_sum_features, age_emb, gender_emb, ethnics_emb], dim=-1)  # torch.Size([16, 768+44])

        # # graph_seq
        if use_graph:
            graph_seqs = simPatents[-1] # torch.Size([16, graphs_size])
            graph_embs = []
            for graph_data in graph_seqs:
                graph_emb = self.batch_to_embedding(graph_data, self.graph_encoder, self.sequence_encoder)  # 16*1
                graph_embs.append(graph_emb.squeeze())
            graph_seq_emb = torch.stack(graph_embs, dim=0)  # torch.Size([16, 768])
            graph_seq_emb = self.drop_seq_graph(graph_seq_emb)  # torch.Size([16, 768])
            # final_sum_features = torch.cat([final_sum_features, graph_seq_emb], dim=-1)  # torch.Size([16, 768+44])
            return final_sum_features, visit_att, self_weight, graph_seq_emb

        else:
            # final_sum_features = torch.cat([final_sum_features, graph_seq_emb], dim=-1)  # torch.Size([16, 768+44])
            # final_sum_features = self.dropout(final_sum_features)  # 16, 1624
            if return_hita_attention:
                return final_sum_features, visit_att, self_weight
            else:
                return final_sum_features


class TransformerTime_copy(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, num_layers, dropout_rate):
        super(TransformerTime, self).__init__()
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = EncoderNew(n_diagnosis_codes + 1, 21, num_layers=num_layers)
        self.self_layer = torch.nn.Linear(768, 1)
        self.classify_layer = torch.nn.Linear(768, 2)
        self.quiry_layer = torch.nn.Linear(768, 64)
        self.quiry_weight_layer = torch.nn.Linear(768, 2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill_(mask.bool(), -np.inf), dim=1)
        return attention

    def forward(self, diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths):
        features = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step,lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues)) 

        self_weight = self.get_self_attention(features, quiryes, mask_mult) 

        time_weight = self.time_encoder(seq_time_step, quiryes, mask_mult) 
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)  

        total_weight = torch.cat((time_weight, self_weight), 2) 
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True) 
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)  
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        return averaged_features


class TransformerTimeAtt(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerTimeAtt, self).__init__()
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = EncoderPure(options['n_diagnosis_codes'] + 1, 21, num_layers=options['layer'])
        self.self_layer = torch.nn.Linear(768, 1)
        self.classify_layer = torch.nn.Linear(768, 2)
        self.quiry_layer = torch.nn.Linear(768, 64)
        self.quiry_weight_layer = torch.nn.Linear(768, 2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill_(mask.bool(), -np.inf), dim=1)
        return attention

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(units.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final, mask_code = units.pad_matrix_new(seq_dignosis_codes,
                                                                                    batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.LongTensor(diagnosis_codes).cuda()
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
            mask_code = torch.Tensor(mask_code).unsqueeze(3).cuda()
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code).unsqueeze(3)
        features = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))

        self_weight = self.get_self_attention(features, quiryes, mask_mult)
        time_weight = self.time_encoder(seq_time_step, quiryes, options, mask_mult)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight


class TransformerTimeEmb(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerTimeEmb, self).__init__()
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = EncoderNew(options['n_diagnosis_codes'] + 1, 21, num_layers=options['layer'])
        self.self_layer = torch.nn.Linear(768, 1)
        self.classify_layer = torch.nn.Linear(768, 2)
        self.quiry_layer = torch.nn.Linear(768, 64)
        self.quiry_weight_layer = torch.nn.Linear(768, 2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill_(mask.bool(), -np.inf), dim=1)
        return attention

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(units.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final, mask_code = units.pad_matrix_new(seq_dignosis_codes,
                                                                                    batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.LongTensor(diagnosis_codes).cuda()
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
            mask_code = torch.Tensor(mask_code).unsqueeze(3).cuda()
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code).unsqueeze(3)
        features = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))

        self_weight = self.get_self_attention(features, quiryes, mask_mult)
        total_weight = self_weight
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight


class TransformerSelf(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerSelf, self).__init__()
        self.feature_encoder = EncoderPure(options['n_diagnosis_codes'] + 1, 21, num_layers=options['layer'])
        self.self_layer = torch.nn.Linear(768, 1)
        self.classify_layer = torch.nn.Linear(768, 2)
        self.quiry_layer = torch.nn.Linear(768, 64)
        self.quiry_weight_layer = torch.nn.Linear(768, 2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill_(mask.bool(), -np.inf), dim=1)
        return attention

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(units.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final, mask_code = units.pad_matrix_new(seq_dignosis_codes,
                                                                                    batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.LongTensor(diagnosis_codes).cuda()
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
            mask_code = torch.Tensor(mask_code).unsqueeze(3).cuda()
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code).unsqueeze(3)
        features = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))

        self_weight = self.get_self_attention(features, quiryes, mask_mult)
        total_weight = self_weight
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight


class TransformerFinal(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerFinal, self).__init__()
        self.feature_encoder = EncoderPure(options['n_diagnosis_codes'] + 1, 21, num_layers=options['layer'])
        self.self_layer = torch.nn.Linear(768, 1)
        self.classify_layer = torch.nn.Linear(768, 2)
        self.quiry_layer = torch.nn.Linear(768, 64)
        self.quiry_weight_layer = torch.nn.Linear(768, 2)
        self.relu = nn.ReLU(inplace=True)
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(units.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final, mask_code = units.pad_matrix_new(seq_dignosis_codes,
                                                                                    batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.LongTensor(diagnosis_codes).cuda()
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
            mask_code = torch.Tensor(mask_code).unsqueeze(3).cuda()
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code).unsqueeze(3)
        features = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1)

        predictions = self.classify_layer(final_statues)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, None

import torch

from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_hita import *
from utils.layers import *
from modeling.transformer_hita import TransformerTime


class QAGNN_Message_Passing(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, input_size, hidden_size, output_size,
                 dropout=0.1):
        super().__init__()
        assert input_size == output_size
        self.args = args
        self.n_ntype = n_ntype
        self.n_etype = n_etype

        assert input_size == hidden_size
        self.hidden_size = hidden_size

        self.emb_node_type = nn.Linear(self.n_ntype, hidden_size // 2)

        self.basis_f = 'sin'  # ['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, hidden_size // 2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, hidden_size // 2)
            self.emb_score = nn.Linear(hidden_size // 2, hidden_size // 2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(hidden_size // 2, hidden_size // 2)

        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size))

        self.k = k
        self.gnn_layers = nn.ModuleList(
            [GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])

        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

    def mp_helper(self, _X, edge_index, edge_type, _node_type, _node_feature_extra, return_attention_weights=True):
        all_gnn_attn = []
        all_edge_map = []
        for _ in range(self.k):
            if return_attention_weights:
                _X, (edge_idx, edge_weight) = self.gnn_layers[_](_X, edge_index, edge_type, _node_type,
                                                                 _node_feature_extra)
                gnn_attn = edge_weight[:, - 1]
                edge_map = edge_idx

                gnn_attn = gnn_attn[0:500]
                edge_map = edge_map[:, 0:500]

                all_gnn_attn.append(gnn_attn)
                all_edge_map.append(edge_map)
            else:
                _X = self.gnn_layers[_](_X, edge_index, edge_type, _node_type, _node_feature_extra)
            _X = self.activation(_X)
            _X = F.dropout(_X, self.dropout_rate, training=self.training)
        if return_attention_weights:
            return _X, (all_edge_map, all_gnn_attn)
        else:
            return _X

    def forward(self, H, A, node_type, node_score, cache_output=False, return_attention_weights=True):
        """
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        _batch_size, _n_nodes = node_type.size()

        # Embed type
        T = make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T)) 

        # Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size // 2).unsqueeze(0).unsqueeze(0).float().to(
                node_type.device)  # [1,1,dim/2]
            js = torch.pow(1.1, js)  # [1,1,dim/2]
            B = torch.sin(js * node_score) 
            node_score_emb = self.activation(self.emb_score(B)) 
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) 
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) 
            node_score_emb = self.activation(self.emb_score(B)) 

        X = H
        edge_index, edge_type = A 
        _X = X.view(-1, X.size(2)).contiguous()
        _node_type = node_type.view(-1).contiguous() 
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0),
                                                                                     -1).contiguous()  # [`total_n_nodes`, dim]

        if return_attention_weights:
            _X, (all_gnn_atten, all_edge_map) = self.mp_helper(_X, edge_index, edge_type, _node_type,
                                                               _node_feature_extra)
        else:
            _X = self.mp_helper(_X, edge_index, edge_type, _node_type, _node_feature_extra)

        X = _X.view(node_type.size(0), node_type.size(1), -1)

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        if return_attention_weights:
            return output, (all_gnn_atten, all_edge_map)
        else:
            return output


class QAGNN(nn.Module):
    def __init__(self, args, pre_dim, k, n_ntype, n_etype, sent_dim,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0, gram_dim=768):
        super().__init__()
        self.pre_dim = pre_dim
        self.init_range = init_range
        # token Embedding
        self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                               use_contextualized=False, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb,
                                               freeze_ent_emb=freeze_ent_emb)
        self.svec2nvec = nn.Linear((sent_dim+44 ), concept_dim)
        # self.svec2nvec = nn.Linear((sent_dim)*2, concept_dim)
        # self.svec2nvec = nn.Linear((sent_dim) * 2 + 44, concept_dim)
        # self.svec2nvec = nn.Linear((sent_dim) * 3 + 44, concept_dim)
        # self.svec2nvec = nn.Linear((sent_dim)*02+44+100, concept_dim)
        # self.svec2nvec = nn.Linear((sent_dim)+44, concept_dim)

        self.concept_dim = concept_dim

        self.activation = GELU()
        self.gnn = QAGNN_Message_Passing(args, k=k, n_ntype=n_ntype, n_etype=n_etype,
                                         input_size=concept_dim, hidden_size=concept_dim, output_size=concept_dim,
                                         dropout=p_gnn)

        # self.pooler = MultiheadAttPoolLayer(n_attention_head, (sent_dim ), concept_dim) 
        # self.pooler = MultiheadAttPoolLayer(n_attention_head, (sent_dim*3+44 ), concept_dim) 
        # self.pooler = MultiheadAttPoolLayer(n_attention_head, (sent_dim)*2+44, concept_dim) 
        # self.pooler = MultiheadAttPoolLayer(n_attention_head, (sent_dim)*2, concept_dim) 
        # self.pooler = MultiheadAttPoolLayer(n_attention_head, (sent_dim)+44, concept_dim) 
        self.pooler = MultiheadAttPoolLayer(n_attention_head, (sent_dim)+44, concept_dim) 

        # self.fc = MLP( (sent_dim ), fc_dim, self.pre_dim, n_fc_layer, p_fc,layer_norm=True)
        # self.fc = MLP(sent_dim*2, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True) 
        # self.fc = MLP(sent_dim * 2 + 44, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)  
        # self.fc = MLP(sent_dim * 3 + 44, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)  
        # self.fc = MLP((sent_dim+44)*2, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True) 
        self.is_pretrain = False
        if self.is_pretrain:
            self.fc = MLP((sent_dim + 44), fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)
        else:
            self.use_graph = True
            if self.use_graph:
                self.fc = MLP((sent_dim+44)*2+768, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)
                # self.fc = MLP((sent_dim + 44) + 768, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)
            else:
                self.fc = MLP((sent_dim + 44) * 2, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)
        # self.fc2 = MLP((sent_dim + 44)*2 + 768, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)  
        # self.fc2 = MLP((sent_dim + 44) + 768, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)  
        self.fc2 = MLP((sent_dim + 44)*2 + 768+200, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)  
        # self.fc2 = MLP((sent_dim + 44)*2, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)  
        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)
        self.dropout_fc2 = nn.Dropout(p_fc+0.2)
        self.dropout_g = nn.Dropout(0.9)  # mimic iii
        self.dropout_z = nn.Dropout(0.9)

        self.activateOut = torch.nn.Sigmoid()

        if init_range > 0:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, sent_vecs, dl_vec, concept_ids, node_type_ids, node_scores, adj_lengths, adj, emb_data=None,
                cache_output=False, return_attention_weights=True, return_P_emb=False,simp_emb=None, graph_seq_emb=None, isPretrain=False):
        gnn_input0 = self.activation(self.svec2nvec(sent_vecs)).unsqueeze(1)  
        gnn_input1 = self.concept_emb(concept_ids[:, 1:] - 1, emb_data)  
        gnn_input1 = gnn_input1.to(node_type_ids.device)
        gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1))  

        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(
            1)).float() 
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :]  
        node_scores = node_scores.squeeze(2)
        node_scores = node_scores * _mask
        mean_norm = (torch.abs(node_scores)).sum(dim=1) / adj_lengths 
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05)  
        node_scores = node_scores.unsqueeze(2)  

        if return_attention_weights:
            gnn_output, (edge_idx, edge_weight) = self.gnn(gnn_input, adj, node_type_ids, node_scores)
        else:
            gnn_output = self.gnn(gnn_input, adj, node_type_ids, node_scores)

        Z_vecs = gnn_output[:, 0]

        mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(
            1)  

        mask = mask | (node_type_ids == 3) 
        mask[mask.all(1), 0] = 0 

        sent_vecs_for_pooler = sent_vecs
        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)

        if cache_output:
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        # concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs, Z_vecs), 1)) 
        concat = self.dropout_fc(sent_vecs) 

        if isPretrain:
            logits = self.fc(concat)  
            logits = self.activateOut(logits)  
            if return_attention_weights:
                return logits, pool_attn, (edge_idx, edge_weight)
            else:
                return logits, pool_attn


        if return_P_emb:
            return concat, pool_attn
        else:
            # 1. 
            if graph_seq_emb is not None:
                concat = self.dropout_fc(torch.cat((sent_vecs, dl_vec, graph_seq_emb), 1)) 
            else:
                concat = self.dropout_fc(torch.cat((sent_vecs, dl_vec), 1)) 

            # 2. - dl_vec
            # if graph_seq_emb is not None:
            #     concat = self.dropout_fc(
            #         torch.cat((sent_vecs, graph_seq_emb), 1)) 
            # else:
            #     concat = self.dropout_fc(sent_vecs) 

            # 3. qagnn_graph
            # if graph_seq_emb is not None:
            #     ct = torch.cat((dl_vec, graph_seq_emb, graph_vecs, Z_vecs), -1)
            #     ct = self.dropout_fc(ct)
            #     concat = torch.cat((sent_vecs, ct), -1)
            #     concat = self.dropout_fc(concat)
            #
            #
            #
            #     # sent_vec_dim = 812 
            #     # other_vec_dims = [812, 768, 100, 100]  
            #     # model = AttentionConcat(sent_vec_dim, other_vec_dims).to(sent_vecs.device)
            #     # result = model(sent_vecs, [dl_vec, graph_seq_emb, graph_vecs, Z_vecs])
            #     # # print('result: ',result.shape)
            #     # concat = self.dropout_fc(result) 
            #     # exit()
            #
            #     # concat = self.dropout_fc(torch.cat((sent_vecs, dl_vec, graph_seq_emb, graph_vecs, Z_vecs), 1)) 
            #     # print('sent_vecs: ',sent_vecs.shape)
            #     # print('dl_vec: ',dl_vec.shape)
            #     # print('graph_seq_emb: ',graph_seq_emb.shape)
            #     # print('graph_vecs: ',graph_vecs.shape)
            #     # print('Z_vecs: ',Z_vecs.shape)
            #     # exit()
            # else:
            #     concat = self.dropout_fc(torch.cat((sent_vecs, dl_vec), 1)) 

            logits = self.fc(concat)  # bs*961
            logits = self.activateOut(logits)  # bs*5985
            if return_attention_weights:
                return logits, pool_attn, (edge_idx, edge_weight)
            else:
                return logits, pool_attn

class FeatureFusion(nn.Module):
    def __init__(self, input_dims, output_dim, num_heads):
        super(FeatureFusion, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(0.7)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_dim, output_dim) for in_dim in input_dims
        ])

        self.multihead_attn = nn.MultiheadAttention(output_dim, num_heads)

    def forward(self, *tensors):
        assert len(tensors) == len(self.input_dims), "Number of input tensors must match number of input dimensions."

        transformed_tensors = [linear(tensor).unsqueeze(0) for tensor, linear in zip(tensors, self.linear_layers)]
        combined_tensor = torch.cat(transformed_tensors, dim=0)

        attn_output, _ = self.multihead_attn(combined_tensor, combined_tensor, combined_tensor)
        flattened_output = attn_output.transpose(0, 1).reshape(attn_output.size(1), -1)
        return flattened_output


class AttentionConcat(nn.Module):
    def __init__(self, sent_vec_dim, other_vec_dims):
        super(AttentionConcat, self).__init__()
        self.align_dim = sent_vec_dim
        self.dropout = nn.Dropout(0.7)

        self.align_layers = nn.ModuleDict()
        for i, dim in enumerate(other_vec_dims):
            if dim != sent_vec_dim:
                self.align_layers[str(i)] = nn.Linear(dim, sent_vec_dim)

    def attention(self, query, key):
        scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        return F.softmax(scores, dim=-1)

    def forward(self, sent_vecs, other_vecs):
        final_out_feature = 0

        for i, vec in enumerate(other_vecs):
            if str(i) in self.align_layers:
                vec = self.align_layers[str(i)](vec)

            final_out_feature += self.attention(sent_vecs, vec) @ vec

        final_out_feature = self.dropout(final_out_feature)
        return torch.cat([final_out_feature, sent_vecs], dim=-1)


class LM_QAGNN(nn.Module):
    def __init__(self, args, pre_dim, model_name, k, n_ntype, n_etype,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.0, encoder_config={}, hita_config={}):
        super().__init__()
        self.encoder_PreTrain = TextEncoder(model_name, **encoder_config)
        self.encoder_HITA = TransformerTime(**hita_config)

        self.decoder = QAGNN(args, pre_dim, k, n_ntype, n_etype,
                             self.encoder_PreTrain.sent_dim,
                             n_concept, concept_dim, concept_in_dim, n_attention_head,
                             fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                             pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                             init_range=init_range)
        self.use_gram_emb = True


        self.fc = MLP(768 + 44, fc_dim, 5985, n_fc_layer, p_fc,
                      layer_norm=True)  

        self.activateOut = torch.nn.Sigmoid()

        self.dropout_att = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.2)

        self.w1 = torch.nn.Parameter(torch.randn(768+44, 768+44))
        self.cross_attention = CrossAttention(768+44, 768+44)

    def get_pretrain_emb(self, simPatients, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                return_attention_weights=True, return_hita_attention=True, return_P_emb=False,
                         simp_emb=None, use_graph=False, isPretrain=False):
        bs, nc = inputs[0].size(0), inputs[0].size(1)

        edge_index_orig, edge_type_orig = inputs[-2:]  
        _inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]] + [
            x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]] + [sum(x, []) for x in inputs[-2:]]

        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs  
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device),
               edge_type.to(node_type_ids.device))  

        if use_graph:
            if return_hita_attention:
                vecs_hita, visit_att, self_att, graph_seq_emb = self.encoder_HITA(simPatients,
                                                                   main_codes, sub_codes1, sub_codes2, ages, genders,
                                                                   ethnics,
                                                                   diagnosis_codes, seq_time_step,
                                                                   mask_mult, mask_final, mask_code,
                                                                   lengths, seq_time_step2,
                                                                   return_hita_attention, use_graph=use_graph)
            else:
                vecs_hita = self.encoder_HITA(simPatients,
                                              main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                                              diagnosis_codes, seq_time_step,
                                              mask_mult, mask_final, mask_code,
                                              lengths, seq_time_step2, return_hita_attention, use_graph=use_graph)
        else:
            if return_hita_attention:
                vecs_hita, visit_att, self_att = self.encoder_HITA(simPatients,
                                                                   main_codes, sub_codes1, sub_codes2, ages, genders,
                                                                   ethnics,
                                                                   diagnosis_codes, seq_time_step,
                                                                   mask_mult, mask_final, mask_code,
                                                                   lengths, seq_time_step2,
                                                                   return_hita_attention, use_graph=use_graph)
            else:
                vecs_hita = self.encoder_HITA(simPatients,
                                              main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                                              diagnosis_codes, seq_time_step,
                                              mask_mult, mask_final, mask_code,
                                              lengths, seq_time_step2, return_hita_attention, use_graph=use_graph)

        sent_vec = vecs_hita
        if isPretrain:
            return self.decoder(sent_vec.to(node_type_ids.device),
                                                                 simp_emb,
                                                                 concept_ids,
                                                                 node_type_ids, node_scores, adj_lengths, adj,
                                                                 emb_data=None, cache_output=cache_output,
                                                                 return_P_emb=False, simp_emb=None,
                                                                 graph_seq_emb=None, isPretrain=True)

        if return_P_emb:
            concat_emb,_ = self.decoder(sent_vec.to(node_type_ids.device),
                                        vecs_hita.to(node_type_ids.device),
                                        concept_ids,
                                        node_type_ids, node_scores, adj_lengths, adj,
                                        emb_data=None, cache_output=cache_output, return_P_emb=return_P_emb)
            return concat_emb
        else:
            # print('simp_emb: ',simp_emb.shape)
            simp_emb = self.sum_sim_patients(simp_emb, sent_vec)
            if use_graph:
                graph_seq_emb = graph_seq_emb



            # simp_emb = simp_emb.mean(dim=1)
            # print('simp_emb: ', simp_emb.shape)
                if return_attention_weights:
                    logits, attn, (edge_idx, edge_weight) = self.decoder(sent_vec.to(node_type_ids.device),
                                                                         simp_emb.to(node_type_ids.device),
                                                                         concept_ids,
                                                                         node_type_ids, node_scores, adj_lengths, adj,
                                                                         emb_data=None, cache_output=cache_output, return_P_emb=return_P_emb,simp_emb=simp_emb, graph_seq_emb=graph_seq_emb)
                else:
                    logits, attn = self.decoder(sent_vec.to(node_type_ids.device),
                                                vecs_hita.to(node_type_ids.device),
                                                concept_ids,
                                                node_type_ids, node_scores, adj_lengths, adj,
                                                emb_data=None, cache_output=cache_output, return_P_emb=return_P_emb,simp_emb=simp_emb)
            else:
                if return_attention_weights:
                    logits, attn, (edge_idx, edge_weight) = self.decoder(sent_vec.to(node_type_ids.device),
                                                                         simp_emb.to(node_type_ids.device),
                                                                         concept_ids,
                                                                         node_type_ids, node_scores, adj_lengths, adj,
                                                                         emb_data=None, cache_output=cache_output, return_P_emb=return_P_emb,simp_emb=simp_emb)
                else:
                    logits, attn = self.decoder(sent_vec.to(node_type_ids.device),
                                                vecs_hita.to(node_type_ids.device),
                                                concept_ids,
                                                node_type_ids, node_scores, adj_lengths, adj,
                                                emb_data=None, cache_output=cache_output, return_P_emb=return_P_emb,simp_emb=simp_emb)

        # logits = logits.view(bs, nc)
        # for i in range(16):
        #     batch_i = edge_index_orig[i]
        #     for j in range(batch_i.size(1)):
        #         a = batch_i[:j]
        #         b = edge_index[:j]
        # print(a ==b )

        if not detail:
            if return_attention_weights:
                return logits, attn, (edge_idx, edge_weight), visit_att, self_att
            else:
                return logits, attn
        else:
            if return_attention_weights:
                return logits, attn, concept_ids.view(bs, nc, -1), \
                    node_type_ids.view(bs, nc, -1), edge_index_orig, \
                    edge_type_orig, (edge_idx, edge_weight)
            else:
                return logits, attn, concept_ids.view(bs, nc, -1), \
                    node_type_ids.view(bs, nc, -1), edge_index_orig, \
                    edge_type_orig

    def forward(self, simPatients, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                return_attention_weights=True, return_hita_attention=True, return_P_emb=False,
                return_emb=True,simp_emb=None, use_graph=False, isPretrain=False):
        if isPretrain:
            return self.get_pretrain_emb(simPatients, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                                         diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                                         seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                                         return_attention_weights=True, return_hita_attention=True,
                                         return_P_emb=False,
                                         simp_emb=None, use_graph=False, isPretrain=True)

        if return_emb:
            return self.get_pretrain_emb(simPatients, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                    diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                    seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                    return_attention_weights=True, return_hita_attention=True, return_P_emb=return_P_emb,
                                         simp_emb=None, use_graph=use_graph, isPretrain=isPretrain)
        else:
            return_P_emb = False
            if use_graph:
                use_graph = True
            else:
                use_graph = False
            simp_emb = simp_emb
            return self.get_pretrain_emb(simPatients, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                    diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                    seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                    return_attention_weights=True, return_hita_attention=True, return_P_emb=return_P_emb,
                                         simp_emb=simp_emb, use_graph=use_graph, isPretrain=isPretrain)

    def multi_head_attention_forward(self, query, key, value, num_heads, dropout):
        batch_size = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]
        head_dim = query.shape[2] // num_heads
        assert head_dim * num_heads == query.shape[2], "Embedding size must be divisible by num_heads"

        query = query.reshape(batch_size, query_len, num_heads, head_dim)
        key = key.reshape(batch_size, key_len, num_heads, head_dim)
        value = value.reshape(batch_size, value_len, num_heads, head_dim)

        scales = query @ key.transpose(-2, -1) / (head_dim ** 0.5)
        attention = torch.softmax(scales, dim=-1)
        attention = F.dropout(attention, p=dropout, training=True)
        out = attention @ value

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * head_dim)

        return out

    def sum_sim_patients2(self, sim_P, P_emb, num_heads=4, dropout=0.2):
        # sim_P: [bsz, seq_len, embed_size], P_emb: [bsz, embed_size]
        P_emb_unsqueezed = P_emb.unsqueeze(1)

        attention_output = self.multi_head_attention_forward(
            P_emb_unsqueezed, sim_P, sim_P, num_heads, dropout
        )

        print('attention_output: ',attention_output.shape)
        attention_output = attention_output.sum(dim=1)

        # sum_P = attention_output.squeeze(1)
        return attention_output

    def sum_sim_patients(self, sim_P, P_emb):
        attention_weight = self.cross_attention(P_emb, sim_P) # [bsz, 1, 7]
        # print('attention_weight: ',attention_weight.shape)
        # print('sim_P: ',sim_P.shape)
        attended_sim_P = attention_weight.squeeze().unsqueeze(2) * sim_P
        attended_sim_P = attended_sim_P.sum(dim=1)
        # attended_sim_P = self.dropout_att(attended_sim_P)

        # print('attended_sim_P: ',attended_sim_P.shape)
        # print('P_emb: ',P_emb.shape)


        # P_emb_transformed = torch.matmul(P_emb, self.w1)  # size([bsz, 768]
        #
        # attention_scores = torch.bmm(sim_P, P_emb_transformed.unsqueeze(2))  # size([bsz, 10, 1])
        #
        # attention_weights = F.softmax(attention_scores, dim=1)  # size([bsz, 10, 1])
        #
        # weighted_sum = torch.sum(sim_P * attention_weights, dim=1)  # size([bsz, 768])
        #
        # return weighted_sum
















        #
        # 假设这些张量已经被定义和初始化
        # bsz = sim_P.size(0)
        # # sim_P 的尺寸: [bsz, 10, 768]
        # # P_emb 的尺寸: [bsz, 768]
        #
        # P_emb_expanded = P_emb.unsqueeze(1).expand(-1, sim_P.size(1), -1)
        # # 计算注意力分数
        # # u_states * w1，其中u_states是sim_P，w1是权重矩阵
        # # 注意torch.bmm期望批次在第一维，因此我们无需调整w1的维度，只需扩展它
        # scores = torch.bmm(torch.tanh(torch.bmm(sim_P, self.w1.unsqueeze(0).expand(bsz, -1, -1))),
        #                    P_emb_expanded.transpose(1, 2))
        #
        # # 计算softmax，这里的dim=1是指第二维，我们通过转置操作后得到正确的维度
        # attention_weights = F.softmax(scores.transpose(1, 2), dim=1)
        # sumP = torch.matmul(attention_weights, sim_P).squeeze()
        # print('sumP: ',sumP.shape)
        # sumP = sumP.sum(dim=1)
        # sumP = self.dropout_att(sumP)


        # # # sim_P: [bsz, 10, 768], P_emb: [bsz, 768]
        # # 将sim_P中的每个元素与P_emb计算注意力，然后加权求和
        # key = torch.tanh(torch.bmm(sim_P, self.w1.unsqueeze(0).expand(bsz, -1, -1)))  # [bsz, 10, 768]
        # query = P_emb.unsqueeze(2)  # [bsz, 768, 1]
        # attention = torch.bmm(key, query)  # [bsz, 10, 1]
        # # attention = F.dropout(attention, p=0.7, training=True)
        # attention = F.softmax(attention, dim=1)  # [bsz, 10, 1]
        # weighted_P = attention * sim_P  # [bsz, 10, 768]
        # # sum_P = torch.sum(weighted_P, dim=1)  # [bsz, 768]
        # sum_P = torch.sum(weighted_P, dim=1)  # [bsz, 768]
        #
        # sim_P = self.dropout_att(sim_P)
        #
        return attended_sim_P


    # def classify_icd(self, P_emb):
    #     logits = self.fc(P_emb)
    #     logits = self.activateOut(logits)
    #     return logits



    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        # edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        # edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1)  # [2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0)  # [total_E, ]
        return edge_index, edge_type

class CrossAttention(nn.Module):
    """交叉注意力模块"""
    def __init__(self, patient_rep_dim=768+44, similar_patients_rep_dim=768+44):
        super(CrossAttention, self).__init__()
        self.key_fc = nn.Linear(similar_patients_rep_dim, patient_rep_dim)
        self.query_fc = nn.Linear(patient_rep_dim, patient_rep_dim)
        self.scale = 1.0 / math.sqrt(patient_rep_dim)
        self.tanh = nn.Tanh()

    def forward(self, query, keys):
        # query: 当前患者表征, 形状 (batch_size, patient_rep_dim)
        # keys: 相似患者表征, 形状 (batch_size, num_similar_patients, similar_patients_rep_dim)
        query = self.query_fc(query).unsqueeze(1) # 转换查询，并增加一个维度以便广播
        # keys = self.key_fc(keys) # 转换键值
        energy = torch.bmm(query, keys.transpose(1, 2)) * self.scale # 计算能量值
        # energy = torch.bmm(query, keys.transpose(1, 2))
        attention_weights = F.softmax(energy, dim=2) # 应用softmax得到权重
        return attention_weights

class LM_QAGNN_DataLoader(object):

    def __init__(self, args, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=20,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1.0, use_cache=True):
        super().__init__()  # 调用超类初始化
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device  # device=(device0, device1)
        self.is_inhouse = is_inhouse  # 是否选用内定数据，若有的话

        # 此name为编码器，encoder='cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
        # 对应的model_type为‘bert’
        model_type = MODEL_NAME_TO_CLASS[model_name]  # 类型与可调度模型名字的映射
        (self.train_qids, self.train_HF_labels, self.train_Diag_labels, \
            self.train_main_codes, self.train_sub_code1s, self.train_sub_code2s, \
            self.train_ages, self.train_genders, self.train_ethnicities, \
            self.train_diagnosis_codes, self.train_seq_time_step, self.train_mask_mult, \
            self.train_mask_final, self.train_mask_code, self.train_lengths, self.train_seq_time_step2,
         self.train_main_diagnose_list, *self.train_encoder_data) = \
            load_input_tensors(train_statement_path, model_type, model_name, max_seq_length)

        (self.dev_qids, self.dev_HF_labels, self.dev_Diag_labels, \
            self.dev_main_codes, self.dev_sub_code1s, self.dev_sub_code2s, \
            self.dev_ages, self.dev_genders, self.dev_ethnicities, \
            self.dev_diagnosis_codes, self.dev_seq_time_step, self.dev_mask_mult, \
            self.dev_mask_final, self.dev_mask_code, self.dev_lengths, self.dev_seq_time_step2,
         self.main_diagnose_list, *self.dev_encoder_data) = \
            load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length)

        # 选项数目
        num_choice = self.train_encoder_data[0].size(1)
        self.num_choice = num_choice
        print('num_choice:', num_choice)
        *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(train_adj_path,
                                                                                              max_node_num, num_choice,
                                                                                              args)
        *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num,
                                                                                          num_choice, args)

        if test_statement_path is not None:
            (self.test_qids, self.test_HF_labels, self.test_Diag_labels, \
                self.test_main_codes, self.test_sub_code1s, self.test_sub_code2s, \
                self.test_ages, self.test_genders, self.test_ethnicities, \
                self.test_diagnosis_codes, self.test_seq_time_step, self.test_mask_mult, \
                self.test_mask_final, self.test_mask_code, self.test_lengths, self.test_seq_time_step2,
             self.test_main_diagnose_list, *self.test_encoder_data) = \
                load_input_tensors(test_statement_path, model_type, model_name, max_seq_length)
            *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj_path,
                                                                                                max_node_num,
                                                                                                num_choice, args)
            # assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

        # self.is_inhouse = 0 # 不要，不会改。修掉修掉！
        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor(
                [i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor(
                [i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            self.train_qids = self.train_qids[:n_train]
            self.train_HF_labels = self.train_HF_labels[:n_train]
            self.train_Diag_labels = self.train_Diag_labels[:n_train]
            self.train_diagnosis_codes = self.train_diagnosis_codes[:n_train]
            self.train_seq_time_step = self.train_seq_time_step[:n_train]
            self.train_seq_time_step2 = self.train_seq_time_step2[:n_train]
            self.train_main_diagnose_list = self.train_main_diagnose_list[:n_train]
            self.train_mask_mult = self.train_mask_mult[:n_train]
            self.train_mask_final = self.train_mask_final[:n_train]
            self.train_mask_code = self.train_mask_code[:n_train]
            self.train_lengths = self.train_lengths[:n_train]
            self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
            self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
            self.train_adj_data = self.train_adj_data[:n_train]
            # assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    # 训练集大小（假如有内定，则以内定大小为主；否则，返回外部导入的数据长度
    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    # 验证集大小（未设定，自认不合理
    def dev_size(self):
        return len(self.dev_qids)

    # 测试集大小
    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0
    def train_all_data(self):
        return

    def dev_all_data(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'dev',
                                                   self.device0,
                                                   self.device1,
                                                   self.batch_size,
                                                   dev_indexes,
                                                   self.dev_qids,
                                                   self.dev_HF_labels,
                                                   self.dev_Diag_labels,
                                                   self.dev_main_codes,
                                                   self.dev_sub_code1s,
                                                   self.dev_sub_code2s,
                                                   self.dev_ages,
                                                   self.dev_genders,
                                                   self.dev_ethnicities,
                                                   self.dev_diagnosis_codes,
                                                   self.dev_seq_time_step,
                                                   self.dev_mask_mult,
                                                   self.dev_mask_final,
                                                   self.dev_mask_code,
                                                   self.dev_lengths,
                                                   self.dev_seq_time_step2,
                                                   self.dev_main_diagnose_list,
                                                   tensors0=self.dev_encoder_data,
                                                   tensors1=self.dev_decoder_data,
                                                   adj_data=self.dev_adj_data)

    def test_all_data(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'test',
                                                   self.device0,
                                                   self.device1,
                                                   self.batch_size,
                                                   test_indexes,
                                                   self.test_qids,
                                                   self.test_HF_labels,
                                                   self.test_Diag_labels,
                                                   self.test_main_codes,
                                                   self.test_sub_code1s,
                                                   self.test_sub_code2s,
                                                   self.test_ages,
                                                   self.test_genders,
                                                   self.test_ethnicities,
                                                   self.test_diagnosis_codes,
                                                   self.test_seq_time_step,
                                                   self.test_mask_mult,
                                                   self.test_mask_final,
                                                   self.test_mask_code,
                                                   self.test_lengths,
                                                   self.test_seq_time_step2,
                                                   self.test_main_diagnose_list,
                                                   tensors0=self.test_encoder_data,
                                                   tensors1=self.test_decoder_data,
                                                   adj_data=self.test_adj_data)

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'train',
                                                   self.device0,
                                                   self.device1,
                                                   self.batch_size,
                                                   train_indexes,
                                                   self.train_qids,
                                                   self.train_HF_labels,
                                                   self.train_Diag_labels,
                                                   self.train_main_codes,
                                                   self.train_sub_code1s,
                                                   self.train_sub_code2s,
                                                   self.train_ages,
                                                   self.train_genders,
                                                   self.train_ethnicities,
                                                   self.train_diagnosis_codes,
                                                   self.train_seq_time_step,
                                                   self.train_mask_mult,
                                                   self.train_mask_final,
                                                   self.train_mask_code,
                                                   self.train_lengths,
                                                   self.train_seq_time_step2,
                                                   self.train_main_diagnose_list,
                                                   tensors0=self.train_encoder_data,
                                                   tensors1=self.train_decoder_data,
                                                   adj_data=self.train_adj_data)

    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval',
                                                   self.device0, self.device1,
                                                   self.eval_batch_size,
                                                   torch.arange(len(self.dev_qids)),
                                                   self.dev_qids,
                                                   self.dev_HF_labels,
                                                   self.dev_Diag_labels,
                                                   self.dev_main_codes,
                                                   self.dev_sub_code1s,
                                                   self.dev_sub_code2s,
                                                   self.dev_ages,
                                                   self.dev_genders,
                                                   self.dev_ethnicities,
                                                   self.dev_diagnosis_codes,
                                                   self.dev_seq_time_step,
                                                   self.dev_mask_mult,
                                                   self.dev_mask_final,
                                                   self.dev_mask_code,
                                                   self.dev_lengths,
                                                   self.dev_seq_time_step2,
                                                   self.dev_main_diagnose_list,
                                                   tensors0=self.dev_encoder_data,
                                                   tensors1=self.dev_decoder_data,
                                                   adj_data=self.dev_adj_data)

    def test(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1,
                                                   self.eval_batch_size,
                                                   torch.arange(len(self.test_qids)),
                                                   self.test_qids,
                                                   self.test_HF_labels,
                                                   self.test_Diag_labels,
                                                   self.test_main_codes,
                                                   self.test_sub_code1s,
                                                   self.test_sub_code2s,
                                                   self.test_ages,
                                                   self.test_genders,
                                                   self.test_ethnicities,
                                                   self.test_diagnosis_codes,
                                                   self.test_seq_time_step,
                                                   self.test_mask_mult,
                                                   self.test_mask_final,
                                                   self.test_mask_code,
                                                   self.test_lengths,
                                                   self.test_seq_time_step2,
                                                    self.test_main_diagnose_list,
                                                   tensors0=self.test_encoder_data,
                                                   tensors1=self.test_decoder_data,
                                                   adj_data=self.test_adj_data)


###############################################################################
############################### GNN architecture ##############################
###############################################################################

from torch.autograd import Variable


def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter


# 在GAT中，沿着边的信息传递
class GATConvE(MessagePassing):
    """
    Args:
        emb_dim (int): dimensionality of GNN hidden states
        n_ntype (int): number of node types (e.g. 4)
        n_etype (int): number of edge relation types (e.g. 38)
    """

    def __init__(self, args, emb_dim, n_ntype, n_etype, edge_encoder, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        self.args = args

        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype;
        self.n_etype = n_etype
        self.edge_encoder = edge_encoder

        # For attention（注意力部分
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(3 * emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3 * emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(2 * emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        # For final MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))

    def forward(self, x, edge_index, edge_type, node_type, node_feature_extra, return_attention_weights=True):
        # x: [N, emb_dim]
        # edge_index: [2, E]
        # edge_type [E,] -> edge_attr: [E, 39] / self_edge7+attr: [N, 39]
        # node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        # node_feature_extra [N, dim]

        # Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype + 1)  # [E, 39]
        self_edge_vec = torch.zeros(x.size(0), self.n_etype + 1).to(edge_vec.device)
        self_edge_vec[:, self.n_etype] = 1

        head_type = node_type[edge_index[0]]  # [E,] #head=src
        tail_type = node_type[edge_index[1]]  # [E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype)  # [E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype)  # [E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1)  # [E,8]
        self_head_vec = make_one_hot(node_type, self.n_ntype)  # [N,4]
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1)  # [N,8]

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0)  # [E+N, ?]
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0)  # [E+N, ?]
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1))  # [E+N, emb_dim]
        # edge_vec:torch.Size([18926, 35]) headtail_vec: torch.Size([22126, 8])
        # Add self loops to edge_index
        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)  # [2, E+N]

        x = torch.cat([x, node_feature_extra], dim=1)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)  # [N, emb_dim]
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, edge_index, x_i, x_j, edge_attr):  # i: tgt, j:src
        # print ("edge_attr.size()", edge_attr.size()) #[E, emb_dim]
        # print ("x_j.size()", x_j.size()) #[E, emb_dim]
        # print ("x_i.size()", x_i.size()) #[E, emb_dim]
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2 * self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count,
                                                                       self.dim_per_head)  # [E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count,
                                                                       self.dim_per_head)  # [E, heads, _dim]
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]

        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2)  # [E, heads]
        src_node_index = edge_index[0]  # [E,]
        alpha = softmax(scores, src_node_index)  # [E, heads] #group by src side node
        self._alpha = alpha

        # adjust by outgoing degree of src
        E = edge_index.size(1)  # n_edges
        N = int(src_node_index.max()) + 1  # n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index]  # [E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1)  # [E, heads]

        out = msg * alpha.view(-1, self.head_count, 1)  # [E, heads, _dim]
        return out.view(-1, self.head_count * self.dim_per_head)  # [E, emb_dim]

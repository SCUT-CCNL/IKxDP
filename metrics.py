import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from modeling.transformer import TransformerTime
from modeling.units import adjust_input_hita
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def f1(y_true_hot, y_pred, metrics='weighted'):
    result = np.zeros_like(y_true_hot)
    for i in range(len(result)):
        true_number = np.sum(y_true_hot[i] == 1)
        result[i][y_pred[i][:true_number]] = 1
    return f1_score(y_true=y_true_hot, y_pred=result, average=metrics, zero_division=0)


def top_k_prec_recall(y_true_hot, y_pred, ks):
    a = np.zeros((len(ks),))
    r = np.zeros((len(ks),))
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        # if len(t)==0:
        #     len_t = len(t)+100000000
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            # r[i] += len(it) / min(k, len_t)
            try:
                r[i] += len(it) / len(t)
            except:
                r[i] += len(it) / 10000000
    return a / len(y_true_hot), r / len(y_true_hot)


def calculate_occurred(historical, y, preds, ks):
    # y_occurred = np.sum(np.logical_and(historical, y), axis=-1)
    # y_prec = np.mean(y_occurred / np.sum(y, axis=-1))
    r1 = np.zeros((len(ks),))
    r2 = np.zeros((len(ks),))
    n = np.sum(y, axis=-1)
    for i, k in enumerate(ks):
        # n_k = np.minimum(n, k)
        n_k = n
        pred_k = np.zeros_like(y)
        for T in range(len(pred_k)):
            pred_k[T][preds[T][:k]] = 1
        # pred_occurred = np.sum(np.logical_and(historical, pred_k), axis=-1)
        pred_occurred = np.logical_and(historical, pred_k)
        pred_not_occurred = np.logical_and(np.logical_not(historical), pred_k)
        pred_occurred_true = np.logical_and(pred_occurred, y)
        pred_not_occurred_true = np.logical_and(pred_not_occurred, y)
        r1[i] = np.mean(np.sum(pred_occurred_true, axis=-1) / n_k)
        r2[i] = np.mean(np.sum(pred_not_occurred_true, axis=-1) / n_k)
    return r1, r2


def evaluate_codes2(model, dataset, loss_fn, output_size, historical=None):
    model.eval()
    total_loss = 0.0
    labels = dataset.label()
    preds = []
    for step in range(len(dataset)):
        code_x, visit_lens, divided, y, neighbors = dataset[step]
        output = model(code_x, divided, neighbors, visit_lens)
        pred = torch.argsort(output, dim=-1, descending=True)
        preds.append(pred)
        loss = loss_fn(output, y)
        total_loss += loss.item() * output_size * len(code_x)
        print('\r    Evaluating step %d / %d' % (step + 1, len(dataset)), end='')
    avg_loss = total_loss / dataset.size()
    preds = torch.vstack(preds).detach().cpu().numpy()
    f1_score = f1(labels, preds)
    prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])
    if historical is not None:
        r1, r2 = calculate_occurred(historical, labels, preds, ks=[10, 20, 30, 40])
        print(
            '\r    Evaluation: loss: %.4f --- f1_score: %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f  --- occurred: %.4f, %.4f, %.4f, %.4f  --- not occurred: %.4f, %.4f, %.4f, %.4f'
            % (avg_loss, f1_score, recall[0], recall[1], recall[2], recall[3], r1[0], r1[1], r1[2], r1[3], r2[0], r2[1],
               r2[2], r2[3]))
    else:
        print('\r    Evaluation: loss: %.4f --- f1_score: %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f'
              % (avg_loss, f1_score, recall[0], recall[1], recall[2], recall[3]))
    return avg_loss, f1_score


def evaluate_hf2(model, dataset, loss_fn, output_size=1, historical=None):
    model.eval()
    total_loss = 0.0
    labels = dataset.label()
    outputs = []
    preds = []
    for step in range(len(dataset)):
        code_x, visit_lens, divided, y, neighbors = dataset[step]
        output = model(code_x, divided, neighbors, visit_lens).squeeze()
        loss = loss_fn(output, y)
        total_loss += loss.item() * output_size * len(code_x)
        output = output.detach().cpu().numpy()
        outputs.append(output)
        pred = (output > 0.5).astype(int)
        preds.append(pred)
        print('\r    Evaluating step %d / %d' % (step + 1, len(dataset)), end='')
    avg_loss = total_loss / dataset.size()
    outputs = np.concatenate(outputs)
    preds = np.concatenate(preds)
    auc = roc_auc_score(labels, outputs)
    f1_score_ = f1_score(labels, preds)
    print('\r    Evaluation: loss: %.4f --- auc: %.4f --- f1_score: %.4f' % (avg_loss, auc, f1_score_))
    return avg_loss, f1_score_


def evaluate_hf(dataset, model, name='dev'):
    model.eval()
    # labels = dataset.HF_labels
    outputs = []
    preds = []
    labels = []
    loss_fn = torch.nn.BCELoss()
    all_embeddings = []

    with torch.no_grad():
        for qids, main_diagnoses, simPatients, Hf_labels, Diag_labels, main_codes, sub_codes1, sub_codes2, *input_data in tqdm(
                dataset, desc=name):
            demo_emb = model(simPatients, main_codes, sub_codes1, sub_codes2, *input_data, return_P_emb=True,
                             return_emb=True)  
            all_embeddings.append(demo_emb.cpu().detach().numpy())
        all_embeddings = np.vstack(all_embeddings)
        print('all_embeddings:', all_embeddings.shape)

        for qids, main_diagnoses, simPatients, Hf_labels, Diag_labels, main_codes, sub_codes1, sub_codes2, *input_data in tqdm(
                dataset, desc=name):
            Hf_label = Hf_labels.float()
            labels.append(Hf_label.cpu().numpy())
            P_emb = model(simPatients, main_codes, sub_codes1, sub_codes2, *input_data, return_P_emb=True,
                          return_emb=True)  
            similarity_matrix = cosine_similarity(P_emb.cpu().detach().numpy(),
                                                  all_embeddings)  # shape: (batch_size, total_patients)

            topk_indices = np.argsort(similarity_matrix, axis=1)[:, -7:]  # shape: (batch_size, 10)

            topk_embeddings = torch.tensor(all_embeddings[topk_indices]).to(
                Diag_labels.device)  # shape: (batch_size, 10, embed_dim)

            logits, _, _, _, _ = model(simPatients, main_codes, sub_codes1, sub_codes2, *input_data,
                                       return_P_emb=False,
                                       return_emb=False, simp_emb=topk_embeddings, use_graph=True)

            # loss = loss_fn(logits.squeeze(), Hf_label)
            output = logits.detach().cpu().numpy()
            pred = (output > 0.5).astype(int)
            preds.append(pred)
            outputs.append(output)
        outputs = np.concatenate(outputs)
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        auc = roc_auc_score(labels, outputs)
        f1_score_ = f1_score(labels, preds)
        return f1_score_, auc


def write_file(file_name, input_text):
    with open(file_name, "a") as file:
        file.write(input_text)


def evaluate_codes(eval_set, model, name='eval'):
    model.eval()
    total_loss = 0.0
    preds = []
    loss_fn = torch.nn.BCELoss()
    labels = []
    all_embeddings = []
    with torch.no_grad():
        for qids, main_diagnoses, simPatients, Hf_labels, Diag_labels, main_codes, sub_codes1, sub_codes2, *input_data in tqdm(
                eval_set, desc=name):
            Diag_labels = Diag_labels.float()
            labels.append(Diag_labels)

            logits, _, _ = model(simPatients, main_codes, sub_codes1, sub_codes2, *input_data, return_P_emb=False,
                                 return_emb=False, simp_emb=None, use_graph=True, isPretrain=False)

            pred = torch.argsort(logits, dim=-1, descending=True)
            # for b in range( pred.size(0) ):
            #     write_file('explain.txt', str(qids[b]) )
            #     write_file('explain.txt', '\t'.join([str(x_i) for x_i in pred[b].cpu().numpy()])+'\n')
            #     write_file('explain.txt', '\t'.join([str(x_i) for x_i in Diag_labels[b].cpu().numpy()])+'\n')
            #     write_file('explain.txt', '\t'.join(paths[b])+'\n')
            #     write_file('explain.txt', '\t'.join([str(x_i) for x_i in attention_score.detach().cpu().numpy().tolist()])+'\n' )
            #
            #     write_file('pred_codes.txt', str(qids[b])+'\t'.join(pred[b].cpu().numpy())+'\n')
            #     write_file('path.txt', '\t'.join(paths[b])+'\n' )
            #     write_file('path_attention.txt', '\t'.join([str(x_i) for x_i in  attention_score.detach().cpu().numpy().tolist()])+'\n' )

            preds.append(pred)
            # loss = loss_fn(logits, Diag_labels)
            # total_loss += loss.item()

        # print('evaluate loss: \t', total_loss)
        preds = torch.vstack(preds).detach().cpu().numpy()
        labels = torch.vstack(labels).detach().cpu().numpy()
        f1_score = f1(labels, preds)
        prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])
        # print(name+' f1_score: %.8f --- top_k_recall: %.4f, %.4f, %.4f, %.4f'
        #           % (f1_score, recall[0], recall[1], recall[2], recall[3]))
        return f1_score, recall


def evaluate_codes_pretrain(eval_set, model, name='eval'):
    model.eval()
    total_loss = 0.0
    preds = []
    loss_fn = torch.nn.BCELoss()
    labels = []
    all_embeddings = []
    with torch.no_grad():
        for qids, main_diagnoses, simPatients, Hf_labels, Diag_labels, main_codes, sub_codes1, sub_codes2, *input_data in tqdm(
                eval_set, desc=name):
            main_diagnoses = main_diagnoses.float()
            labels.append(main_diagnoses)

            logits, _, _ = model(simPatients, main_codes, sub_codes1, sub_codes2, *input_data, return_P_emb=False,
                                 return_emb=False, simp_emb=None, use_graph=True, isPretrain=True)

            pred = torch.argsort(logits, dim=-1, descending=True)
            # for b in range( pred.size(0) ):
            #     write_file('explain.txt', str(qids[b]) )
            #     write_file('explain.txt', '\t'.join([str(x_i) for x_i in pred[b].cpu().numpy()])+'\n')
            #     write_file('explain.txt', '\t'.join([str(x_i) for x_i in Diag_labels[b].cpu().numpy()])+'\n')
            #     write_file('explain.txt', '\t'.join(paths[b])+'\n')
            #     write_file('explain.txt', '\t'.join([str(x_i) for x_i in attention_score.detach().cpu().numpy().tolist()])+'\n' )
            #
            #     write_file('pred_codes.txt', str(qids[b])+'\t'.join(pred[b].cpu().numpy())+'\n')
            #     write_file('path.txt', '\t'.join(paths[b])+'\n' )
            #     write_file('path_attention.txt', '\t'.join([str(x_i) for x_i in  attention_score.detach().cpu().numpy().tolist()])+'\n' )

            preds.append(pred)
            # loss = loss_fn(logits, Diag_labels)
            # total_loss += loss.item()

        # print('evaluate loss: \t', total_loss)
        preds = torch.vstack(preds).detach().cpu().numpy()
        labels = torch.vstack(labels).detach().cpu().numpy()
        f1_score = f1(labels, preds)
        prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])
        # print(name+' f1_score: %.8f --- top_k_recall: %.4f, %.4f, %.4f, %.4f'
        #           % (f1_score, recall[0], recall[1], recall[2], recall[3]))
        return f1_score, recall


def evaluate_codes0(eval_set, model, name='eval'):
    model.eval()
    total_loss = 0.0
    preds = []
    loss_fn = torch.nn.BCELoss()
    labels = []
    all_embeddings = []
    with torch.no_grad():
        for qids, main_diagnoses, simPatients, Hf_labels, Diag_labels, main_codes, sub_codes1, sub_codes2, *input_data in tqdm(
                eval_set, desc=name):
            demo_emb = model(simPatients, main_codes, sub_codes1, sub_codes2, *input_data, return_P_emb=True,
                             return_emb=True)  
            all_embeddings.append(demo_emb.cpu().detach().numpy())
        all_embeddings = np.vstack(all_embeddings)
        print('all_embeddings:', all_embeddings.shape)

        for qids, main_diagnoses, simPatients, Hf_labels, Diag_labels, main_codes, sub_codes1, sub_codes2, *input_data in tqdm(
                eval_set, desc=name):
            Diag_labels = Diag_labels.float()
            labels.append(Diag_labels)
            # print(simPatients)
            P_emb = model(simPatients, main_codes, sub_codes1, sub_codes2, *input_data, return_P_emb=True,
                          return_emb=True) 
            similarity_matrix = cosine_similarity(P_emb.cpu().detach().numpy(),
                                                  all_embeddings)  # shape: (batch_size, total_patients)

            topk_indices = np.argsort(similarity_matrix, axis=1)[:, -7:]  # shape: (batch_size, 10)

            topk_embeddings = torch.tensor(all_embeddings[topk_indices]).to(
                Diag_labels.device)  # shape: (batch_size, 10, embed_dim)

            # final_embedding = model.sum_sim_patients(topk_embeddings, P_emb)  # shape: (batch_size, embed_dim)

            logits, _, _, _, _ = model(simPatients, main_codes, sub_codes1, sub_codes2, *input_data, return_P_emb=False,
                                       return_emb=False, simp_emb=topk_embeddings, use_graph=True)

            pred = torch.argsort(logits, dim=-1, descending=True)
            # for b in range( pred.size(0) ):
            #     write_file('explain.txt', str(qids[b]) )
            #     write_file('explain.txt', '\t'.join([str(x_i) for x_i in pred[b].cpu().numpy()])+'\n')
            #     write_file('explain.txt', '\t'.join([str(x_i) for x_i in Diag_labels[b].cpu().numpy()])+'\n')
            #     write_file('explain.txt', '\t'.join(paths[b])+'\n')
            #     write_file('explain.txt', '\t'.join([str(x_i) for x_i in attention_score.detach().cpu().numpy().tolist()])+'\n' )
            #
            #     write_file('pred_codes.txt', str(qids[b])+'\t'.join(pred[b].cpu().numpy())+'\n')
            #     write_file('path.txt', '\t'.join(paths[b])+'\n' )
            #     write_file('path_attention.txt', '\t'.join([str(x_i) for x_i in  attention_score.detach().cpu().numpy().tolist()])+'\n' )

            preds.append(pred)
            # loss = loss_fn(logits, Diag_labels)
            # total_loss += loss.item()

        # print('evaluate loss: \t', total_loss)
        preds = torch.vstack(preds).detach().cpu().numpy()
        labels = torch.vstack(labels).detach().cpu().numpy()
        f1_score = f1(labels, preds)
        prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])
        # print(name+' f1_score: %.8f --- top_k_recall: %.4f, %.4f, %.4f, %.4f'
        #           % (f1_score, recall[0], recall[1], recall[2], recall[3]))
        return f1_score, recall


def evaluate_codes_copy(eval_set, model, name='eval'):
    model.eval()
    total_loss = 0.0
    preds = []
    loss_fn = torch.nn.BCELoss()
    labels = []
    with torch.no_grad():
        for qids, simPatients, Hf_labels, Diag_labels, main_codes, sub_codes1, sub_codes2, *input_data in tqdm(eval_set,
                                                                                                               desc=name):
            Diag_labels = Diag_labels.float()
            labels.append(Diag_labels)
            # print(simPatients)
            # logits, paths, attention_score = model(*input_data)  # 前向传播，得到输出向量
            logits, _, _, _, _ = model(simPatients, main_codes, sub_codes1, sub_codes2, *input_data)  # 前向传播，得到输出向量
            pred = torch.argsort(logits, dim=-1, descending=True)
            # for b in range( pred.size(0) ):
            #     write_file('explain.txt', str(qids[b]) )
            #     write_file('explain.txt', '\t'.join([str(x_i) for x_i in pred[b].cpu().numpy()])+'\n')
            #     write_file('explain.txt', '\t'.join([str(x_i) for x_i in Diag_labels[b].cpu().numpy()])+'\n')
            #     write_file('explain.txt', '\t'.join(paths[b])+'\n')
            #     write_file('explain.txt', '\t'.join([str(x_i) for x_i in attention_score.detach().cpu().numpy().tolist()])+'\n' )
            #
            #     write_file('pred_codes.txt', str(qids[b])+'\t'.join(pred[b].cpu().numpy())+'\n')
            #     write_file('path.txt', '\t'.join(paths[b])+'\n' )
            #     write_file('path_attention.txt', '\t'.join([str(x_i) for x_i in  attention_score.detach().cpu().numpy().tolist()])+'\n' )

            preds.append(pred)
            loss = loss_fn(logits, Diag_labels)
            total_loss += loss.item()

        # print('evaluate loss: \t', total_loss)
        preds = torch.vstack(preds).detach().cpu().numpy()
        labels = torch.vstack(labels).detach().cpu().numpy()
        f1_score = f1(labels, preds)
        prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])
        return f1_score, recall


def attention(query, key):
    attn_score = torch.matmul(query, key.transpose(-2, -1))
    attn_score = F.softmax(attn_score, dim=-1)
    return attn_score


def evaluate_codes_pre(eval_set, model, device, name='eval'):
    model.eval()
    total_loss = 0.0
    preds = []
    loss_fn = torch.nn.BCELoss()
    labels = []
    with torch.no_grad():
        all_embeddings = []
        all_pids = []
        for batch_triple in eval_set: 
            pids, batch, td1, td2, ages, genders, ethnicitys, diag_labels, hf_labels = batch_triple
            # pid_test = pid[0]
            batch = torch.tensor(batch).to(device)
            td1 = torch.tensor(td1).to(device)
            td2 = torch.tensor(td2).to(device)
            ages = torch.tensor(ages).to(device)
            genders = torch.tensor(genders).to(device)
            ethnicitys = torch.tensor(ethnicitys).to(device)
            embeddings = model(batch, td1, td2, ages, genders, ethnicitys)
            all_embeddings.append(embeddings.cpu().detach().numpy())
            all_pids = pids

        all_embeddings = np.vstack(all_embeddings)  # shape: (total_patients, embed_dim)

        for pid_test, batch, td1, td2, ages, genders, ethnicitys, diag_labels, hf_labels in tqdm(eval_set, desc=name):
            pid_test = pid_test[0]
            batch = torch.tensor(batch).to(device)
            td1 = torch.tensor(td1).to(device)
            td2 = torch.tensor(td2).to(device)
            ages = torch.tensor(ages).to(device)
            genders = torch.tensor(genders).to(device)
            ethnicitys = torch.tensor(ethnicitys).to(device)
            diag_labels = torch.tensor(diag_labels).clone().detach().to(device)
            embeddings = model(batch, td1, td2, ages, genders, ethnicitys)  # shape: (batch_size, embed_dim)

            similarity_matrix = cosine_similarity(embeddings.cpu().detach().numpy(),
                                                  all_embeddings)  # shape: (batch_size, total_patients)

            topk_indices = np.argsort(similarity_matrix, axis=1)[:, -10:]  # shape: (batch_size, 10)

            topk_embeddings = torch.tensor(all_embeddings[topk_indices]).to(
                device)  # shape: (batch_size, 10, embed_dim)
            attn_weights = attention(embeddings, topk_embeddings)  # shape: (batch_size, 10)
            weighted_topk = torch.sum(attn_weights.unsqueeze(-1) * topk_embeddings,
                                      dim=1)  # shape: (batch_size, embed_dim)
            weighted_topk_aggregated = torch.sum(weighted_topk, dim=1)  # [batch_size, embed_dim]

            final_embedding = torch.cat([embeddings, weighted_topk_aggregated], dim=-1)

            fc_layer = nn.Linear(9760, 4880).to(device)

            final_embedding = fc_layer(final_embedding)

            logits = F.sigmoid(final_embedding)
            pred = torch.argsort(logits, dim=-1, descending=True)
            diag_labels = torch.tensor(diag_labels).clone().detach().to(device)
            preds.append(pred)
            labels.append(diag_labels)
        preds = torch.vstack(preds).detach().cpu().numpy()
        labels = torch.vstack(labels).detach().cpu().numpy()
        f1_score = f1(labels, preds)
        prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])
        return f1_score, recall


def evaluate_codes_pre2(eval_set, model, device, name='eval'):
    model.eval()
    total_loss = 0.0
    preds = []
    loss_fn = torch.nn.BCELoss()
    labels = []
    with torch.no_grad():
        for pid_test, batch, td1, td2, ages, genders, ethnicitys, diag_labels, hf_labels in tqdm(eval_set, desc=name):
            pid_test = pid_test[0]
            batch = torch.tensor(batch).to(device)
            td1 = torch.tensor(td1).to(device)
            td2 = torch.tensor(td2).to(device)
            ages = torch.tensor(ages).to(device)
            genders = torch.tensor(genders).to(device)
            ethnicitys = torch.tensor(ethnicitys).to(device)
            diag_labels = torch.tensor(diag_labels).to(device)
            diag_labels = diag_labels.to(device).float()
            logits = model(batch, td1, td2, ages, genders, ethnicitys)
            pred = torch.argsort(logits, dim=-1, descending=True)
            preds.append(pred)
            labels.append(diag_labels)

        preds = torch.vstack(preds).detach().cpu().numpy()
        labels = torch.vstack(labels).detach().cpu().numpy()
        f1_score = f1(labels, preds)
        prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])
        return f1_score, recall


def evaluate_codes_hita(eval_set, model, name='eval', options=None):
    model.eval()
    total_loss = 0.0
    preds = []
    loss_fn = torch.nn.BCELoss()
    labels = []
    with torch.no_grad():
        for qids, Hf_labels, Diag_labels, batch_diag, batch_time_seq, \
                *input_data in tqdm(eval_set, desc=name):
            batch_diagnosis_codes, batch_time_step = adjust_input_hita(batch_diag, batch_time_seq, max_len=50,
                                                                       n_diagnosis_codes=4880)
            lengths = np.array([len(seq) for seq in batch_diagnosis_codes])  # bsz
            maxlen = np.max(lengths)
            hita_model = TransformerTime(n_diagnosis_codes=4880, batch_size=16, options=options)
            hita_model.cuda()
            hita_embedding = hita_model(batch_diagnosis_codes,
                                        batch_time_step,
                                        options, maxlen)

            Diag_labels = Diag_labels.float()
            labels.append(Diag_labels)
            logits, _ = model(hita_embedding, *input_data) 
            pred = torch.argsort(logits, dim=-1, descending=True)
            preds.append(pred)
            loss = loss_fn(logits, Diag_labels)
            total_loss += loss.item()

        print('evaluate loss: \t', total_loss)
        preds = torch.vstack(preds).detach().cpu().numpy()
        labels = torch.vstack(labels).detach().cpu().numpy()
        f1_score = f1(labels, preds)
        prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])
        return f1_score, recall

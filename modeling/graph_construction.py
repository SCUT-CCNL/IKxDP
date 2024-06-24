import json
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
import numpy as np
# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# mimiciii
# with open("./data/mimic/statement/all.statement.jsonl", "r") as f:
# # with open("./data/mimic/statement/patient_data.json", "r") as f:
#     patient_records = [json.loads(line) for line in f]

# mimiciv
with open("./data/mimiciv/statement/all_mimiciv.statement.jsonl", "r") as f:
# with open("./data/mimic/statement/patient_data.json", "r") as f:
    patient_records = [json.loads(line) for line in f]

cooccurrence = defaultdict(int)
theta = 7  
for record in tqdm(patient_records, desc='cooccurrence'):
    for visit in record['all_records']:
        for i in range(len(visit)):
            for j in range(i + 1, len(visit)):
                disease_a, disease_b = sorted([visit[i], visit[j]])
                cooccurrence[(disease_a, disease_b)] += 1
import matplotlib.pyplot as plt

counts = list(cooccurrence.values())

print("Mean:", np.mean(counts))
print("Median:", np.median(counts))
print("Min:", np.min(counts))
print("Max:", np.max(counts))
print("Std:", np.std(counts))
cooccurrences = list(cooccurrence.values())

threshold_75 = np.percentile(cooccurrences, 75)
threshold_90 = np.percentile(cooccurrences, 90)

print('threshold_75:', threshold_75)
print('threshold_90:', threshold_90)

# plt.hist(counts, bins=50)
# plt.xlabel('Co-occurrence Counts')
# plt.ylabel('Frequency')
# plt.title('Histogram of Disease Pair Co-occurrence Counts')
# plt.show()


cooccurrence = {pair: count for pair, count in cooccurrence.items() if count > threshold_90}
#

def construct_patient_graph(patient_record, cooccurrence):
    graph_seq = []
    pid2graph = {}
    visit = patient_record['all_records']
    pid = patient_record['pid']

    for i in range(len(visit)):
        graph = nx.Graph()
        if i == 0:
            last_record = None
        else:
            last_record = visit[i-1]

        record_t = set(visit[i])
        record_t_minus_1 = set(last_record) if last_record else set()

        cooccurring_diseases_t_minus_1 = set()
        for disease in record_t_minus_1:
            cooccurring_diseases_t_minus_1.update([d for d, _ in cooccurrence if d == disease or _ == disease])

        for disease in record_t:
            graph.add_node(disease, diagnose_type='current')
        for disease in record_t_minus_1:
            graph.add_node(disease, diagnose_type='previous')
        for disease in cooccurring_diseases_t_minus_1:
            graph.add_node(disease, diagnose_type='cooccurring')

        for disease_a in record_t:
            for disease_b in record_t:
                if disease_a != disease_b:
                    graph.add_edge(disease_a, disease_b, relation='Cooccurring')

        for disease_a in record_t:
            for disease_b in record_t_minus_1:
                if disease_a == disease_b:
                    graph.add_edge(disease_a, disease_b, relation='Inherited')

        for disease_a in record_t:
            for disease_b in cooccurring_diseases_t_minus_1:
                if disease_b not in record_t_minus_1:
                    graph.add_edge(disease_a, disease_b, relation='Caused')

        if len(graph.edges) > 0:
            graph.remove_nodes_from(list(nx.isolates(graph)))

        graph_seq.append(graph.copy())
        pid2graph[pid] = graph_seq

    return graph_seq, pid, pid2graph


patient_graphs = [construct_patient_graph(record, cooccurrence) for record in
                  tqdm(patient_records, desc='construct graph')]

# mimic iii
# index2id = np.load('./data/icd2idx.npy', allow_pickle=True).item()
# id2index = {v: k for k, v in index2id.items()}

# mimic iv
index2id = np.load('./data/icd2idx-mimiciv.npy', allow_pickle=True).item()
id2index = {v: k for k, v in index2id.items()}

def patient_graphs_to_pyg_data():
    patient_data = []
    pids = []
    pid2graph_data = {}
    max_length = 512
    for graphs, pid, pid2graph in tqdm(patient_graphs, desc='construct pyg_data'):
        pyg_data_list = []
        for graph in graphs:
            x = [index2id[node] for node in graph.nodes]

            # node4bert = [index2id[node] for node in graph.nodes]
            # embeddings = []
            # for i in range(0, len(node4bert), max_length):
            #     batch = node4bert[i: i + max_length]
            #     batch = torch.tensor(batch).unsqueeze(0).to(device)
            #     embedding = bert_model(batch).last_hidden_state.squeeze()
            #     embeddings.append(embedding)
            #     x = torch.cat(embeddings, dim=0)

            edge_index = torch.tensor([[list(graph.nodes).index(u), list(graph.nodes).index(v)] for u, v in graph.edges], dtype=torch.long).t().contiguous().to(device)
            node_type = torch.tensor([1 if graph.nodes[node]['diagnose_type'] == 'current' else (2 if graph.nodes[node]['diagnose_type'] == 'previous' else 0) for node in graph.nodes], dtype=torch.long).to(device)
            edge_type = torch.tensor([1 if graph.edges[u, v]['relation'] == 'Cooccurring' else (2 if graph.edges[u, v]['relation'] == 'Caused' else 0) for u, v in graph.edges], dtype=torch.long).to(device)
            data = Data(x=x, edge_index=edge_index, node_type=node_type, edge_type=edge_type).to(device)
            pyg_data_list.append(data)
        patient_data.append(pyg_data_list)
        pids.append(pid)
        pid2graph_data[pid] = pyg_data_list

    # torch.save(pid2graph_data, './data/mimic/statement/pid2graph_data_bert.pt')

    # mimiciv
    torch.save(pid2graph_data, './data/mimiciv/statement/pid2graph_data_bert.pt')



    print('save pid2graph_data.pt successfully!')
    return patient_data, pids, pid2graph_data

if __name__ == '__main__':
    patient_graphs_to_pyg_data()
import argparse
import random
from Utils.process_smiles import *
import csv
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from Utils.process_smiles import smile_to_graph, GraphDataset_v, collate  # 按你项目路径导入

def drug_fea_process_with_sim(smiles_file, drug_sim_input, drug_num):
    """
    构建融合了相似性特征的药物图
    支持 drug_sim_input 是 文件路径(str) 或 numpy.ndarray
    """
    # === 1. 自动读取相似性矩阵 ===
    if isinstance(drug_sim_input, str):
        drug_sim_matrix = np.loadtxt(drug_sim_input, delimiter='\t')
    elif isinstance(drug_sim_input, np.ndarray):
        drug_sim_matrix = drug_sim_input
    else:
        raise TypeError(f"drug_sim_input 类型错误：应为 str 或 ndarray，但收到 {type(drug_sim_input)}")

    assert drug_sim_matrix.shape[0] == drug_num, \
        f"相似性矩阵尺寸不符：{drug_sim_matrix.shape} vs drug_num={drug_num}"

    # === 2. 加载 SMILES 并生成分子图 ===
    reader = csv.reader(open(smiles_file, encoding='utf-8'))
    smile_graph = []

    for i, item in enumerate(reader):
        smile = item[1]
        try:
            c_size, features, edge_index = smile_to_graph(smile)
        except Exception as e:
            print(f"[跳过] 第 {i} 个 SMILES '{smile}' 解析失败: {e}")
            continue

        g = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long)
        )
        g.__setitem__('c_size', torch.tensor([c_size], dtype=torch.long))
        g.__setitem__('cid', torch.tensor([i], dtype=torch.long))

        # 拼接 drug 相似性向量
        sim_vec = torch.tensor(drug_sim_matrix[i], dtype=torch.float).unsqueeze(0)
        sim_feature = sim_vec.repeat(g.x.shape[0], 1)
        g.x = torch.cat([g.x, sim_feature], dim=1)

        smile_graph.append(g)

    # === 3. 构建 PyG 批次 ===
    dru_dataset = GraphDataset_v(xc=smile_graph, cid=[i for i in range(len(smile_graph))])
    dru_loader = DataLoader(dataset=dru_dataset, batch_size=drug_num, shuffle=False, collate_fn=collate)

    for _, batch_drug in enumerate(dru_loader):
        return batch_drug

def parameters_set():
    parser = argparse.ArgumentParser(description='using GAT process association matrix and GIN&Conv process feature')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='which GPU to use. Set -1 to use CPU.')
    parser.add_argument('--hypergraph_loss_ratio', type=float, default=0.8,
                        help='the weight of hypergraph_loss_ratio')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of training epochs')
    parser.add_argument('--top_1', type=int, default=1,
                        help='hit@top,ndcg@top')
    parser.add_argument('--top_3', type=int, default=3,
                        help='hit@top,ndcg@top')
    parser.add_argument('--top_5', type=int, default=5,
                        help='hit@top,ndcg@top')

    parser.add_argument('--train_data', type=str, default='4_type',
                        help='train_data_neg_type: '
                             'ratio_4_type,4_type,del_1th_type,del_2th_type,del_3th_type,del_4th_type,'
                             'only_1th_type,only_2th_type,only_3th_type,only_4th_type')  # 4_type

    parser.add_argument('--bio_out_dim', type=int, default=32,
                        help='bio_out_dim of the dru,mic,dis ')  # 32 warm-start最佳, 8 cold-start最佳
    parser.add_argument('--hgnn_dim_1', type=int, default=512,
                        help='hgnn_dim_1 of feature')  # 512

    parser.add_argument('--cold_class', type=str, default='dis')

    parser.add_argument('--lr', type=float, default=0.001,  # 0.005
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='the random seeds')
    parser.add_argument('--k_fold', type=int, default=5,
                        help='k_fold')
    parser.add_argument('--BATCH_SIZE', type=int, default=30,
                        help='BATCH_SIZE')

    args = parser.parse_args()
    return args

class Metrics():
    def __init__(self, step, test_data, predict_1, batch_size, top):
        self.pair = []
        self.step = step
        self.test_data = test_data
        self.predict_1 = predict_1
        self.top = top
        self.dcgsum = 0
        self.idcgsum = 0
        self.hit = 0
        self.ndcg = 0
        self.batch_size = batch_size
        self.val_top = []

    def hits_ndcg(self):
        for i in range(self.step * self.batch_size, (self.step + 1) * self.batch_size):
            g = []
            g.extend([self.test_data[i, 3], self.predict_1[i].item()])
            self.pair.append(g)
        np.random.seed(1)
        np.random.shuffle(self.pair)
        pre_val = sorted(self.pair, key=lambda item: item[1], reverse=True)
        self.val_top = pre_val[0: self.top]

        for i in range(len(self.val_top)):
            if self.val_top[i][0] == 1:
                self.hit = self.hit + 1
                self.dcgsum = (2 ** self.val_top[i][0] - 1) / np.log2(i + 2)
                break
        ideal_list = sorted(self.val_top, key=lambda item: item[0], reverse=True)
        for i in range(len(ideal_list)):
            if ideal_list[i][0] == 1:
                self.idcgsum = (2 ** ideal_list[i][0] - 1) / np.log2(i + 2)
                break
        if self.idcgsum == 0:
            self.ndcg = 0
        else:
            self.ndcg = self.dcgsum / self.idcgsum
        return self.hit, self.ndcg

def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[
        (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix * real_score.T

    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
    PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index, 0]
    accuracy = accuracy_list[max_index, 0]
    specificity = specificity_list[max_index, 0]
    recall = recall_list[max_index, 0]
    precision = precision_list[max_index, 0]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]

def hit_ndcg_value(pred_val, val_data, top):
    loader_val = torch.utils.data.DataLoader(dataset=pred_val, batch_size=30, shuffle=False)
    hits = 0
    ndcg_val = 0
    for step, batch_val in enumerate(loader_val):
        metrix = Metrics(step, val_data, pred_val, batch_size=30, top=top)
        hit, ndcg = metrix.hits_ndcg()
        hits = hits + hit
        ndcg_val = ndcg_val + ndcg
    hits = hits / int((len(val_data)) / 30)
    ndcg = ndcg_val / int((len(val_data)) / 30)
    return hits, ndcg

def write_type_1234(file_name, fold_th, hits_1_max, ndcg_1_max, hits_2_max, ndcg_2_max, hits_3_max, ndcg_3_max,
                    hits_4_max, ndcg_4_max, epoch_max_1=None, epoch_max_2=None, epoch_max_3=None,
                    epoch_max_4=None, top='top1'):
    with open(file_name, 'a') as f:
        f.write(str(fold_th) + '\t' + str(top) + '\t' + str(hits_1_max) + '\t' + str(ndcg_1_max) + '\t' + str(
            epoch_max_1) + '\t' + str(hits_2_max) + '\t' + str(ndcg_2_max) + '\t' + str(
            epoch_max_2) + '\t' + str(hits_3_max) + '\t' + str(ndcg_3_max) + '\t' + str(
            epoch_max_3) + '\t' + str(hits_4_max) + '\t' + str(ndcg_4_max) + '\t' + str(
            epoch_max_4) + '\n')

def build_hypergraph(data):
    pairs = np.array(data[:, 0:3]).reshape(1, -1)
    pairs_num = np.expand_dims(np.arange(len(data)), axis=-1)
    hyper_edge_num = np.concatenate((pairs_num, pairs_num, pairs_num), axis=1)
    hyper_edge_num = np.array(hyper_edge_num).reshape(1, -1)
    hyper_graph = np.concatenate((pairs, hyper_edge_num), axis=0)
    hyper_graph = torch.from_numpy(hyper_graph).type(torch.LongTensor)
    return hyper_graph
# ====== 同类相似性 → 组超边 + 组内子边 工具 ======
import numpy as np
import torch

def normalize_sim(sim, to01=True, zero_diag=True, symmetrize=True):
    S = sim.copy().astype(np.float32)
    if to01:
        mn, mx = float(S.min()), float(S.max())
        if mx > mn: S = (S - mn) / (mx - mn + 1e-12)
    if zero_diag:
        np.fill_diagonal(S, 0.0)
    if symmetrize:
        S = 0.5 * (S + S.T)
    return S

def knn_hyperedges_from_similarity(sim, k, etype, weight_mode='mean'):
    """
    sim: (N,N) 同类相似度（已归一化、对称，diag=0 更好）
    k  : 每个“中心”取 k 个最近邻（不含自己），形成一条“组超边”：[center, n1, n2, ...]
    etype: 'drug'|'mirna'|'disease' —— 只影响全局ID偏移
    weight_mode: 'mean'|'max'|'sum' —— 该组超边的单一权重如何由 sim 聚合
    返回：
      group_edges: List[List[int]]  （全局ID）
      group_w    : List[float]      （与 group_edges 对齐的组边权）
      # 供“组内注意”使用的原始 (center, neigh, sim) 三元组
      center_idx: List[int]         （局部索引0..N-1）
      neigh_idx : List[List[int]]   （每个中心对应的“邻居局部索引列表”）
      neigh_sim : List[List[float]] （与上面对应的相似度列表）
    """
    N = sim.shape[0]
    offset = {'drug':0, 'mirna':16, 'disease':16+8}[etype]
    group_edges, group_w = [], []
    center_idx, neigh_idx, neigh_sim = [], [], []

    for i in range(N):
        idx = np.argsort(-sim[i])  # 降序
        neigh = [j for j in idx if j != i][:k]
        if len(neigh) == 0: continue

        nodes = [i+offset] + [j+offset for j in neigh]
        group_edges.append(nodes)

        sims = sim[i, neigh].astype(np.float32)
        if   weight_mode == 'mean': w = float(np.mean(sims))
        elif weight_mode == 'max' : w = float(np.max(sims))
        elif weight_mode == 'sum' : w = float(np.sum(sims))
        else:                       w = float(np.mean(sims))
        group_w.append(w)

        center_idx.append(i)
        neigh_idx.append(neigh)
        neigh_sim.append(sims.tolist())

    return group_edges, group_w, center_idx, neigh_idx, neigh_sim

def intra_subedges_from_similarity(center_idx, neigh_idx, neigh_sim, etype):
    """
    把每条“组超边”的组内连接拆成多条 2-节点“子边”（中心-邻居），用于模拟“组内注意（逐邻居差异）”
    返回：
      sub_edges: List[List[int]]  —— 每条子边 [global(center), global(neigh)]
      sub_w    : List[float]      —— 每条子边的初始权重（直接用对应 sim）
    """
    offset = {'drug':0, 'mirna':16, 'disease':16+8}[etype]
    sub_edges, sub_w = [], []
    for c, neighs, sims in zip(center_idx, neigh_idx, neigh_sim):
        for nj, sj in zip(neighs, sims):
            sub_edges.append([c+offset, nj+offset])
            sub_w.append(float(sj))
    return sub_edges, sub_w

def merge_triple_and_homo_groups(triples,
                                 drug_pack, mir_pack, dis_pack,
                                 add_subedges=True,
                                 triple_w=1.0, group_w_scale=0.6, sub_w_scale=0.6):
    """
    仅合并：三元超边 + 三类同类kNN“组超边” (+ 可选子边)
    triples: 训练/负采样所得三元列表（全局ID、只用前三列 [d,m,z]）
    *_pack: 形如：
      {
        "group_edges": [...],
        "group_w": [...],
        "center_idx": [...],
        "neigh_idx": [...],
        "neigh_sim": [...]
      }
    add_subedges: 是否把“中心-邻居”子边也加入（用于组内注意）
    *scale: 三种边初始权重的相对系数
    返回：
      hyperedges: List[List[int]]
      edge_w    : torch.FloatTensor
      edge_type : torch.LongTensor   0=triple,
                                     1=group_drug, 2=group_mir, 3=group_dis,
                                     4=sub_drug,   5=sub_mir,   6=sub_dis
    """
    hyperedges, ws, types = [], [], []

    # 三元
    for t in triples:
        d, m, z = int(t[0]), int(t[1]), int(t[2])
        hyperedges.append([d,m,z]); ws.append(float(triple_w)); types.append(0)

    def add_groups(pack, tcode, scale):
        for E, w in zip(pack["group_edges"], pack["group_w"]):
            hyperedges.append([int(x) for x in E]); ws.append(scale*float(w)); types.append(tcode)

    add_groups(drug_pack, 1, group_w_scale)
    add_groups(mir_pack,  2, group_w_scale)
    add_groups(dis_pack,  3, group_w_scale)

    if add_subedges:
        def add_sub(pack, etype_code, etype):
            subE, subW = intra_subedges_from_similarity(pack["center_idx"], pack["neigh_idx"], pack["neigh_sim"], etype)
            for E, w in zip(subE, subW):
                hyperedges.append([int(x) for x in E]); ws.append(sub_w_scale*float(w)); types.append(etype_code)
        add_sub(drug_pack, 4, 'drug')
        add_sub(mir_pack,  5, 'mirna')
        add_sub(dis_pack,  6, 'disease')

    edge_w    = torch.tensor(ws, dtype=torch.float32)
    edge_type = torch.tensor(types, dtype=torch.long)
    return hyperedges, edge_w, edge_type

def to_hyperedge_index(hyperedges):
    # incidence: [2, num_incidence] with rows = [node, edge_id]
    inc = []
    for eid, nodes in enumerate(hyperedges):
        for n in nodes:
            inc.append([int(n), int(eid)])
    return torch.tensor(inc, dtype=torch.long).t()

import numpy as np

def threshold_hyperedges_from_similarity(
    S: np.ndarray,
    thr: float = 0.9,
    etype: str = 'drug',
    weight_mode: str = 'softmax',  # 'softmax' | 'mean' | 'none'
    tau: float = 0.2,
    add_subedges: bool = False,
    fallback_min_k: int = 1,       # 若没有任何邻居 ≥thr，是否回退连 top-k（0=不回退）
):
    """
    基于“相似度阈值”的组超边构图：
      - 对每个中心 i，找所有 j 使得 S[i,j] >= thr（排除 i 自身），这些 j 与 i 共同形成一条“组超边”；
      - 组边权重 = 组内邻居权重（见 weight_mode）的均值；
      - 可选：为每个中心-邻居添加“子边”（组内注意）。

    返回：
      group_edges:  List[List[int]]    # 每条为 [i] + 邻居
      group_w:     List[float]         # 每条组边的权重
      center_idx:  List[int]
      neigh_idx:   List[List[int]]
      neigh_sim:   List[List[float]]
      （若 add_subedges=True 还会返回 pack['sub_edges'], pack['sub_w'] 供 merge 时用）
    """
    N = S.shape[0]
    group_edges, group_w, center_idx, neigh_idx, neigh_sim = [], [], [], [], []
    sub_edges, sub_w = [], []

    for i in range(N):
        # 选出“过阈值”的邻居
        mask = S[i] >= thr
        mask[i] = False
        idx = np.where(mask)[0].tolist()

        # 如无邻居且需要回退，取 top-k
        if len(idx) == 0 and fallback_min_k > 0:
            row = S[i].copy()
            row[i] = -np.inf
            idx = np.argsort(-row)[:fallback_min_k].tolist()

        if len(idx) == 0:
            continue

        sims = np.array([S[i, j] for j in idx], dtype=np.float32)

        # 计算邻居权重
        if weight_mode == 'softmax':
            logits = sims / max(tau, 1e-6)
            logits = logits - logits.max()
            w_ij = np.exp(logits)
            w_ij = w_ij / (w_ij.sum() + 1e-12)
        elif weight_mode == 'mean':
            w_ij = np.ones_like(sims, dtype=np.float32) / len(sims)
        elif weight_mode == 'none':
            w_ij = np.ones_like(sims, dtype=np.float32)
        else:
            raise ValueError(f'Unknown weight_mode: {weight_mode}')

        # 组超边（中心 + 邻居）
        edge_nodes = [i] + idx
        group_edges.append(edge_nodes)
        group_w.append(float(w_ij.mean()))  # 你也可以改成 sum

        center_idx.append(i)
        neigh_idx.append(idx)
        neigh_sim.append(sims.tolist())

        if add_subedges:
            for j, wij in zip(idx, w_ij):
                sub_edges.append([i, j])
                sub_w.append(float(wij))

    pack = (group_edges, group_w, center_idx, neigh_idx, neigh_sim)
    if add_subedges:
        return (*pack, sub_edges, sub_w)
    return pack


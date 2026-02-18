# main_indep_hyper_graph_16_mirna_updated_可运行.py

import os
import torch
import numpy as np
import torch.backends.cudnn
import torch.backends.cudnn
from model import *
from Utils.Tools2 import *
from Utils.right_negative_sample_generate import *
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score,precision_score, recall_score, confusion_matrix

from model import *  # HGNN / HgnnEncoder / BioEncoder / Decoder
from Utils.Tools2 import *  # parameters_set / hit_ndcg_value / kNN工具等
from Utils.right_negative_sample_generate import neg_data_generate
from Utils.Tools2 import drug_fea_process_with_sim

# ----------------- 常量（与你工程约定一致） -----------------
DRUG_NUM = 16
MIRNA_NUM = 8
DISEASE_NUM = 252
N_SAMPLES_PER_CLASS = 2680


def build_global_maps(adj_all_np):
    """
    adj_all_np: 全量三元组 (N,3) int，列为 [drug, mirna, disease]（不含label）
    返回3个dict: drug_map, mir_map, dis_map，把原始ID统一映射到0..n-1。
    """
    drug_ids = np.unique(adj_all_np[:, 0])
    mir_ids  = np.unique(adj_all_np[:, 1])
    dis_ids  = np.unique(adj_all_np[:, 2])
    drug_map = {int(old): i for i, old in enumerate(sorted(map(int, drug_ids)))}
    mir_map  = {int(old): i for i, old in enumerate(sorted(map(int, mir_ids)))}
    dis_map  = {int(old): i for i, old in enumerate(sorted(map(int, dis_ids)))}
    return drug_map, mir_map, dis_map


def apply_maps(triples_np, drug_map, mir_map, dis_map):
    """
    triples_np: 任意子集三元组 (N,4) 或 (N,3)，前三列为 d,m,s，可能带label列
    作用：用全局映射把 d/m/s 原始ID 改成 0..n-1 的压缩索引；label列原样保留。
    """
    has_label = (triples_np.shape[1] >= 4)
    t = triples_np.copy()
    t = t.astype(int)
    d = t[:, 0]
    m = t[:, 1]
    s = t[:, 2]
    try:
        d = np.vectorize(lambda x: drug_map[int(x)])(d)
        m = np.vectorize(lambda x: mir_map[int(x)])(m)
        s = np.vectorize(lambda x: dis_map[int(x)])(s)
    except KeyError as e:
        raise KeyError(f"[ID映射失败] 发现未出现在全量映射中的原始ID: {e}. 请确认所有fold来源于同一全量集合。")
    if has_label:
        out = np.column_stack([d, m, s, t[:, 3]])
    else:
        out = np.column_stack([d, m, s])
    return out


def train1(drug_fea, mic_fea, dis_fea, hg_pos,  train_data, epoch, device):
    model.train()
    optimizer.zero_grad()
    d_idx = torch.from_numpy(train_data[:, 0].astype(np.int64)).to(device)
    m_idx = torch.from_numpy(train_data[:, 1].astype(np.int64)).to(device)
    z_idx = torch.from_numpy(train_data[:, 2].astype(np.int64)).to(device)

    _, pred,_  = model(drug_fea, mic_fea, dis_fea, hg_pos,
                                                                        d_idx, m_idx,
                                                                        z_idx, )
    logits = pred  # pred即上面的logit
    # train1() 里，在构造 BCEWithLogitsLoss 前：
    labels = torch.from_numpy(train_data[:, 3]).float().to(device).view(-1)
    n_pos = labels.sum().item()
    n_neg = len(train_data) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1e-6)], device=device)

    loss_1 = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
        logits.view(-1),
        labels
    )


    T = max(1, args.epochs - 1)


    alpha = max(0.3, 1 - np.exp(-epoch / 50))
    loss = loss_1

    loss.backward()
    optimizer.step()

    # 计算训练F1-score
    pred_binary = (pred > 0.35).float().cpu().numpy()
    train_f1 = f1_score(train_data[:, 3], pred_binary)

    print('epoch:{:02d}, loss_train:{:.6f}, F1_train:{:.4f}'.format(epoch + 1, loss.item(), train_f1))
    return loss


max_auc_pr = 0


@torch.no_grad()
@torch.no_grad()
def test1(drug_fea, mic_fea, dis_fea, hg_pos,  val_data, device,epoch):
    import numpy as np
    from sklearn.metrics import precision_recall_curve, f1_score, recall_score, confusion_matrix, auc, roc_auc_score

    model.eval()

    d_idx = torch.from_numpy(val_data[:, 0].astype(np.int64)).to(device)
    m_idx = torch.from_numpy(val_data[:, 1].astype(np.int64)).to(device)
    z_idx = torch.from_numpy(val_data[:, 2].astype(np.int64)).to(device)
    # pred_val 是 **logit**
    _, pred_val,graph_embed_pos  = model(drug_fea, mic_fea, dis_fea,hg_pos, d_idx, m_idx, z_idx, )

    labels = torch.from_numpy(val_data[:, 3]).float().to(device).view(-1)
    n_pos = labels.sum().item()
    n_neg = len(val_data) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1e-6)], device=device)

    # BCE损失
    loss_1 = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
        pred_val.view(-1),
        labels
    )

    # DGI损失


    # 组合损失（使用与训练相同的alpha权重）


    alpha = max(0.3, 1 - np.exp(-epoch / 50))
    val_loss =loss_1

    true = val_data[:, 3].astype(int)
    prob = torch.sigmoid(pred_val).detach().cpu().numpy()  # logit → 概率

    # AUC-ROC / AUC-PR
    auc_roc = roc_auc_score(true, prob)
    prec, rec, thr = precision_recall_curve(true, prob)
    auc_pr = auc(rec, prec)

    # 阈值寻优（按 F1 最大）
    f1s = (2 * prec * rec) / (prec + rec + 1e-12)
    best_idx = f1s.argmax()
    # 注意：precision_recall_curve 的阈值数组长度比点少 1
    #best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
    best_thr=0.5
    pred_bin = (prob >= best_thr).astype(int)
    f1 = f1_score(true, pred_bin)
    prec_ = precision_score(true, pred_bin)
    rec_ = recall_score(true, pred_bin)
    tn, fp, fn, tp = confusion_matrix(true, pred_bin).ravel()

    print('loss_train:{:.6f}'.format( val_loss.item()))
    print(f'AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}')
    print(f'F1: {f1:.4f}, Recall: {rec_:.4f},Precision:{prec_:.4f}')
    print(f'Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}')
    print(f'best_thr: {best_thr:.4f},')
    print('-' * 50)

    return auc_roc, auc_pr, f1, rec_, prec_, val_loss, true, prob,graph_embed_pos


def prepare_fixed_10fold_multineg(pos_samples, neg_samples_list, n_folds=10, seed=42):
    """
    固定十折交叉验证划分，支持4类负样本。
    每类负样本与同一批正样本进行匹配划分。
    参数：
        pos_samples: ndarray (N,3)
        neg_samples_list: list(4个负样本集，每个ndarray(N,3))
    返回：
        folds[fold_idx][neg_type_idx] = {'train': ndarray(...,4), 'val': ndarray(...,4)}
    """
    np.random.seed(seed)

    # 固定正样本 2680
    pos_selected = pos_samples[:2685].copy()
    np.random.shuffle(pos_selected)

    # 每类负样本固定2680
    neg_selected_list = []
    shuffled_3 = []
    for i in range(3):
        neg = neg_samples_list[i].copy()
        np.random.shuffle(neg)  # 先打乱
        selected = neg[:2685]  # 再取 2685 个
        shuffled_3.append(neg)  # 保存完整打乱后（供第四类使用）
        neg_selected_list.append(selected)  # 保存前3类采样结果
    #
    # neg = neg_samples_list[0].copy()
    # np.random.shuffle(neg)  # 先打乱
    # selected = neg[:3440]  # 再取 2685 个
    # shuffled_3.append(neg)  # 保存完整打乱后（供第四类使用）
    # neg_selected_list.append(selected)  # 保存前3类采样结果
    #
    # neg = neg_samples_list[1].copy()
    # np.random.shuffle(neg)  # 先打乱
    # selected = neg[:5370]  # 再取 2685 个
    # shuffled_3.append(neg)  # 保存完整打乱后（供第四类使用）
    # neg_selected_list.append(selected)  # 保存前3类采样结果
    #
    # neg = neg_samples_list[2].copy()
    # np.random.shuffle(neg)  # 先打乱
    # selected = neg[:5370]  # 再取 2685 个
    # shuffled_3.append(neg)  # 保存完整打乱后（供第四类使用）
    # neg_selected_list.append(selected)  # 保存前3类采样结果

    mixed_parts = []
    mixed_parts.append(shuffled_3[0][:895])
    mixed_parts.append(shuffled_3[1][:895])
    mixed_parts.append(shuffled_3[2][:895])
    mixed_all = np.vstack(mixed_parts)

    # 再打乱混合结果（保证第四类随机性）
    np.random.shuffle(mixed_all)
    neg_selected_list.append(mixed_all)
    # 分割正样本为10折
    fold_size = len(pos_selected) // n_folds
    pos_folds = [pos_selected[i*fold_size : (i+1)*fold_size if i < n_folds-1 else len(pos_selected)] for i in range(n_folds)]

    # 每类负样本都分10折
    neg_folds_list = []
    for neg_selected in neg_selected_list:
        fold_size_neg = len(neg_selected) // n_folds
        neg_folds = [neg_selected[i*fold_size_neg : (i+1)*fold_size_neg if i < n_folds-1 else len(neg_selected)]
                     for i in range(n_folds)]
        neg_folds_list.append(neg_folds)

    # 构造 folds 结构
    folds = [[None for _ in range(4)] for _ in range(n_folds)]

    for fold_idx in range(n_folds):
        # 当前折的正样本划分
        val_pos = pos_folds[fold_idx]
        train_pos = np.vstack([pos_folds[i] for i in range(n_folds) if i != fold_idx])
        pos_train_labeled = np.hstack([train_pos, np.ones((len(train_pos), 1))])
        pos_val_labeled = np.hstack([val_pos, np.ones((len(val_pos), 1))])

        for neg_type_idx in range(4):
            val_neg = neg_folds_list[neg_type_idx][fold_idx]
            train_neg = np.vstack([neg_folds_list[neg_type_idx][i] for i in range(n_folds) if i != fold_idx])
            neg_train_labeled = np.hstack([train_neg, np.zeros((len(train_neg), 1))])
            neg_val_labeled = np.hstack([val_neg, np.zeros((len(val_neg), 1))])

            # 合并并打乱
            train_data = np.vstack([pos_train_labeled, neg_train_labeled])
            val_data = np.vstack([pos_val_labeled, neg_val_labeled])
            np.random.shuffle(train_data)
            np.random.shuffle(val_data)

            folds[fold_idx][neg_type_idx] = {
                'train': train_data,
                'val': val_data
            }

    return folds

from sklearn.metrics.pairwise import euclidean_distances
def compute_dynamic_gip_kernels(train_triples):
    """
    根据训练数据动态计算三维GIP核
    train_triples: 训练集中的正样本三元组 [drug, mirna, disease]
    返回: (K_drug, K_mirna, K_disease)
    """
    # 构建训练数据的张量
    n_d = 16
    n_m = 8
    n_s = 252
    A_train = np.zeros((n_d, n_m, n_s), dtype=np.float32)
    for triple in train_triples:
        d, m, s = int(triple[0]), int(triple[1]), int(triple[2])
        A_train[d, m, s] = 1.0

    # 计算三维GIP核
    def gip_3d_from_slices(A_slices, eps=1e-12, diag_fix=True):
        N, H, W = A_slices.shape
        X = A_slices.reshape(N, H * W).astype(np.float64)
        D = euclidean_distances(X)
        D2 = D ** 2
        energy = np.sum(X ** 2, axis=1)
        gamma = 1.0 / (np.mean(energy) + eps)
        K = np.exp(-gamma * D2)
        if diag_fix:
            np.fill_diagonal(K, 1.0)
        return K.astype(np.float32)

    # 分别计算三类核
    K_drug = gip_3d_from_slices(A_train)  # (n_d, n_d)
    K_mirna = gip_3d_from_slices(np.transpose(A_train, (1, 0, 2)))  # (n_m, n_m)
    K_disease = gip_3d_from_slices(np.transpose(A_train, (2, 0, 1)))  # (n_s, n_s)

    return K_drug, K_mirna, K_disease

if __name__ == '__main__':
    args = parameters_set()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = [
        HGNN(
            BioEncoder_wo_BioAttributes(
                drug_num=DRUG_NUM,
                mirna_num=MIRNA_NUM,
                disease_num=DISEASE_NUM,
                output=256
            ),
            HgnnEncoder(256, 512),
            Decoder((512 // 2) * 3),
            out_channels=(512 // 2)
        ).to(device)
        for _ in range(5)
    ]

    optimizers = [torch.optim.Adam(m.parameters(), lr=5e-4, weight_decay=5e-5) for m in models]
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizers[i], mode='max', factor=0.5, patience=3, min_lr=1e-6, verbose=True
    ) for i in range(5)]

    # 6) 构建模型D:\shaoying\MCHNN-main\（测试中15)_3维+3维(没有分子图和对比学习)修改参数256-128
    results_dir = "D:/shaoying/MCHNN-main/（实验15)_256-128 - 副本/results2-10"
    import time, os
    csv_log_file = os.path.join(results_dir, "result_五折交叉验证结果.csv")
    csv_log_file1 = os.path.join(results_dir, "result_每折的结果.csv")
    os.makedirs(results_dir, exist_ok=True)
    with open(csv_log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Average AUC-PR", "avg_auc_roc", "avg_prec", "avg_f1", "avg_recall","kind","train_loss","val_loss"])
    with open(csv_log_file1, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Average AUC-PR", "avg_auc_roc", "avg_prec", "avg_f1", "avg_recall","kind","train_loss","val_loss","fold"])


    def log_to_csv(epoch, Average_AUC_PR, avg_auc_roc,avg_prec,avg_f1,avg_recall,kind,train_loss,val_loss):
        with open(csv_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, Average_AUC_PR, avg_auc_roc,avg_prec,avg_f1,avg_recall,kind,train_loss,val_loss])

    def log_to_csv1(epoch, Average_AUC_PR, avg_auc_roc,avg_prec,avg_f1,avg_recall,kind,train_loss,val_loss,fold):
        with open(csv_log_file1, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, Average_AUC_PR, avg_auc_roc,avg_prec,avg_f1,avg_recall,kind,train_loss,val_loss,fold])


    global my_loss, optimizer
    my_loss = torch.nn.BCELoss()

    # ) 加载正样本和四类负样本
    adj_data = np.loadtxt('./adj_del_4mic_myid.txt')  # 正样本 (n, 3)
    # 假设 adj_data 形状为 (N,4)：[drug, mirna, disease, label]
    adj_all = adj_data[:, :3].astype(int)
    DRUG_MAP, MIR_MAP, DIS_MAP = build_global_maps(adj_all)

    # （可选但推荐）把全量数据也先映射一次，保证后续一致
    adj_data = apply_maps(adj_data, DRUG_MAP, MIR_MAP, DIS_MAP)


    fixed_pos_samples = adj_data[:, :3].astype(int)  # 全部正样本


    # 检查数据维度并调整 r5v
    def ensure_3_columns(data):
        if data.ndim == 1:
            data = data.reshape(-1, 3)
        elif data.shape[1] > 3:
            data = data[:, :3]  # 只取前三列
        return data

    # 加载负样本，确保都是3列
    unrelated_neg = np.loadtxt('./mapping/negative_sample/unrelated_negative_triplets.txt', dtype=int)
    one_relation_neg = np.loadtxt('./mapping/negative_sample/one_relation_negative_triplets.txt', dtype=int)
    two_relations_neg = np.loadtxt('./mapping/negative_sample/two_relations_negative_triplets.txt', dtype=int)
    summary_neg = np.loadtxt('./mapping/negative_sample/negative_triplets_summary.txt', dtype=int)

    unrelated_neg = ensure_3_columns(unrelated_neg)
    one_relation_neg = ensure_3_columns(one_relation_neg)
    two_relations_neg = ensure_3_columns(two_relations_neg)
    summary_neg = ensure_3_columns(summary_neg)

    neg_samples_list = [unrelated_neg, one_relation_neg, two_relations_neg, summary_neg]

    folds_all = prepare_fixed_10fold_multineg(
        pos_samples=fixed_pos_samples,
        neg_samples_list=neg_samples_list,
        n_folds=5,
        seed=42
    )
    import torch
    import os
    import numpy as np

    THR_DRUG = 0.9
    THR_MIR = 0.9
    THR_DIS = 0.9

    TAU_DRUG = 0.2
    TAU_MIR = 0.2
    TAU_DIS = 0.3

    GROUP_W_SCALE = 0.6

    # =========================================================
    # === （1）预计算每折 × 每类负样本的动态特征与超图结构 ===
    # =========================================================
    cache_path = "precomputed_structures.pt"
    if os.path.exists(cache_path):
        print(f"\n✅ 已检测到缓存文件 '{cache_path}'，正在加载预计算结果...")
        precomputed_structures = torch.load(cache_path)
        print("✅ 加载完成！")
    else:
        print("\n⚙️ 未检测到缓存文件，开始首次预计算 (十折 × 四类负样本)...")

        precomputed_structures = {}
        for fold_idx in range(5):
            for neg_type_idx, neg_type_name in enumerate(['unrelated', 'one_relation', 'two_relations', 'summary']):
                print(f"\n=== 预计算 Fold {fold_idx + 1}/5, NegType: {neg_type_name} ===")

                # 提取该折的数据
                fold_data = folds_all[fold_idx][neg_type_idx]
                train_data = fold_data['train']
                train_data_global = apply_maps(train_data.copy().astype(int), DRUG_MAP, MIR_MAP, DIS_MAP)

                # === 1️⃣ 提取当折训练集中的正样本 ===
                train_pos_global = train_data_global[train_data_global[:, 3] == 1][:, :3].astype(int)

                # === 2️⃣ 动态计算 3D GIP 核 ===
                print("计算动态 3D GIP 核...")
                K_drug_3d_dynamic, K_mirna_3d_dynamic, K_disease_3d_dynamic = compute_dynamic_gip_kernels(
                    train_pos_global)



                # === 4️⃣ 更新药物/miRNA/疾病特征 ===
                print("更新节点特征...")
                batch_drug_fused = torch.from_numpy(K_drug_3d_dynamic).float().to(device)
                mic_input_fused = torch.from_numpy(K_mirna_3d_dynamic).float().to(device)
                dis_input_fused = torch.from_numpy(K_disease_3d_dynamic).float().to(device)

                # === ✅ 替换构建 group_edges 的输入 ===
                print("基于融合相似性矩阵生成超图群...")
                drug_groups, drug_w, c_d, n_d, s_d = threshold_hyperedges_from_similarity(
                    K_drug_3d_dynamic, thr=THR_DRUG, etype='drug',
                    weight_mode='softmax', tau=TAU_DRUG, add_subedges=False, fallback_min_k=1
                )
                mirn_groups, mirn_w, c_m, n_m, s_m = threshold_hyperedges_from_similarity(
                    K_mirna_3d_dynamic, thr=THR_MIR, etype='mirna',
                    weight_mode='softmax', tau=TAU_MIR, add_subedges=False, fallback_min_k=1
                )
                dise_groups, dise_w, c_z, n_z, s_z = threshold_hyperedges_from_similarity(
                    K_disease_3d_dynamic, thr=THR_DIS, etype='disease',
                    weight_mode='softmax', tau=TAU_DIS, add_subedges=False, fallback_min_k=1
                )

                drug_pack = {
                    "group_edges": drug_groups, "group_w": drug_w,
                    "center_idx": c_d, "neigh_idx": n_d, "neigh_sim": s_d
                }
                mirn_pack = {
                    "group_edges": mirn_groups, "group_w": mirn_w,
                    "center_idx": c_m, "neigh_idx": n_m, "neigh_sim": s_m
                }
                dise_pack = {
                    "group_edges": dise_groups, "group_w": dise_w,
                    "center_idx": c_z, "neigh_idx": n_z, "neigh_sim": s_z
                }

                # === 5️⃣ 构建正样本超图 ===
                print("构建正样本超图...")
                he_pos, w_pos, _ = merge_triple_and_homo_groups(
                    triples=train_pos_global,
                    drug_pack=drug_pack, mir_pack=mirn_pack, dis_pack=dise_pack,
                    add_subedges=False, triple_w=1.0, group_w_scale=GROUP_W_SCALE, sub_w_scale=0.0
                )
                edge_index_pos = to_hyperedge_index(he_pos)
                hypergraph_pos = (edge_index_pos, w_pos)

                # === 6️⃣ 构建负样本超图 ===
                print("构建负样本超图...")
                train_neg_global = train_data_global[train_data_global[:, 3] == 0][:, :3].astype(int)
                pos_set_fold = set(map(tuple, train_pos_global))
                neg_clean = np.array([tri for tri in map(tuple, train_neg_global) if tri not in pos_set_fold],
                                     dtype=int)
                if len(neg_clean) > len(train_pos_global):
                    rng = np.random.default_rng(seed=42)
                    sel = rng.choice(len(neg_clean), size=len(train_pos_global), replace=False)
                    neg_clean = neg_clean[sel]

                he_neg, w_neg, _ = merge_triple_and_homo_groups(
                    triples=neg_clean,
                    drug_pack=drug_pack, mir_pack=mirn_pack, dis_pack=dise_pack,
                    add_subedges=False, triple_w=1.0, group_w_scale=GROUP_W_SCALE, sub_w_scale=0.0
                )
                edge_index_neg = to_hyperedge_index(he_neg)
                hypergraph_neg_ls = [(edge_index_neg, w_neg)]

                # === 7️⃣ 存入缓存结构 ===
                precomputed_structures[(fold_idx, neg_type_idx)] = {
                    "batch_drug_fused": batch_drug_fused,
                    "mic_input_fused": mic_input_fused,
                    "dis_input_fused": dis_input_fused,
                    "hypergraph_pos": hypergraph_pos,
                    "hypergraph_neg_ls": hypergraph_neg_ls
                }

        # === 保存预计算结果 ===
        torch.save(precomputed_structures, cache_path)
        print(f"\n✅ 已完成所有预计算并保存到 '{cache_path}'")

    low_val_loss = 100
    best_ave_auc_epoch = 0
    best_avg_auc_pr_sum = 0
    # 8) 新的训练验证流程
    # 8) 新的训练验证流程
    best_auc_pr = 0
    best_experiment_info = {}
    patience = 0  # 忍耐度
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch} ===")
        val_loss = 0
        train_loss = 0
        # 每轮重新随机选择N_SAMPLES_PER_CLASS个正样本
        auc_pr_sum = np.zeros((5, 4), dtype=float)

        epoch_auc_roc = np.zeros((5, 4), dtype=float)
        epoch_f1 = np.zeros((5, 4), dtype=float)
        epoch_prec = np.zeros((5, 4), dtype=float)
        epoch_recall = np.zeros((5, 4), dtype=float)
        epoch_auc_scores = []
        # 十折交叉验证
        for fold_idx in range(5):
            model = models[fold_idx]
            optimizer = optimizers[fold_idx]
            model.train()
            optimizer.zero_grad()
            scheduler = schedulers[fold_idx]

            x = 0
            print(f"\n--- Fold {fold_idx + 1}/5 ---")
            for neg_type_idx, neg_type_name in enumerate(['unrelated', 'one_relation', 'two_relations', 'summary']):
                print(f"Negative type: {neg_type_name}")

                fold_data = folds_all[fold_idx][neg_type_idx]
                train_data = fold_data['train']
                val_data = fold_data['val']

                # 将数据转换为全局编号
                train_data_global = apply_maps(train_data.copy().astype(int), DRUG_MAP, MIR_MAP, DIS_MAP)
                val_data_global = apply_maps(val_data.copy().astype(int), DRUG_MAP, MIR_MAP, DIS_MAP)

                # ✅ 直接加载预计算的结构
                struct = precomputed_structures[(fold_idx, neg_type_idx)]

                batch_drug_fused = struct["batch_drug_fused"]
                mic_input_fused = struct["mic_input_fused"]
                dis_input_fused = struct["dis_input_fused"]
                hypergraph_pos = struct["hypergraph_pos"]
                hypergraph_neg_ls = struct["hypergraph_neg_ls"]

                train_loss1 = train1(
                    drug_fea=batch_drug_fused,
                    mic_fea=mic_input_fused,
                    dis_fea=dis_input_fused,
                    hg_pos=hypergraph_pos,

                    train_data=train_data_global,
                    epoch=epoch,
                    device=device
                )

                # 测试（直接传入整个验证集，不再分割）
                auc_roc, auc_pr, f1, rec_, prec_, loss, y_true, y_prob, graph_embed_pos = test1(
                    drug_fea=batch_drug_fused,
                    mic_fea=mic_input_fused,
                    dis_fea=dis_input_fused,
                    hg_pos=hypergraph_pos,

                    val_data=val_data_global,
                    device=device,
                    epoch=epoch
                )
                # 保存预测结果（自动创建文件）
                save_dir = os.path.join(results_dir, "npz_predictions")
                os.makedirs(save_dir, exist_ok=True)
                np.savez(
                    os.path.join(save_dir, f"fold{fold_idx + 1}_{neg_type_name}.npz"),
                    y_true=y_true,
                    y_prob=y_prob
                )
                # ================== ✅ 保存样本级特征用于 t-SNE ==================
                from sklearn.preprocessing import StandardScaler

                save_tsne_dir = "./tsne_data"
                os.makedirs(save_tsne_dir, exist_ok=True)

                # 1️⃣ 提取当前 fold 的输入特征（原始）
                drug_np = batch_drug_fused.cpu().numpy()
                mic_np = mic_input_fused.cpu().numpy()
                dis_np = dis_input_fused.cpu().numpy()

                # 2️⃣ 归一化
                scaler = StandardScaler()
                drug_np = scaler.fit_transform(drug_np)
                mic_np = scaler.fit_transform(mic_np)
                dis_np = scaler.fit_transform(dis_np)

                # 3️⃣ 获取样本索引
                d_idx = val_data[:, 0].astype(int)
                m_idx = val_data[:, 1].astype(int)
                z_idx = val_data[:, 2].astype(int)
                labels = val_data[:, 3].astype(int)

                # 4️⃣ 拼接“原始特征”（每个样本三元组的输入拼起来）
                orig_features = np.concatenate([
                    drug_np[d_idx],
                    mic_np[m_idx],
                    dis_np[z_idx]
                ], axis=1)

                # 5️⃣ 拼接“模型学习特征”（用 test1 返回的 graph_embed_pos）
                sample_embeddings = np.concatenate([
                    graph_embed_pos[d_idx].cpu().numpy(),
                    graph_embed_pos[m_idx].cpu().numpy(),
                    graph_embed_pos[z_idx].cpu().numpy()
                ], axis=1)
                print(f"[t-SNE保存] 原始特征形状: {orig_features.shape}")
                print(f"[t-SNE保存] 模型嵌入形状: {sample_embeddings.shape}")
                print(f"[t-SNE保存] 标签数量: {labels.shape}")

                # 6️⃣ 保存 t-SNE 可视化数据
                np.savez(
                    os.path.join(save_tsne_dir, f"fold{fold_idx + 1}_{neg_type_name}.npz"),
                    original_features=orig_features,
                    learned_embeddings=sample_embeddings,
                    labels=labels
                )

                log_to_csv1(epoch, auc_pr, auc_roc, prec_, f1, rec_, neg_type_name,
                            train_loss1.item(), loss.item(), fold_idx)
                train_loss += train_loss1.item()
                val_loss += loss.item()
                auc_pr_sum[fold_idx, x] = (auc_pr)
                epoch_f1[fold_idx, x] = (f1)
                epoch_auc_roc[fold_idx, x] = (auc_roc)
                epoch_recall[fold_idx, x] = (rec_)
                epoch_prec[fold_idx, x] = (prec_)

                x += 1
                # 更新最佳结果
                if auc_pr > best_auc_pr:
                    best_auc_pr = auc_pr
                    best_experiment_info = {
                        'epoch': epoch,
                        'fold': fold_idx + 1,
                        'neg_type': neg_type_name,
                        'auc_pr': auc_pr
                    }
        val_loss = val_loss / 20
        train_loss = train_loss / 20
        patience += 1
        if (val_loss < low_val_loss):
            low_val_loss = val_loss
            patience = 0
        # 输出本轮平均AUC-PR
        avg_f1 = epoch_f1.mean(axis=0)
        avg_auc_roc = epoch_auc_roc.mean(axis=0)
        avg_recall = epoch_recall.mean(axis=0)
        avg_auc_pr = auc_pr_sum.mean(axis=0)
        avg_prec = epoch_prec.mean(axis=0)

        print(f"\nEpoch {epoch}")
        import time, os

        os.makedirs(results_dir, exist_ok=True)
        epoch_log_path = os.path.join(results_dir, 'epoch_log.txt')
        for i in range(4):
            if (i == 0):
                kind = 'un'
            if (i == 1):
                kind = 'one'
            if (i == 2):
                kind = 'two'
            if (i == 3):
                kind = 'sum'
            log_to_csv(epoch, avg_auc_pr[i], avg_auc_roc[i], avg_prec[i], avg_f1[i], avg_recall[i], kind, train_loss,
                       val_loss)
        metric_for_lr = float(avg_auc_pr.mean())
        if (patience >= 50):
            print("耐心都度过高，退出循环")
            break
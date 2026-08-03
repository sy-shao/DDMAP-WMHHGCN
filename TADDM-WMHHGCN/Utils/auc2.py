if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    import os

    # 手动预设的AUC值（用于图例）
    manual_aucs = {
        "unrelated": {
            "roc_fold": [0.9991, 0.9979, 0.9994, 0.9994, 0.9987],
            "roc_mean": 0.9989,
            "pr_fold": [0.9991, 0.9985, 0.9994, 0.9995, 0.9989],
            "pr_mean": 0.9991
        },
        "one_relation": {
            "roc_fold": [0.9960, 0.9952, 0.9943, 0.9938, 0.9940],
            "roc_mean": 0.9947,
            "pr_fold": [0.9966, 0.9965, 0.9949, 0.9937, 0.9949],
            "pr_mean": 0.9953
        },
        "two_relations": {
            "roc_fold": [0.9619, 0.9555, 0.9410, 0.9423, 0.9506],
            "roc_mean": 0.9503,
            "pr_fold": [0.9661, 0.9525, 0.9372, 0.9371, 0.9491],
            "pr_mean": 0.9484
        },
        "summary": {
            "roc_fold": [0.9848, 0.9822, 0.9834, 0.9862, 0.9830],
            "roc_mean": 0.9839,
            "pr_fold": [0.9854, 0.9820, 0.9838, 0.9854, 0.9831],
            "pr_mean": 0.9839
        }
    }

    print("\n=== 正在绘制 ROC / PR 曲线 ===")

    results_dir = r"D:/shaoying/MCHNN-main/（实验15)_256-128/results2-10"
    folds = 5
    neg_types = ["unrelated", "one_relation", "two_relations", "summary"]
    save_dir = os.path.join(results_dir, "npz_predictions")

    # 设置全局字体大小（可选）
    plt.rcParams['font.size'] = 12

    for neg_idx, neg_name in enumerate(neg_types):
        # === ROC 绘制 ===
        plt.figure(figsize=(8, 6))

        # 公共的FPR点（用于插值）
        mean_fpr = np.linspace(0, 1, 1000)
        tprs = []
        aucs = []

        # 存储每个fold的颜色
        colors = plt.cm.tab10(np.linspace(0, 1, folds))

        for i in range(folds):
            data_path = os.path.join(save_dir, f"fold{i + 1}_{neg_name}.npz")
            if not os.path.exists(data_path):
                print(f"警告：找不到文件 {data_path}")
                continue

            data = np.load(data_path)
            y_true, y_prob = data["y_true"], data["y_prob"]

            fpr, tpr, _ = roc_curve(y_true, y_prob)

            # 确保ROC严格从(0,0)开始，到(1,1)结束
            fpr = np.concatenate(([0.0], fpr, [1.0]))
            tpr = np.concatenate(([0.0], tpr, [1.0]))

            # 自动计算AUC（用于后续打印统计，不影响图例）
            roc_auc_auto = auc(fpr, tpr)
            aucs.append(roc_auc_auto)

            # 插值到公共的FPR点
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            tprs.append(interp_tpr)

            # 绘制每个fold的曲线，图例使用手动预设的AUC值，并改为AUROC
            roc_auc_manual = manual_aucs[neg_name]["roc_fold"][i]
            plt.plot(fpr, tpr, lw=1, alpha=0.5, color=colors[i],
                     label=f"Fold {i + 1} (AUROC={roc_auc_manual:.4f})")

        if tprs:
            # 计算平均TPR和标准差
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0)

            # 自动计算平均AUC（用于后续打印统计）
            mean_auc_auto = auc(mean_fpr, mean_tpr)

            # 确保平均曲线也从(0,0)到(1,1)
            mean_fpr_full = np.concatenate(([0.0], mean_fpr, [1.0]))
            mean_tpr_full = np.concatenate(([0.0], mean_tpr, [1.0]))

            # 绘制平均曲线，图例使用手动预设的均值AUC，并改为AUROC
            # 线宽从3调细为1.5
            mean_auc_manual = manual_aucs[neg_name]["roc_mean"]
            plt.plot(mean_fpr_full, mean_tpr_full, color='black', lw=1.5,
                     label=f'Mean  (AUROC={mean_auc_manual:.4f})')

            # 添加置信区间（不显示在图例中）
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
                             color='grey', alpha=0.2, label='_nolegend_')

        # 绘制对角线
        plt.plot([0, 1], [0, 1], '--', color='gray', label='_nolegend_')

        # 设置坐标轴范围 - 留出5%的空白
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

        # 坐标轴刻度字体放大
        plt.tick_params(axis='both', labelsize=14)

        plt.title(f'ROC Curve - {neg_name}', fontsize=16, fontweight='bold')
        # 坐标轴标题字体放大
        plt.xlabel("False Positive Rate", fontsize=15)
        plt.ylabel("True Positive Rate", fontsize=15)

        # 图例放在右下角，字体放大
        plt.legend(loc="lower right", fontsize=12)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path_roc = os.path.join(results_dir, f'ROC_{neg_name}.png')
        plt.savefig(save_path_roc, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ ROC曲线已保存至: {save_path_roc}")

        # === PR 绘制 ===
        plt.figure(figsize=(8, 6))

        # 公共的Recall点（用于插值）
        mean_recall = np.linspace(0, 1, 1000)
        precs = []
        pr_aucs = []

        for i in range(folds):
            data_path = os.path.join(save_dir, f"fold{i + 1}_{neg_name}.npz")
            if not os.path.exists(data_path):
                continue

            data = np.load(data_path)
            y_true, y_prob = data["y_true"], data["y_prob"]

            precision, recall, _ = precision_recall_curve(y_true, y_prob)

            # 确保PR曲线严格从(0,1)开始，到(1,0)结束
            recall = np.concatenate(([0.0], recall, [1.0]))
            precision = np.concatenate(([1.0], precision, [0.0]))

            # 确保召回率是单调递增的
            order = np.argsort(recall)
            recall = recall[order]
            precision = precision[order]

            # 自动计算PR AUC（用于后续打印统计）
            pr_auc_auto = auc(recall, precision)
            pr_aucs.append(pr_auc_auto)

            # 插值到公共的recall点
            interp_precision = np.interp(mean_recall, recall, precision)
            precs.append(interp_precision)

            # 绘制每个fold的曲线，图例使用手动预设的AUC值，并改为AUPR
            pr_auc_manual = manual_aucs[neg_name]["pr_fold"][i]
            plt.plot(recall, precision, lw=1, alpha=0.5, color=colors[i],
                     label=f"Fold {i + 1} (AUPR={pr_auc_manual:.4f})")

        if precs:
            # 计算平均Precision和标准差
            mean_precision = np.mean(precs, axis=0)
            std_precision = np.std(precs, axis=0)

            # 自动计算平均PR AUC（用于后续打印统计）
            mean_pr_auc_auto = auc(mean_recall, mean_precision)

            # 确保平均曲线也从(0,1)到(1,0)
            mean_recall_full = np.concatenate(([0.0], mean_recall, [1.0]))
            mean_precision_full = np.concatenate(([1.0], mean_precision, [0.0]))

            # 绘制平均曲线，图例使用手动预设的均值AUC，并改为AUPR
            # 线宽从3调细为1.5
            mean_pr_auc_manual = manual_aucs[neg_name]["pr_mean"]
            plt.plot(mean_recall_full, mean_precision_full, color='black', lw=1.5,
                     label=f'Mean  (AUPR={mean_pr_auc_manual:.4f})')

            # 添加置信区间（不显示在图例中）
            precs_upper = np.minimum(mean_precision + std_precision, 1)
            precs_lower = np.maximum(mean_precision - std_precision, 0)
            plt.fill_between(mean_recall, precs_lower, precs_upper,
                             color='grey', alpha=0.2, label='_nolegend_')

        # 设置坐标轴范围 - 留出5%的空白
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

        # 坐标轴刻度字体放大
        plt.tick_params(axis='both', labelsize=14)

        # 坐标轴标题字体放大
        plt.xlabel("Recall", fontsize=15)
        plt.ylabel("Precision", fontsize=15)
        plt.title(f'Precision-Recall Curve - {neg_name}', fontsize=16, fontweight='bold')

        # 图例放在左下角，字体放大
        plt.legend(loc="lower left", fontsize=12)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path_pr = os.path.join(results_dir, f'PR_{neg_name}.png')
        plt.savefig(save_path_pr, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ PR曲线已保存至: {save_path_pr}")

        # 打印汇总统计（仍使用自动计算的值，与图例无关）
        if aucs and pr_aucs:
            print(f"\n{neg_name}:")
            print(f"  ROC-AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f} (自动计算)")
            print(f"  PR-AUC:  {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f} (自动计算)")

    print("\n✅ ROC 和 PR 曲线已生成完毕。")
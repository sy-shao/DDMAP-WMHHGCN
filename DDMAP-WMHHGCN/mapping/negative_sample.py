# import pandas as pd
#
# # 读取文件
# adj_del_4mic_myid = pd.read_csv('adj_del_4mic_myid.txt', sep='\t', header=None)
# drug_mapping = pd.read_csv('drug_mapping_file.csv')
# miRNA_mapping = pd.read_csv('miRNA_mapping_file.csv')
# disease_mapping = pd.read_csv('disease_mapping_file.csv')
# drug_disease_pairs = pd.read_csv('drug_disease_pairs.csv')
# disease_miRNA_pairs = pd.read_csv('diseaseID_miRNA_pairs.csv')
# drug_miRNA_pairs = pd.read_csv('DrugBankID_miRNA_pairs.csv')
#
# # 为了方便后续处理，设置数据框的列名
# adj_del_4mic_myid.columns = ['drug', 'miRNA', 'disease', 'value']
# drug_ids = drug_mapping.iloc[:, 0].values
# miRNA_ids = miRNA_mapping.iloc[:, 0].values
# disease_ids = disease_mapping.iloc[:, 0].values
#
# # 提取已知关系
# known_drug_disease_pairs = set(zip(drug_disease_pairs['DrugBankID'], drug_disease_pairs['DiseaseID']))
# known_disease_miRNA_pairs = set(zip(disease_miRNA_pairs['DiseaseID'], disease_miRNA_pairs['miRNA']))
# known_drug_miRNA_pairs = set(zip(drug_miRNA_pairs['DrugBankID'], drug_miRNA_pairs['miRNA']))
#
# # 定义负三元组列表
# negative_triplets_no_rel = []
# negative_triplets_one_rel = []
# negative_triplets_two_rels = []
# negative_triplets_all = []
#
# # 1. 完全没有关系的负三元组
# for drug in drug_ids:
#     for miRNA in miRNA_ids:
#         for disease in disease_ids:
#             if (drug, disease) not in known_drug_disease_pairs and \
#                (disease, miRNA) not in known_disease_miRNA_pairs and \
#                (drug, miRNA) not in known_drug_miRNA_pairs:
#                 negative_triplets_no_rel.append((drug, miRNA, disease))
#                 negative_triplets_all.append((drug, miRNA, disease))
#
# # 2. 只有一种关系的负三元组
# for drug in drug_ids:
#     for miRNA in miRNA_ids:
#         for disease in disease_ids:
#             if (drug, disease) in known_drug_disease_pairs and \
#                (disease, miRNA) not in known_disease_miRNA_pairs and \
#                (drug, miRNA) not in known_drug_miRNA_pairs:
#                 negative_triplets_one_rel.append((drug, miRNA, disease))
#                 negative_triplets_all.append((drug, miRNA, disease))
#
#             if (disease, miRNA) in known_disease_miRNA_pairs and \
#                (drug, disease) not in known_drug_disease_pairs and \
#                (drug, miRNA) not in known_drug_miRNA_pairs:
#                 negative_triplets_one_rel.append((drug, miRNA, disease))
#                 negative_triplets_all.append((drug, miRNA, disease))
#
#             if (drug, miRNA) in known_drug_miRNA_pairs and \
#                (disease, miRNA) not in known_disease_miRNA_pairs and \
#                (drug, disease) not in known_drug_disease_pairs:
#                 negative_triplets_one_rel.append((drug, miRNA, disease))
#                 negative_triplets_all.append((drug, miRNA, disease))
#
# # 3. 有两种关系的负三元组
# for drug in drug_ids:
#     for miRNA in miRNA_ids:
#         for disease in disease_ids:
#             if (drug, disease) in known_drug_disease_pairs and \
#                (drug, miRNA) in known_drug_miRNA_pairs and \
#                (disease, miRNA) not in known_disease_miRNA_pairs:
#                 negative_triplets_two_rels.append((drug, miRNA, disease))
#                 negative_triplets_all.append((drug, miRNA, disease))
#
#             if (drug, miRNA) in known_drug_miRNA_pairs and \
#                (disease, miRNA) in known_disease_miRNA_pairs and \
#                (drug, disease) not in known_drug_disease_pairs:
#                 negative_triplets_two_rels.append((drug, miRNA, disease))
#                 negative_triplets_all.append((drug, miRNA, disease))
#
#             if (disease, miRNA) in known_disease_miRNA_pairs and \
#                (drug, disease) in known_drug_disease_pairs and \
#                (drug, miRNA) not in known_drug_miRNA_pairs:
#                 negative_triplets_two_rels.append((drug, miRNA, disease))
#                 negative_triplets_all.append((drug, miRNA, disease))
#
# # 将负三元组转化为 DataFrame
# negative_triplets_no_rel_df = pd.DataFrame(negative_triplets_no_rel, columns=['drug', 'miRNA', 'disease'])
# negative_triplets_one_rel_df = pd.DataFrame(negative_triplets_one_rel, columns=['drug', 'miRNA', 'disease'])
# negative_triplets_two_rels_df = pd.DataFrame(negative_triplets_two_rels, columns=['drug', 'miRNA', 'disease'])
# negative_triplets_all_df = pd.DataFrame(negative_triplets_all, columns=['drug', 'miRNA', 'disease'])
#
# # 保存为文件
# negative_triplets_no_rel_df.to_csv('./negative_sample/negative_triplets_no_rel.csv', index=False)
# negative_triplets_one_rel_df.to_csv('./negative_sample/negative_triplets_one_rel.csv', index=False)
# negative_triplets_two_rels_df.to_csv('./negative_sample/negative_triplets_two_rels.csv', index=False)
# negative_triplets_all_df.to_csv('./negative_sample/negative_triplets_all.csv', index=False)
#
# print("文件已保存！")

import pandas as pd

# 读取映射文件
drug_mapping = pd.read_csv('drug_mapping_file.csv')
miRNA_mapping = pd.read_csv('miRNA_mapping_file.csv')
disease_mapping = pd.read_csv('disease_mapping_file.csv')

# 创建映射字典
drug_dict = dict(zip(drug_mapping['DrugBankID'], drug_mapping['DrugID']))
miRNA_dict = dict(zip(miRNA_mapping['miRNA'], miRNA_mapping['miRNAMappedID']))
disease_dict = dict(zip(disease_mapping['DiseaseID'], disease_mapping['DiseaseMappedID']))

# 读取关系文件
disease_miRNA_pairs = pd.read_csv('diseaseID_miRNA_pairs.csv')
drug_disease_pairs = pd.read_csv('drug_disease_pairs.csv')
drug_miRNA_pairs = pd.read_csv('DrugBankID_miRNA_pairs.csv')

# 创建用于快速查找的集合
disease_miRNA_set = set(zip(disease_miRNA_pairs['DiseaseID'], disease_miRNA_pairs['miRNA']))
drug_disease_set = set(zip(drug_disease_pairs['DrugBankID'], drug_disease_pairs['DiseaseID']))
drug_miRNA_set = set(zip(drug_miRNA_pairs['DrugBankID'], drug_miRNA_pairs['miRNA']))

# 读取正样本文件（adj_del_4mic_myid.txt）
positive_triplets = pd.read_csv('adj_del_4mic_myid.txt', sep='\t', header=None)
positive_triplets.columns = ['drug', 'miRNA', 'disease', 'label']

# 将正样本的三元组（drug, miRNA, disease）存入集合
positive_triplet_set = set(zip(positive_triplets['drug'], positive_triplets['miRNA'], positive_triplets['disease']))

# 创建负三元组的列表
unrelated_triplets = []
one_relation_triplets = []
two_relations_triplets = []

# 生成所有可能的三元组
drugs = list(drug_dict.keys())
miRNAs = list(miRNA_dict.keys())
diseases = list(disease_dict.keys())

# 检查三元组的关系
def check_relations(drug, miRNA, disease):
    relations = 0
    if (drug, disease) in drug_disease_set:
        relations += 1
    if (drug, miRNA) in drug_miRNA_set:
        relations += 1
    if (disease, miRNA) in disease_miRNA_set:
        relations += 1
    return relations

# 生成负三元组
for drug in drugs:
    for miRNA in miRNAs:
        for disease in diseases:
            if (drug, miRNA, disease) in positive_triplet_set:
                continue  # 跳过正样本

            relations = check_relations(drug, miRNA, disease)
            if relations == 0:
                unrelated_triplets.append((drug_dict[drug], miRNA_dict[miRNA], disease_dict[disease]))
            elif relations == 1:
                one_relation_triplets.append((drug_dict[drug], miRNA_dict[miRNA], disease_dict[disease]))
            elif relations == 2:
                two_relations_triplets.append((drug_dict[drug], miRNA_dict[miRNA], disease_dict[disease]))

# 写入单独的文件
def write_triplets(filename, triplets):
    with open(filename, 'w') as f:
        for triplet in triplets:
            f.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\t0\n")

# 写入各类负三元组文件
write_triplets('./negative_sample/unrelated_negative_triplets.txt', unrelated_triplets)
write_triplets('./negative_sample/one_relation_negative_triplets.txt', one_relation_triplets)
write_triplets('./negative_sample/two_relations_negative_triplets.txt', two_relations_triplets)

# 汇总所有负三元组
all_negative_triplets = unrelated_triplets + one_relation_triplets + two_relations_triplets

# 写入汇总文件
write_triplets('./negative_sample/negative_triplets_summary.txt', all_negative_triplets)

# 输出汇总文件成功
print("负三元组文件生成成功：unrelated_negative_triplets.txt, one_relation_negative_triplets.txt, two_relations_negative_triplets.txt, negative_triplets_summary.txt")

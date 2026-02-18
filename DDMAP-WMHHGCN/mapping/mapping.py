import pandas as pd

# 读取数据文件
triplets_file = 'drug_disease_miRNA_triplets.csv'
smiles_file = 'cleaned_drug_smiles_16.csv'

# 读取drug_disease_miRNA_triplets.csv和cleaned_drug_smiles_16.csv
triplets_df = pd.read_csv(triplets_file)
smiles_df = pd.read_csv(smiles_file)

# 生成药物、疾病和miRNA的映射
drug_mapping = {drug: idx for idx, drug in enumerate(triplets_df['DrugBankID'].unique())}
disease_mapping = {disease: idx for idx, disease in enumerate(triplets_df['DiseaseID'].unique())}
mirna_mapping = {mirna: idx for idx, mirna in enumerate(triplets_df['miRNA'].unique())}

# 将原始数据中的DrugBankID, DiseaseID, miRNA映射到新的编号
triplets_df['DrugID'] = triplets_df['DrugBankID'].map(drug_mapping)
triplets_df['DiseaseID'] = triplets_df['DiseaseID'].map(disease_mapping)
triplets_df['miRNAID'] = triplets_df['miRNA'].map(mirna_mapping)

# 生成adj_del_4mic_myid.txt格式的文件
adj_data = triplets_df[['DrugID', 'miRNAID','DiseaseID' ]].copy()
adj_data['label'] = 1  # 设置标签为1，表示有关系
adj_data = adj_data[['DrugID', 'miRNAID', 'DiseaseID', 'label']]

# 保存为文件
adj_output_file = 'adj_del_4mic_myid.txt'
adj_data.to_csv(adj_output_file, sep='\t', header=False, index=False)

# 生成映射文件
drug_mapping_data = pd.DataFrame(list(drug_mapping.items()), columns=['DrugBankID', 'DrugID'])
disease_mapping_data = pd.DataFrame(list(disease_mapping.items()), columns=['DiseaseID', 'DiseaseMappedID'])
mirna_mapping_data = pd.DataFrame(list(mirna_mapping.items()), columns=['miRNA', 'miRNAMappedID'])

# 保存映射文件
drug_mapping_file = 'drug_mapping_file.csv'
disease_mapping_file = 'disease_mapping_file.csv'
mirna_mapping_file = 'miRNA_mapping_file.csv'

drug_mapping_data.to_csv(drug_mapping_file, index=False)
disease_mapping_data.to_csv(disease_mapping_file, index=False)
mirna_mapping_data.to_csv(mirna_mapping_file, index=False)

# 打印输出路径
print(f"Generated files: \n{adj_output_file}\n{drug_mapping_file}\n{disease_mapping_file}\n{mirna_mapping_file}")

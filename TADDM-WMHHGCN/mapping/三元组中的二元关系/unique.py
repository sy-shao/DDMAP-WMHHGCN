import pandas as pd

# 1. 读取三元关系表格
df = pd.read_csv('drug_disease_miRNA_triplets.csv')

# 2. 提取并去重：Drug - Disease
drug_disease = df[['DrugBankID', 'DiseaseID']].drop_duplicates()

# 3. 提取并去重：Disease - miRNA
disease_miRNA = df[['DiseaseID', 'miRNA']].drop_duplicates()

# 4. 提取并去重：Drug - miRNA
drug_miRNA = df[['DrugBankID', 'miRNA']].drop_duplicates()

# 5. 保存为 CSV 文件
drug_disease.to_csv('drug_disease_pairs.csv', index=False)
disease_miRNA.to_csv('disease_miRNA_pairs.csv', index=False)
drug_miRNA.to_csv('drug_miRNA_pairs.csv', index=False)

# 6. 打印统计结果
print(f"{'关系类型':<20} | {'去重后的数量':<10}")
print("-" * 35)
print(f"{'Drug - Disease':<20} | {len(drug_disease):<10}")
print(f"{'Disease - miRNA':<20} | {len(disease_miRNA):<10}")
print(f"{'Drug - miRNA':<20} | {len(drug_miRNA):<10}")
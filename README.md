# DDMAP-WMHHGCN
A Weighted Multi-Head Hypergraph Convolutional Network for Drug–Disease–miRNA Association Prediction.
# 1. Description
This repository implements DDMAP-WMHHGCN, an end-to-end weighted hypergraph neural network for drug–disease–miRNA association prediction. The framework includes hypergraph construction, high-order representation learning, structured negative sampling, and triplet-level prediction and optimization.
# 2. Requirements
Python == 3.7
PyTorch == 1.8.0
torch-geometric == 1.7.2
torch-scatter == 2.0.8
torch-sparse == 0.6.12
torchvision == 0.9.0
torchaudio == 0.8.0
numpy == 1.20.2
scipy == 1.7.3
pandas == 1.2.5
scikit-learn == 1.0.2
# 3. How to use
After installing the required dependencies, you can directly run the main training and evaluation script:
python main_5cv.py
This script performs 5-fold cross-validation for drug–disease–miRNA association prediction under the predefined experimental settings.
# 4. Get result
After running main_5cv.py, the experimental results are automatically saved in the project directory.
Two result files will be generated:
result_五折交叉验证结果.csv
Contains the averaged performance metrics over five folds for each negative sampling setting (DN, 1LN, 2LN, and MN).
result_每折的结果.csv
Contains the detailed results for each individual fold and for all training epochs.
Both files record the performance of all training epochs. The final training epoch corresponds to the last row in the file. In our implementation, the best-performing model is selected from 50 epochs before the final training epoch. Therefore:
Best epoch = (Final epoch number − 50)
Users can locate the row corresponding to this epoch in the result file to obtain the reported best performance.

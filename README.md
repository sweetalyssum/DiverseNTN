# Modeling Document Novelty with Neural Tensor Network for Search Result Diversification

## Training data
* feature.txt and feature_test.txt: document/query featture vectors 
* idealfile: groundtruth ranking for each query 

## Command line
python DiverseNTN.py feature.txt feature_test.txt idealfile config.yml

## Results
The training results are saved in fold *reslut* 

## Reference
[1] Long Xia, Jun Xu, Yanyan Lan, Jiafeng Guo, Xueqi Cheng. Modeling Document Novelty with Neural Tensor Network for Search Result Diversification. In Proc. SIGIR 2016. 

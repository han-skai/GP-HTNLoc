## GP-HTNLoc

GP-HTNLoc is an ncRNA subcellular multilabel localization predictor based on graph prototype head-tail network. It adopts a model of separate training of head and tail classes to solve the problem of dataset class imbalance, in addition, it introduces a novel graph prototype module to construct heterogeneous graphs by utilizing the correlation information between ncRNA subcellular localization labels and samples, and learns rich structural and label correlation information from the heterogeneous graphs in order to enhance the classification ability in the case of lack of samples.

## Environmental requirements

```
python == 3.7.12
torch == 1.13.1 
dgl == 1.1.2 
numpy  == 1.21.6
pandas == 1.3.5
scikit-learn == 1.0.2
```

## Data

The benchmark dataset used in this study is from zhou et al. The link to access it is：https://github.com/guofei-tju/Identify-NcRNA-Sub-Loc-MKGHkNN
The independent dataset from Bai et al. Available from this link: https://bliulab.net/ncRNALocate-EL.

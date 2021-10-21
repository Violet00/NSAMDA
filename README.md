# NSAMDA
## A novel model for predicting miRNA-disease associations

The dataset used in this paper 
---------------------------------------
HMDD v2.0. Dataset summary:
- NumNodes: 878
- NumEdges: 21720
- Disease nodes: 383
- MiRNA nodes:  495

Requirements
--------------------------------
- PyTorch 0.4.1+
- dgl 0.6.0


How to run
--------------------------------
The parameter configuration information is in the `main.py`

**Please use `main.py`**


```python
python main.py
```




Performance
-------------------------

 AUC mean: 0.9369, variance: 0.0041 
 Accuracy mean: 0.8585, variance: 0.0095 
 Precision mean: 0.8495, variance: 0.0226 
 Recall mean: 0.8728, variance: 0.0191 
 F1-score mean: 0.8605, variance: 0.0077 
 PRC mean: 0.9337, variance: 0.0060


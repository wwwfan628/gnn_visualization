# GNN Visualization 
This is code implementation for semester project [Understanding and Visualizing Graph Neural Networks
](https://github.com/wwwfan628/gnn_visualization/blob/master/doc/Report.pdf).

## Usage
All commands should be executed within the `src/run` subfolder. The relevant configuration files for experiments are in `src/configs`. 
### 1) Fixed Point
- training GCN with joint loss function
```
python fixpoint.py --dataset cora --fixpoint_loss --exp_times 10
```
where `cora` is the dataset name and may be changed to `pubmed` or `citeseer`. If `--fixpoint_loss` is set `True`, then 
GCN is trained with proposed joint loss function, otherwise it's trained with normal entropy loss for classification.
`--exp_times` represent the repeating times of the experiments, the result shown in final report is the average of 10 experiments.

- to visualize the accuracy on 3 citation datasets, apply the above command for each dataset respectively and then 
head over to `notebooks/fixedpoint_visualization.ipynb`. Results taken from final report:

<div align=center><img width=55% height=55% src="https://github.com/wwwfan628/gnn_visualization/blob/master/doc/fixpoint.png"/></div>

<div align=center>
<table style="width:100%">
  <tr>
    <th>Dataset</th>
    <th>GCN</th>
    <th>SSE</th>
    <th>GCN with joint loss function</th>
  </tr>
  <tr>
    <td>Cora</td>
    <td>81.5</td>
    <td>79.0</td>
    <td>70.3</td>
  </tr>
  <tr>
    <td>PubMed</td>
    <td>81.2</td>
    <td>79.7</td>
    <td>69.0</td>
  </tr>
  <tr>
    <td>CiteSeer</td>
    <td>79.4</td>
    <td>75.8</td>
    <td>72.5</td>
  </tr>
  </table>
  </div>


### 2) Identifiability

- executing the experiment to check the node embedding identifiability 
```
python identifiability.py --dataset cora --knn 1 --repeat_times 5 --max_gcn_layers 10
```
where `--dataset` is used to determine the dataset in the experiment and can be chosen from `cora`,`pubmed` and `citeseer`.
`--knn` is used to set the k-nearest-neighbourhood search after recovering the input node features. `--repeat_times` represent
how many times the experiment will be repeated. `--max_gcn_layers` determine the maximal layers of GCN model used in the experiment.


- Results are visualized in the script `notebooks/identifiability_visualization.ipynb`. Example visualization results are shown below:

<img src="https://github.com/wwwfan628/gnn_visualization/blob/master/doc/id_cora.png" width=50% /><img src="https://github.com/wwwfan628/gnn_visualization/blob/master/doc/rr_cora.png" width=50% />
<img src="https://github.com/wwwfan628/gnn_visualization/blob/master/doc/acc_cora.png" width=50% /><img src="https://github.com/wwwfan628/gnn_visualization/blob/master/doc/acc_id_unid_cora.png" width=50% />


### 3) GNN-N

- to compute ![](http://latex.codecogs.com/gif.latex?\hat{P}^{MLP}_{i,features}), we need to execute experiments of 100-layer GCN
```
python gnn_n_100layerGCN.py --dataset cora --exp_times 10 --num_random_features 10
```
The parameter `--dataset` can be chosen from 7 node classification datasets, namely  `cora`, `pubmed`, `citeseer`, 
`amazon_photo`, `amazon_computers`, `coauthors_cs` and `coauthors_physics`. You can train 100-layer GCN several times and this 
is decided by `--exp_times`, while for each training trail the trained model is tested with 10 different random features
that can be changed by `--num_random_features`.


- to compute ![](http://latex.codecogs.com/gif.latex?\hat{P}_{i,graph\,structure}^{100\text{-}layer\,GCN}), we need to 
execute experiments of 3-layer MLP
```
python gnn_n_3layerMLP.py --dataset cora --exp_times 10
```
Similar as experiments of 100-layer GCN, `--dataset` can be chosen from 7 node classification datasets and `--exp_times`
determines how many times the experiment process will be repeated.

- computing GNN-N values
```
python gnn_n.py --dataset cora --mlp_exp_times 10 --gcn_exp_times 10 --`gcn_num_random_features` 10
```
In this step, experimental possibilities ![](http://latex.codecogs.com/gif.latex?\hat{P}^{MLP}_{i,features}) and 
![](http://latex.codecogs.com/gif.latex?\hat{P}_{i,graph\,structure}^{100\text{-}layer\,GCN}) are computed, and then GNN-N 
value is derived. `--mlp_exp_times` must be set the same as `--exp_times` used in the 3-layer MLP experiment. 
`--gcn_exp_times` and `--gcn_num_random_features` on the other hand must be set the same as `--exp_times` and 
`--num_random_features` used in experiments of 100-layer GCN respectively.


- After executing experiments and computing GNN-N values for all 7 datasets, you can visualize the results using 
`notebooks/gnn_n_3layerMLP_visualization.ipynb`. 
<div align=center><img width=55% height=55% src="https://github.com/wwwfan628/gnn_visualization/blob/master/doc/GNN-N.png"/></div> 


- The script `notebooks/gnn_n_3layerMLP_visualization.ipynb` is used to visualize results of 3-layer MLP experiments.
<div align=center><img width=55% height=55% src="https://github.com/wwwfan628/gnn_visualization/blob/master/doc/MLP_acc_rr.png"/></div> 

- Results of 100-layer GCN eperiments can be visualized in `notebooks/gnn_n_100layerGCN_visualization.ipynb`, for example
the accuracy with random features and original features and different repeating rates.
<div align=center><img width=55% height=55% src="https://github.com/wwwfan628/gnn_visualization/blob/master/doc/100GCN_acc_rr.png"/></div>
<div align=center><img width=55% height=55% src="https://github.com/wwwfan628/gnn_visualization/blob/master/doc/100layerGCN_TTRR.png"/></div>  


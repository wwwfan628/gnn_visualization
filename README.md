# GNN Visualization 
This is code implementation for semester project [Understanding and Visualizing Graph Neural Networks
](https://github.com/wwwfan628/gnn_visualization/blob/master/doc/Report.pdf).

## Usage
All commands should be executed within the `src/run` subfolder. The relevant configuration files for experiments are in `src/configs`. 
### 1) Fixed Point
- training GCN with joint loss function
```
python fixpoint.py cora --fixpoint_loss --exp_times 10
```
where `cora` is the dataset name and may be changed to `pubmed` or `citeseer`. If `--fixpoint_loss` is set `True`, then 
GCN is trained with proposed joint loss function, otherwise it's trained with normal entropy loss for classification.
`--exp_times` represent the repeating times of the experiments, the result shown in final report is the average of 10 experiments.

- to visualize the result, head over to `notebooks/fixedpoint_visulization.ipynb`. Results from final report is shown as following:

![]()

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
  


### 2) Identifiability

- executing the experiment to check the node embedding identifiability 
```
python identifiability.py --dataset cora --knn 1 --repeat_times 5 --max_gcn_layers 10
```
where `--dataset` is used to determine the dataset in the experiment and can be chosen from `cora`,`pubmed` and `citeseer`.
`--knn` is used to set the k-nearest-neighbourhood search after recovering the input node features. `--repeat_times` represent
how many times the experiment will be repeated. `--max_gcn_layers` determine the maximal layers of GCN model used in the experiment.


- to visualize the result, head over to `notebooks/identifiability_visulization.ipynb`. Example visualization result is shown below:
![]() 
![]() 
![]() 
![]() 




### 3) GNN-N


## Experimet Results

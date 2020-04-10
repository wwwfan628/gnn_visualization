# gnn_visualization
Semester Project: GNN Visualization
## Usage
All commands should be executed within the `src/run` subfolder. The relevant configuration files for training and optimization are in `src/configs/`. 
### 1) Training GCN Model and Find Fixpoint
```
python fixpoint.py cora graph_optimization --train
```
where `cora` is the dataset name, which may be changed to `reddit-self-loop`, `ppi` or `tu`. `graph_optimization` is the method used to find fixpoint and can also be chosen from `node_optimization` and `newton_method`. If checkpoint for model available, `--train` parameter here should be ignored.

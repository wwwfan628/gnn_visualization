# GNN Visualization
This is the code implementation of semester project "Understanding and Visualizing Graph Neural Networks
"[final report](https://github.com/wwwfan628/gnn_visualization/blob/master/doc/Report.pdf).
## Usage
All commands should be executed within the `src/run` subfolder. The relevant configuration files for training and optimization are in `src/configs/`. 
### 1) Training GCN Model and Find Fixpoint
```
python fixpoint.py cora graph_optimization --train
```
where `cora` is the dataset name, which may be changed to `reddit-self-loop`, `ppi`. `graph_optimization` is the method used to find fixpoint and can also be chosen from `node_optimization`, `newton_method` and `broyden_method`. If checkpoint for model available, `--train` should be ignored.

```
python fixpoint.py tu_aids broyden_method --train --fix_random
```
where `aids` is the TUDataset name, supported TUDatasets are listed in the following table. Prefix `tu_` should be added. `--fix_random` should be set as `True` if the result will be compared in `notebooks/compare_fixpoint.ipynb`

Supported TUDatasets:
<table style="width:100%">
  <tr>
    <th></th>
    <th>Type</th>
    <th>Num. of Graphs</th>
    <th>Num. of Classes</th>
    <th>Avg. Number of Nodes</th>
    <th>Avg. Number of Edges</th>
  </tr>
  <tr>
    <td>AIDS</td>
    <td>Disease</td>
    <td>2000</td>
    <td>2</td>
    <td>15.69</td>
    <td>16.20</td>
  </tr>
  <tr>
    <td>ENZYMES</td>
    <td>Molecular</td>
    <td>600</td>
    <td>6</td>
    <td>32.63</td>
    <td>62.14</td>
  </tr>
  <tr>
    <td>IMDB-BINARY</td>
    <td>Social</td>
    <td>1000</td>
    <td>2</td>
    <td>19.77</td>
    <td>96.53</td>
  </tr>
  <tr>
    <td>IMDB-MULTI</td>
    <td>Social</td>
    <td>1500</td>
    <td>3</td>
    <td>13.00</td>
    <td>65.94</td>
  </tr>
  <tr>
    <td>MSRC_9</td>
    <td>Vision</td>
    <td>211</td>
    <td>8</td>
    <td>40.58</td>
    <td>97.94</td>
  </tr>
  <tr>
    <td>MUTAG</td>
    <td>Molecular</td>
    <td>188</td>
    <td>2</td>
    <td>17.93</td>
    <td>19.79</td>
  </tr>
  <tr>
    <td>NCI1</td>
    <td>Molecular</td>
    <td>4110</td>
    <td>2</td>
    <td>29.87</td>
    <td>32.30</td>
  </tr>
  <tr>
    <td>PROTEINS</td>
    <td>Molecular</td>
    <td>1113</td>
    <td>2</td>
    <td>39.06</td>
    <td>72.82</td>
  </tr>
</table>

- To compare different fixpoint finding methods, head over to `notebooks/compare_cost_func.ipynb` and `notebooks/compare_fixpoint.ipynb`. To run `notebooks/compare_fixpoint.ipynb`, `--fix_random` must be set `True`. 

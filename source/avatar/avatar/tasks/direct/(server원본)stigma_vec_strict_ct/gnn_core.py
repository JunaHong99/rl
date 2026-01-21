
import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add

# ---------------------------------------------------------
# 1. 기본 MLP 블록 (RoboBallet Table S1)
# ---------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = hidden_dim
            
        layers = []
        # Input Layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden Layers
        for _ in range(num_layers - 1):
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        # Output Layer
        if hidden_dim != output_dim:
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------
# 2. GNN Update Functions
# ---------------------------------------------------------
class EdgeModel(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__()
        # Input: Source Node + Dest Node + Edge Attr + Global
        in_dim = node_dim * 2 + edge_dim + global_dim
        self.edge_mlp = MLP(in_dim, hidden_dim, num_layers=3, output_dim=256) #5

    def forward(self, src, dest, edge_attr, u, batch):
        # u[batch]를 통해 각 엣지가 속한 그래프의 글로벌 피쳐를 가져옴
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    def __init__(self, node_dim, edge_output_dim, global_dim, hidden_dim):
        super().__init__()
        # Input: Node + Aggregated Edge + Global
        in_dim = node_dim + edge_output_dim + global_dim
        self.node_mlp = MLP(in_dim, hidden_dim, num_layers=3, output_dim=512) #5

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        # Target 노드(col)로 들어오는 엣지 정보들을 합침 (Sum Aggregation)
        out = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp(out)

class GlobalModel(nn.Module):
    def __init__(self, node_output_dim, edge_output_dim, global_dim, hidden_dim):
        super().__init__()
        # Input: Aggregated Node + Aggregated Edge + Global
        in_dim = node_output_dim + edge_output_dim + global_dim
        self.global_mlp = MLP(in_dim, hidden_dim, num_layers=3, output_dim=512) #5

    def forward(self, x, edge_index, edge_attr, u, batch):
        # 노드 정보 평균 풀링
        node_aggr = scatter_mean(x, batch, dim=0)
        
        # 엣지 정보 평균 풀링 (엣지의 batch 인덱스는 source 노드 기준)
        edge_batch = batch[edge_index[0]]
        edge_aggr = scatter_mean(edge_attr, edge_batch, dim=0)
        
        out = torch.cat([node_aggr, edge_aggr, u], dim=1)
        return self.global_mlp(out)

# ---------------------------------------------------------
# 3. 메인 GNN 블록
# ---------------------------------------------------------
class RoboBalletGNNBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__()
        self.edge_model = EdgeModel(node_dim, edge_dim, global_dim, hidden_dim)
        self.node_model = NodeModel(node_dim, 256, global_dim, hidden_dim)
        self.global_model = GlobalModel(512, 256, global_dim, hidden_dim)
        
        # Output Dimensions (for reference)
        self.out_node_dim = 512
        self.out_global_dim = 512

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        
        # 1. Edge Update
        # row(source), col(dest)
        edge_attr = self.edge_model(x[row], x[col], edge_attr, u, batch[row])
        
        # 2. Node Update
        x = self.node_model(x, edge_index, edge_attr, u, batch)
        
        # 3. Global Update
        u = self.global_model(x, edge_index, edge_attr, u, batch)
        
        return x, edge_attr, u
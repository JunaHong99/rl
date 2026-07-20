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
        
        # Hidden Layers (Deep MLP)
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
# 2. GNN Update Functions (논문 사양 적용)
# ---------------------------------------------------------
class EdgeModel(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim=256):
        super().__init__()
        # Input: Source Node + Dest Node + Edge Attr + Global
        in_dim = node_dim * 2 + edge_dim + global_dim
        
        # [수정] Edge Update: MLP(256, 6)
        # hidden_dim을 256으로 고정, num_layers를 3 -> 6으로 증가
        self.edge_mlp = MLP(in_dim, 256, num_layers=6, output_dim=256) 

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    def __init__(self, node_dim, edge_output_dim, global_dim, hidden_dim=512):
        super().__init__()
        # Input: Node + Aggregated Edge + Global
        in_dim = node_dim + edge_output_dim + global_dim
        
        # [수정] Node Update: MLP(512, 7)
        # hidden_dim을 512로 고정, num_layers를 3 -> 7으로 증가
        self.node_mlp = MLP(in_dim, 512, num_layers=7, output_dim=512) 

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        # Target 노드(col)로 들어오는 엣지 정보들을 합침 (Sum Aggregation)
        out = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp(out)

class GlobalModel(nn.Module):
    def __init__(self, node_output_dim, edge_output_dim, global_dim, hidden_dim=512):
        super().__init__()
        # Input: Aggregated Node + Aggregated Edge + Global
        in_dim = node_output_dim + edge_output_dim + global_dim
        
        # [수정] Global Update: MLP(512, 7)
        # hidden_dim을 512로 고정, num_layers를 3 -> 7으로 증가
        self.global_mlp = MLP(in_dim, 512, num_layers=7, output_dim=512) 

    def forward(self, x, edge_index, edge_attr, u, batch):
        # 노드 정보 평균 풀링
        node_aggr = scatter_mean(x, batch, dim=0)
        
        # 엣지 정보 평균 풀링
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
        
        # [수정] 각 모델의 Output Dimension을 논문 사양(Edge=256, Node=512, Global=512)에 맞게 연결
        
        # 1. Edge Model (Output: 256)
        self.edge_model = EdgeModel(node_dim, edge_dim, global_dim, hidden_dim=256)
        
        # 2. Node Model (Input Edge Dim: 256 -> Output: 512)
        self.node_model = NodeModel(node_dim, 256, global_dim, hidden_dim=512)
        
        # 3. Global Model (Input Node: 512, Input Edge: 256 -> Output: 512)
        self.global_model = GlobalModel(512, 256, global_dim, hidden_dim=512)
        
        # Output Dimensions (참조용)
        self.out_node_dim = 512
        self.out_global_dim = 512
        self.out_edge_dim = 256

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        
        # 1. Edge Update
        edge_attr = self.edge_model(x[row], x[col], edge_attr, u, batch[row])
        
        # 2. Node Update
        x = self.node_model(x, edge_index, edge_attr, u, batch)
        
        # 3. Global Update
        u = self.global_model(x, edge_index, edge_attr, u, batch)
        
        return x, edge_attr, u
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
            
        # 2026-06-17: LayerNorm 제거 (GNN plateau 디버깅). LayerNorm이 feature를 샘플 내 정규화 →
        # goal-error 절대 크기/방향 신호 뭉갬 → 정책이 학습 못 함. 작동하는 mlp_policy도 LayerNorm 없음.
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        if hidden_dim != output_dim:
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
        
        # 2026-06-17 축소: RoboBallet 원본(256×6)은 50M params로 SAC 학습 불가(plateau).
        # MLP 베이스라인(385K) 급으로 축소: Edge 64×2.
        self.edge_mlp = MLP(in_dim, 64, num_layers=2, output_dim=64)

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return edge_attr + self.edge_mlp(out)   # residual (신호 보존)

class NodeModel(nn.Module):
    def __init__(self, node_dim, edge_output_dim, global_dim, hidden_dim=512):
        super().__init__()
        # Input: Node + Aggregated Edge + Global
        in_dim = node_dim + edge_output_dim + global_dim
        
        # 2026-06-17 축소: Node 512×7 → 128×2.
        self.node_mlp = MLP(in_dim, 128, num_layers=2, output_dim=128)

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        # Target 노드(col)로 들어오는 엣지 정보들을 합침 (Sum Aggregation)
        out = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return x + self.node_mlp(out)           # residual (rod pos_err 등 신호 보존)

class GlobalModel(nn.Module):
    def __init__(self, node_output_dim, edge_output_dim, global_dim, hidden_dim=512):
        super().__init__()
        # Input: Aggregated Node + Aggregated Edge + Global
        in_dim = node_output_dim + edge_output_dim + global_dim
        
        # 2026-06-17 축소: Global 512×7 → 128×2.
        self.global_mlp = MLP(in_dim, 128, num_layers=2, output_dim=128)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # 노드 정보 평균 풀링
        node_aggr = scatter_mean(x, batch, dim=0)
        
        # 엣지 정보 평균 풀링
        edge_batch = batch[edge_index[0]]
        edge_aggr = scatter_mean(edge_attr, edge_batch, dim=0)
        
        out = torch.cat([node_aggr, edge_aggr, u], dim=1)
        return u + self.global_mlp(out)         # residual

# ---------------------------------------------------------
# 3. 메인 GNN 블록
# ---------------------------------------------------------
class RoboBalletGNNBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__()
        
        # 2026-06-17 축소: Edge=64, Node=128, Global=128 (원본 256/512/512 → 50M params 회피).
        self.edge_model = EdgeModel(node_dim, edge_dim, global_dim)           # → 64
        self.node_model = NodeModel(node_dim, 64, global_dim)                 # edge 64 → 128
        self.global_model = GlobalModel(128, 64, global_dim)                  # node 128, edge 64 → 128

        # Output Dimensions (참조용)
        self.out_node_dim = 128
        self.out_global_dim = 128
        self.out_edge_dim = 64

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        
        # 1. Edge Update
        edge_attr = self.edge_model(x[row], x[col], edge_attr, u, batch[row])
        
        # 2. Node Update
        x = self.node_model(x, edge_index, edge_attr, u, batch)
        
        # 3. Global Update
        u = self.global_model(x, edge_index, edge_attr, u, batch)
        
        return x, edge_attr, u
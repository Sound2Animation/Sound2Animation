"""Mesh Encoder - Graph Neural Network"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np


class MeshGNN(nn.Module):
    def __init__(self, d_model: int = 256, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(3, hidden_dim)  # vertex position

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, vertices: torch.Tensor, faces: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            vertices: [N_total, 3] all vertices in batch
            faces: [3, E_total] edge indices from faces
            batch: [N_total] batch assignment for each vertex
        Returns:
            [B, d_model] mesh features
        """
        x = self.input_proj(vertices)

        for conv in self.convs:
            x = conv(x, faces)
            x = torch.relu(x)

        # Global pooling per graph
        if batch is None:
            batch = torch.zeros(vertices.shape[0], dtype=torch.long, device=vertices.device)

        x = global_mean_pool(x, batch)  # [B, hidden_dim]
        x = self.output_proj(x)
        return self.norm(x)  # [B, d_model]

    @staticmethod
    def mesh_to_graph(vertices: np.ndarray, faces: np.ndarray, device: str = "cuda") -> Data:
        """Convert mesh to PyG graph data"""
        # Create edges from faces (each face has 3 edges)
        edges = []
        for face in faces:
            edges.extend([
                [face[0], face[1]], [face[1], face[0]],
                [face[1], face[2]], [face[2], face[1]],
                [face[2], face[0]], [face[0], face[2]],
            ])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Normalize vertices to unit sphere
        verts = vertices - vertices.mean(axis=0)
        scale = np.abs(verts).max()
        if scale > 0:
            verts = verts / scale

        x = torch.tensor(verts, dtype=torch.float32)
        return Data(x=x, edge_index=edge_index).to(device)

    @staticmethod
    def batch_graphs(graphs: list[Data]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch multiple graphs"""
        batch = Batch.from_data_list(graphs)
        return batch.x, batch.edge_index, batch.batch

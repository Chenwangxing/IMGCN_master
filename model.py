import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SelfAttention(nn.Module):
    def __init__(self, in_dims=2, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()
        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.scaled_factor = torch.sqrt(torch.Tensor([d_model])).cuda()
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
    def split_heads(self, x):
        # x [batch_size seq_len d_model]
        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()
        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]
    def forward(self, x, mask=False, multi_head=False):
        # batch_size seq_len 2
        assert len(x.shape) == 3
        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model
        if multi_head:
            query = self.split_heads(query)  # B num_heads seq_len d_model
            key = self.split_heads(key)  # B num_heads seq_len d_model
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        else:
            attention = torch.matmul(query, key.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)
        attention = self.softmax(attention / self.scaled_factor)
        if mask is True:
            mask = torch.ones_like(attention)
            attention = attention * torch.tril(mask)
        return attention


class SparseWeightedAdjacency(nn.Module):
    def __init__(self, spa_in_dims=2, tem_in_dims=3, embedding_dims=64, dropout=0,):
        super(SparseWeightedAdjacency, self).__init__()
        # dense interaction
        self.spatial_attention = SelfAttention(spa_in_dims, embedding_dims)
        self.temporal_attention = SelfAttention(tem_in_dims, embedding_dims)
        self.spatial_interaction_output = nn.Sigmoid()
        self.temporal_interaction_output = nn.Sigmoid()
        self.dropout = dropout
        self.spa_softmax = nn.Softmax(dim=-1)
        self.tem_softmax = nn.Softmax(dim=-1)
    def forward(self, graph, identity, obs_traj):
        assert len(graph.shape) == 3
        spatial_graph = graph[:, :, 1:]  # (T N 2)
        temporal_graph = graph.permute(1, 0, 2)  # (N T 3)
        # obs_traj [N, 2, T]-->[T, N, 2]

        spatial_similarity = torch.zeros(spatial_graph.shape[0], spatial_graph.shape[1], spatial_graph.shape[1])    # (T N N)
        temporal_similarity = torch.zeros(spatial_graph.shape[1], spatial_graph.shape[0], spatial_graph.shape[0])    # (N T T)

        ### View-Distance Mask Modeule ###
        ## 1-Distance ## spatial distance threshold = 5
        dis_graph = obs_traj.unsqueeze(0) - obs_traj.unsqueeze(1)
        dis_graph = dis_graph.permute(3, 0, 1, 2)
        distance = torch.norm(dis_graph, dim=3)
        spatial_distance = torch.where(distance < 5, 1, 0)  # spatial distance threshold = 5
        ## 2-View ## Calculate the angle
        for i in range(spatial_graph.shape[0]):
            spatial_visual = spatial_graph[i, :, :]
            dis_visual = dis_graph[i, :, :, :]
            dis_similarity = torch.cosine_similarity(spatial_visual.unsqueeze(1), dis_visual, dim=-1)
            spatial_similarity[i, :, :] = dis_similarity  # spatial_similarity (T N N)
        spatial_similarity = spatial_similarity + torch.eye(spatial_graph.shape[1]).unsqueeze(0)
        spatial_similarity = spatial_similarity.to(device)
        spatial_degrees = torch.acos(spatial_similarity.clamp(-1, 1))
        angles_degrees = spatial_degrees * 180.0 / math.pi
        ## 3-Distance * View ## spatial view threshold = 10
        spatial_similarity = angles_degrees * spatial_distance
        spatial_similarity = torch.where(spatial_similarity.unsqueeze(1) < 10, 1, 0)   # spatial view threshold = 10
        spatial_similarity = spatial_similarity.repeat(1, 4, 1, 1)
        spatial_similarity = spatial_similarity + identity[0].unsqueeze(1)
        #################################

        ### Motion Offset Mask Modeule ###
        temporal_Sgraph = spatial_graph.permute(1, 0, 2)
        for i in range(temporal_Sgraph.shape[0]):
            temporal_visual = temporal_Sgraph[i, 1:, :]
            similarity = torch.cosine_similarity(temporal_visual.unsqueeze(1), temporal_visual.unsqueeze(0), dim=-1)
            temporal_degrees = torch.acos(similarity.clamp(-1, 1))
            temporal_degrees = temporal_degrees * 180.0 / math.pi
            similarity = torch.where(temporal_degrees > 10, 1, 0)      # motion offset threshold = 10
            temporal_similarity[i, 1:, 1:] = similarity
        # spatial_similarity (N T T)
        temporal_similarity = temporal_similarity.unsqueeze(1).to(device)
        temporal_similarity = temporal_similarity.repeat(1, 4, 1, 1)
        temporal_similarity = temporal_similarity + identity[1].unsqueeze(1)
        #################################

        # (T num_heads N N)   (T N d_model)
        dense_spatial_interaction = self.spatial_attention(spatial_graph, multi_head=True)
        # (N num_heads T T)   (N T d_model)
        dense_temporal_interaction = self.temporal_attention(temporal_graph, multi_head=True)

        normalized_spatial_adjacency_matrix = self.spa_softmax(dense_spatial_interaction * spatial_similarity)
        normalized_temporal_adjacency_matrix = self.tem_softmax(dense_temporal_interaction * temporal_similarity)

        return normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix


class GraphConvolution(nn.Module):
    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()
        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()
        self.dropout = dropout
    def forward(self, graph, adjacency):
        # graph [batch_size 1 seq_len 2]
        # adjacency [batch_size num_heads seq_len seq_len]
        gcn_features = self.embedding(torch.matmul(adjacency, graph))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout)
        return gcn_features  # [batch_size num_heads seq_len hidden_size]


class SparseGraphConvolution(nn.Module):
    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()
        self.dropout = dropout
        self.spatial_temporal_sparse_gcn = nn.ModuleList()
        self.temporal_spatial_sparse_gcn = nn.ModuleList()
        self.spatial_temporal_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.temporal_spatial_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
    def forward(self, graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix):
        # graph [1 seq_len num_pedestrians  3]
        # _matrix [batch num_heads seq_len seq_len]
        graph = graph[:, :, :, 1:]
        spa_graph = graph.permute(1, 0, 2, 3)  # (seq_len 1 num_p 2)
        tem_graph = spa_graph.permute(2, 1, 0, 3)  # (num_p 1 seq_len 2)

        gcn_spatial_features = self.spatial_temporal_sparse_gcn[0](spa_graph, normalized_spatial_adjacency_matrix)
        gcn_spatial_features = gcn_spatial_features.permute(2, 1, 0, 3)
        gcn_temporal_features = self.temporal_spatial_sparse_gcn[0](tem_graph, normalized_temporal_adjacency_matrix)

        return gcn_spatial_features, gcn_temporal_features


class TrajectoryModel(nn.Module):
    def __init__(self, embedding_dims=64, number_gcn_layers=1, dropout=0, obs_len=8,
                 pred_len=12, n_tcn=5, out_dims=5, num_heads=4):
        super(TrajectoryModel, self).__init__()
        self.number_gcn_layers = number_gcn_layers
        self.n_tcn = n_tcn
        self.dropout = dropout
        # sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency()
        # graph convolution
        self.stsgcn = SparseGraphConvolution(in_dims=2, embedding_dims=embedding_dims // num_heads, dropout=dropout)

        self.fusion_s = nn.Conv2d(num_heads, num_heads, kernel_size=1, bias=False)
        self.fusion_t = nn.Conv2d(num_heads, num_heads, kernel_size=1, bias=False)

        self.tcns = nn.ModuleList()
        self.tcns.append(nn.Sequential(nn.Conv2d(obs_len, pred_len, 3, padding=1),
            nn.PReLU()))

        for j in range(1, self.n_tcn):
            self.tcns.append(nn.Sequential(nn.Conv2d(pred_len, pred_len, 3, padding=1),
                nn.PReLU()))

        self.output = nn.Linear(embedding_dims // num_heads, out_dims)

    def forward(self, graph, identity, obs_traj):
        # graph 1 obs_len N 3
        # obs_traj 1 obs_len N 2

        normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix = \
            self.sparse_weighted_adjacency_matrices(graph.squeeze(), identity, obs_traj.squeeze())

        gcn_spatial_features, gcn_temporal_features = self.stsgcn(
            graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix
        )

        gcn_representation = self.fusion_s(gcn_spatial_features) + self.fusion_t(gcn_temporal_features)

        gcn_representation = gcn_representation.permute(0, 2, 1, 3)

        features = self.tcns[0](gcn_representation)

        for k in range(1, self.n_tcn):
            features = F.dropout(self.tcns[k](features) + features, p=self.dropout)

        prediction = torch.mean(self.output(features), dim=-2)

        return prediction.permute(1, 0, 2).contiguous()

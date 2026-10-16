"""TS-SGNet core model, with primary settings disclosed in the manuscript."""
from typing import Dict, Tuple
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import softmax as pyg_softmax

TRANSFORMER_D_MODEL = 48
TRANSFORMER_NHEAD = 4
TRANSFORMER_LAYERS = 1
TRANSFORMER_FF_DIM = 96
PROJ_DIM = 32
COORD_VAE_HIDDEN_DIM = 128
SSN_SCALES = (100, 400, 800)
GAT_PRIOR_BETA = 0.80
CLASSIFIER_FUSION_DIM = 128
DROPOUT = 0.20
LEARNING_RATE = 2.0e-4
WEIGHT_DECAY = 3.0e-6
FOCAL_ALPHA = 0.50
FOCAL_GAMMA = 1.00

def finite_tensor(x: torch.Tensor, nan: float = 0.0, pos: float = 1e4, neg: float = -1e4) -> torch.Tensor:
    return torch.nan_to_num(x, nan=nan, posinf=pos, neginf=neg)

class FeatureTokenTransformer(nn.Module):

    def __init__(
        self,
        feat_dim: int,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_LAYERS,
        ff_dim: int = TRANSFORMER_FF_DIM,
        dropout_rate: float = DROPOUT,
        chunk_size: int = 512,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.seq_len = self.feat_dim
        self.d_model = int(d_model)
        self.chunk_size = int(max(1, chunk_size))

        self.scalar_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.token_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def _encode_chunk(self, scalars: torch.Tensor, inject_noise: bool) -> torch.Tensor:
        tokens = self.scalar_proj(scalars.unsqueeze(-1))
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.token_embed[:, : tokens.size(1), :]
        if inject_noise and self.training:
            tokens = tokens + torch.randn_like(tokens) * 0.01
        if self.training:
            h = checkpoint(self.encoder, tokens, use_reentrant=False)
        else:
            h = self.encoder(tokens)
        pooled = self.out_norm(h[:, 0, :])
        return finite_tensor(pooled, nan=0.0, pos=10.0, neg=-10.0)

    def forward(self, x_feat: torch.Tensor, inject_noise: bool = False) -> torch.Tensor:
        x_feat = finite_tensor(x_feat, nan=0.0, pos=8.0, neg=-8.0).float()
        outs = []
        for start in range(0, x_feat.size(0), self.chunk_size):
            end = min(start + self.chunk_size, x_feat.size(0))
            outs.append(self._encode_chunk(x_feat[start:end], inject_noise=inject_noise))
        return torch.cat(outs, dim=0)


class CoordCVAE(nn.Module):

    def __init__(
        self,
        feat_dim: int,
        d_model: int = TRANSFORMER_D_MODEL,
        coord_dim: int = 2,
        hidden: int = 128,
        latent: int = PROJ_DIM,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.transformer_encoder = FeatureTokenTransformer(
            feat_dim=feat_dim,
            d_model=d_model,
            nhead=TRANSFORMER_NHEAD,
            num_layers=TRANSFORMER_LAYERS,
            ff_dim=TRANSFORMER_FF_DIM,
            dropout_rate=DROPOUT,
            chunk_size=512,
        )
        self.encoder_mlp = nn.Sequential(
            nn.Linear(d_model + coord_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.fc_u = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent + coord_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def reparameterize(self, u: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return u + eps * std
        return u

    def forward(self, x_feat: torch.Tensor, coords: torch.Tensor, inject_noise: bool = False) -> Dict[str, torch.Tensor]:
        x_feat = finite_tensor(x_feat, nan=0.0, pos=8.0, neg=-8.0).float()
        coords = finite_tensor(coords, nan=0.0, pos=1.0, neg=0.0).float().clamp(0.0, 1.0)

        transformer_feat = self.transformer_encoder(x_feat, inject_noise=inject_noise)
        transformer_feat = finite_tensor(transformer_feat, nan=0.0, pos=10.0, neg=-10.0)

        enc_in = torch.cat([transformer_feat, coords], dim=1)
        h = self.encoder_mlp(enc_in)
        h = finite_tensor(h, nan=0.0, pos=20.0, neg=-20.0)

        u = finite_tensor(self.fc_u(h), nan=0.0, pos=10.0, neg=-10.0)
        logvar = finite_tensor(self.fc_logvar(h), nan=0.0, pos=6.0, neg=-6.0).clamp(-6.0, 6.0)
        z = finite_tensor(self.reparameterize(u, logvar), nan=0.0, pos=10.0, neg=-10.0)
        recon = finite_tensor(self.decoder(torch.cat([z, coords], dim=1)), nan=0.0, pos=10.0, neg=-10.0)

        return {
            "u": u,
            "mu": u,
            "logvar": logvar,
            "z": z,
            "recon": recon,
            "transformer_feat": transformer_feat,
            "cvae_encoder_input": enc_in,
        }


class PaperSSNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_sp: int,
        grid_res: int,
        ssn_iters: int = 3,
        local_k: int = 9,
        feat_scale: float = 1.0,
        coord_scale: float = 0.35,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_sp = n_sp
        self.grid_res = grid_res
        self.ssn_iters = ssn_iters
        self.local_k = local_k
        self.feat_scale = feat_scale
        self.coord_scale = coord_scale

        self.rows, self.cols = self._infer_grid_shape(n_sp)

        self.center_feat_offset = nn.Parameter(torch.zeros(n_sp, in_channels))
        self.center_coord_offset = nn.Parameter(torch.zeros(n_sp, 2))

        nn.init.normal_(self.center_feat_offset, std=0.02)
        nn.init.normal_(self.center_coord_offset, std=0.01)

    @staticmethod
    def _infer_grid_shape(n_sp: int) -> Tuple[int, int]:
        cols = int(round(math.sqrt(n_sp)))
        cols = max(cols, 1)

        rows = int(math.ceil(n_sp / cols))
        rows = max(rows, 1)

        return rows, cols

    def _regular_grid_init(
        self,
        feat: torch.Tensor,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = finite_tensor(feat, nan=0.0, pos=10.0, neg=-10.0)

        coords = finite_tensor(
            coords,
            nan=0.0,
            pos=1.0,
            neg=0.0,
        ).clamp(0.0, 1.0)

        device = feat.device
        n, c = feat.shape

        x = coords[:, 0]
        y = coords[:, 1]

        col_idx = torch.clamp((x * self.cols).long(), 0, self.cols - 1)
        row_idx = torch.clamp((y * self.rows).long(), 0, self.rows - 1)

        cell_idx = row_idx * self.cols + col_idx
        cell_idx = torch.clamp(cell_idx, 0, self.n_sp - 1)

        sp_feat = torch.zeros(self.n_sp, c, device=device)
        sp_coord = torch.zeros(self.n_sp, 2, device=device)
        counts = torch.zeros(self.n_sp, 1, device=device)

        ones = torch.ones(n, 1, device=device)

        sp_feat.index_add_(0, cell_idx, feat)
        sp_coord.index_add_(0, cell_idx, coords)
        counts.index_add_(0, cell_idx, ones)

        valid = counts.squeeze(1) > 0

        sp_feat[valid] = sp_feat[valid] / counts[valid]
        sp_coord[valid] = sp_coord[valid] / counts[valid]

        if (~valid).any():
            sp_feat[~valid] = feat.mean(dim=0, keepdim=True)
            sp_coord[~valid] = coords.mean(dim=0, keepdim=True)

        sp_feat = sp_feat + self.center_feat_offset
        sp_coord = sp_coord + self.center_coord_offset
        sp_coord = sp_coord.clamp(0.0, 1.0)

        return sp_feat, sp_coord

    def _local_candidate_mask(
        self,
        coords: torch.Tensor,
        sp_coord: torch.Tensor,
    ) -> torch.Tensor:
        n = coords.size(0)
        m = sp_coord.size(0)
        k = min(self.local_k, m)

        dist_xy = torch.cdist(coords, sp_coord, p=2)

        nn_idx = torch.topk(
            dist_xy,
            k=k,
            largest=False,
            dim=1,
        ).indices

        mask = torch.zeros(n, m, device=coords.device, dtype=torch.bool)

        row_idx = torch.arange(
            n,
            device=coords.device,
        ).unsqueeze(1).expand(-1, k)

        mask[row_idx, nn_idx] = True

        return mask

    def _assignment(
        self,
        feat: torch.Tensor,
        coords: torch.Tensor,
        sp_feat: torch.Tensor,
        sp_coord: torch.Tensor,
    ) -> torch.Tensor:

        feat = finite_tensor(feat, nan=0.0, pos=10.0, neg=-10.0)
        coords = finite_tensor(coords, nan=0.0, pos=1.0, neg=0.0).clamp(0.0, 1.0)
        sp_feat = finite_tensor(sp_feat, nan=0.0, pos=10.0, neg=-10.0)
        sp_coord = finite_tensor(sp_coord, nan=0.0, pos=1.0, neg=0.0).clamp(0.0, 1.0)

        n = feat.size(0)
        m = sp_feat.size(0)
        chunk_size = int(max(256, globals().get("SSN_ASSIGN_CHUNK_SIZE", 2048)))
        k = min(int(self.local_k), m)

        with torch.no_grad():
            q = torch.empty((n, m), device=feat.device, dtype=feat.dtype)
            feat_scaled = feat.detach() * self.feat_scale
            coord_scaled = coords.detach() * self.coord_scale
            sp_feat_scaled = sp_feat.detach() * self.feat_scale
            sp_coord_scaled = sp_coord.detach() * self.coord_scale

            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                feat_dist = torch.cdist(feat_scaled[start:end], sp_feat_scaled, p=2) ** 2
                coord_dist = torch.cdist(coord_scaled[start:end], sp_coord_scaled, p=2) ** 2
                dist = finite_tensor(feat_dist + coord_dist, nan=1e6, pos=1e6, neg=0.0)

                if k < m:
                    nn_idx = torch.topk(dist, k=k, largest=False, dim=1).indices
                    local_dist = torch.full_like(dist, 1e6)
                    local_dist.scatter_(1, nn_idx, dist.gather(1, nn_idx))
                    dist = local_dist

                q_chunk = F.softmax(-dist, dim=1)
                q_chunk = finite_tensor(q_chunk, nan=0.0, pos=1.0, neg=0.0)
                q_chunk = q_chunk / q_chunk.sum(dim=1, keepdim=True).clamp_min(1e-6)
                q[start:end] = q_chunk

        return q

    def _update_centers(
        self,
        feat: torch.Tensor,
        coords: torch.Tensor,
        q: torch.Tensor,
    ):
        feat = finite_tensor(feat, nan=0.0, pos=10.0, neg=-10.0)

        coords = finite_tensor(
            coords,
            nan=0.0,
            pos=1.0,
            neg=0.0,
        ).clamp(0.0, 1.0)

        q = finite_tensor(q, nan=0.0, pos=1.0, neg=0.0)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-6)

        mass = q.sum(dim=0, keepdim=True).t().clamp_min(1e-6)

        sp_feat = torch.matmul(q.t(), feat) / mass
        sp_coord = torch.matmul(q.t(), coords) / mass

        sp_feat = finite_tensor(sp_feat, nan=0.0, pos=10.0, neg=-10.0)

        sp_coord = finite_tensor(
            sp_coord,
            nan=0.0,
            pos=1.0,
            neg=0.0,
        ).clamp(0.0, 1.0)

        return sp_feat, sp_coord, mass

    def _build_spatial_graph(
        self,
        sp_feat: torch.Tensor,
        sp_coord: torch.Tensor,
        q: torch.Tensor,
    ):

        sp_feat = finite_tensor(sp_feat, nan=0.0, pos=10.0, neg=-10.0)
        sp_coord = finite_tensor(sp_coord, nan=0.0, pos=1.0, neg=0.0).clamp(0.0, 1.0)
        q = finite_tensor(q, nan=0.0, pos=1.0, neg=0.0)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-6)

        device = sp_feat.device
        n_sp = sp_feat.size(0)
        labels = torch.argmax(q, dim=1).long().clamp(0, n_sp - 1)

        expected_n = int(self.grid_res * self.grid_res)
        if labels.numel() != expected_n:
            return self._build_fallback_mutual_knn_graph(sp_feat, sp_coord)

        label_map = labels.view(self.grid_res, self.grid_res)

        right_a = label_map[:, :-1].reshape(-1)
        right_b = label_map[:, 1:].reshape(-1)
        down_a = label_map[:-1, :].reshape(-1)
        down_b = label_map[1:, :].reshape(-1)

        src = torch.cat([right_a, down_a], dim=0)
        dst = torch.cat([right_b, down_b], dim=0)
        boundary_mask = src != dst
        src = src[boundary_mask]
        dst = dst[boundary_mask]

        if src.numel() == 0:
            loop = torch.arange(n_sp, device=device)
            edge_index = torch.stack([loop, loop], dim=0).contiguous()
            edge_attr = torch.ones((n_sp, 1), device=device, dtype=torch.float32)
            return edge_index, edge_attr, edge_attr.view(-1)


        a = torch.minimum(src, dst)
        b = torch.maximum(src, dst)
        undirected = torch.unique(torch.stack([a, b], dim=1), dim=0)
        edge_pair = torch.cat([undirected, undirected[:, [1, 0]]], dim=0)

        loop = torch.arange(n_sp, device=device)
        loop_pair = torch.stack([loop, loop], dim=1)
        edge_pair = torch.cat([edge_pair, loop_pair], dim=0)

        edge_src = edge_pair[:, 0].long()
        edge_dst = edge_pair[:, 1].long()

        feat_norm = F.normalize(sp_feat, dim=-1, eps=1e-6)
        cosine = (feat_norm[edge_src] * feat_norm[edge_dst]).sum(dim=1)
        cosine = torch.clamp(cosine, min=0.0, max=1.0)

        dist = torch.norm(sp_coord[edge_src] - sp_coord[edge_dst], p=2, dim=1)
        non_loop = edge_src != edge_dst
        sigma = torch.median(dist[non_loop]).clamp_min(1e-6) if torch.any(non_loop) else torch.tensor(1.0, device=device, dtype=dist.dtype)
        dist_weight = torch.exp(-(dist ** 2) / (2.0 * sigma ** 2))
        edge_weight = torch.where(non_loop, dist_weight * cosine, torch.ones_like(dist))


        keep = (~non_loop) | (edge_weight > 1e-6)
        edge_src = edge_src[keep]
        edge_dst = edge_dst[keep]
        edge_weight = edge_weight[keep]

        edge_index = torch.stack([edge_src, edge_dst], dim=0).contiguous()
        edge_attr = edge_weight.view(-1, 1).contiguous().float()
        return edge_index, edge_attr, edge_weight.float()

    def _build_fallback_mutual_knn_graph(
        self,
        sp_feat: torch.Tensor,
        sp_coord: torch.Tensor,
        k_spatial: int = 8,
    ):

        device = sp_feat.device
        n_sp = sp_feat.size(0)

        if n_sp <= 1:
            loop = torch.arange(n_sp, device=device)
            edge_index = torch.stack([loop, loop], dim=0).contiguous()
            edge_attr = torch.ones((max(n_sp, 1), 1), device=device, dtype=torch.float32)
            return edge_index, edge_attr, edge_attr.view(-1)

        dist_mat = torch.cdist(sp_coord, sp_coord, p=2)
        eye_mask = torch.eye(n_sp, device=device, dtype=torch.bool)
        dist_masked = dist_mat.clone()
        dist_masked[eye_mask] = float("inf")

        k = min(k_spatial, max(1, n_sp - 1))
        nn_idx = torch.topk(dist_masked, k=k, largest=False, dim=1).indices
        knn = torch.zeros((n_sp, n_sp), device=device, dtype=torch.bool)
        rows = torch.arange(n_sp, device=device).unsqueeze(1).expand(-1, k)
        knn[rows, nn_idx] = True

        mutual = knn & knn.t()
        mutual.fill_diagonal_(True)
        edge_pair = mutual.nonzero(as_tuple=False)
        edge_src = edge_pair[:, 0].long()
        edge_dst = edge_pair[:, 1].long()

        feat_norm = F.normalize(sp_feat, dim=-1, eps=1e-6)
        cosine = (feat_norm[edge_src] * feat_norm[edge_dst]).sum(dim=1).clamp(0.0, 1.0)
        dist = torch.norm(sp_coord[edge_src] - sp_coord[edge_dst], p=2, dim=1)
        non_loop = edge_src != edge_dst
        sigma = torch.median(dist[non_loop]).clamp_min(1e-6) if torch.any(non_loop) else torch.tensor(1.0, device=device)
        dist_weight = torch.exp(-(dist ** 2) / (2.0 * sigma ** 2))
        edge_weight = torch.where(non_loop, dist_weight * cosine, torch.ones_like(dist))
        keep = (~non_loop) | (edge_weight > 1e-6)

        edge_src = edge_src[keep]
        edge_dst = edge_dst[keep]
        edge_weight = edge_weight[keep]
        edge_index = torch.stack([edge_src, edge_dst], dim=0).contiguous()
        edge_attr = edge_weight.view(-1, 1).contiguous().float()
        return edge_index, edge_attr, edge_weight.float()

    def forward(self, feat: torch.Tensor, coords: torch.Tensor):
        sp_feat, sp_coord = self._regular_grid_init(feat, coords)

        q = None
        mass = None

        for _ in range(self.ssn_iters):
            q = self._assignment(feat, coords, sp_feat, sp_coord)
            sp_feat, sp_coord, mass = self._update_centers(feat, coords, q)

        edge_index, edge_attr, edge_weight = self._build_spatial_graph(sp_feat, sp_coord, q)

        sp_feature_with_coord = torch.cat([sp_feat, sp_coord], dim=-1)

        aux = {
            "Q": q,
            "sp_feat": sp_feat,
            "sp_coord": sp_coord,
            "mass": mass.squeeze(1),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_weight": edge_weight,
            "edge_mode": "RAG_distance_cosine_prior_bias_GATv2",
        }

        return sp_feature_with_coord, edge_index, edge_attr, aux


class PriorGATv2Conv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = False,
        dropout: float = 0.10,
        edge_dim: int = 1,
        beta: float = GAT_PRIOR_BETA,
        eps: float = 1e-4,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.dropout = float(dropout)
        self.edge_dim = int(edge_dim)
        self.beta = float(beta)
        self.eps = float(eps)
        self.negative_slope = float(negative_slope)

        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.empty(heads, out_channels))

        bias_dim = heads * out_channels if concat else out_channels
        self.bias = nn.Parameter(torch.zeros(bias_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        x = finite_tensor(x, nan=0.0, pos=10.0, neg=-10.0)
        n_nodes = x.size(0)

        if edge_index is None or edge_index.numel() == 0:
            out = self.lin_src(x).view(n_nodes, self.heads, self.out_channels)
            out = out.reshape(n_nodes, self.heads * self.out_channels) if self.concat else out.mean(dim=1)
            return finite_tensor(out + self.bias, nan=0.0, pos=10.0, neg=-10.0)

        src = edge_index[0].long()
        dst = edge_index[1].long()
        n_edges = src.numel()

        if edge_attr is None:
            edge_attr = x.new_ones((n_edges, self.edge_dim))
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        edge_attr = edge_attr.to(device=x.device, dtype=x.dtype)
        edge_prior = edge_attr[:, 0].clamp_min(self.eps)

        x_src = self.lin_src(x).view(n_nodes, self.heads, self.out_channels)
        x_dst = self.lin_dst(x).view(n_nodes, self.heads, self.out_channels)
        e_emb = self.lin_edge(edge_attr).view(n_edges, self.heads, self.out_channels)


        att_input = x_dst[dst] + x_src[src] + e_emb
        att_input = F.leaky_relu(att_input, negative_slope=self.negative_slope)
        score = (att_input * self.att.unsqueeze(0)).sum(dim=-1)


        score = score + self.beta * torch.log(edge_prior).unsqueeze(-1)
        score = finite_tensor(score, nan=-1e4, pos=1e4, neg=-1e4)

        alpha = pyg_softmax(score, dst, num_nodes=n_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = x_src[src] * alpha.unsqueeze(-1)
        out = x.new_zeros((n_nodes, self.heads, self.out_channels))
        out.index_add_(0, dst, msg)

        if self.concat:
            out = out.reshape(n_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        return finite_tensor(out, nan=0.0, pos=10.0, neg=-10.0)


class PaperSSNGATBranch(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_sp: int,
        grid_res: int,
        ssn_iters: int = 3,
        local_k: int = 9,
    ):
        super().__init__()

        self.ssn = PaperSSNLayer(
            in_dim,
            n_sp,
            grid_res,
            ssn_iters,
            local_k,
            feat_scale=1.0,
            coord_scale=0.35,
        )

        self.gat = PriorGATv2Conv(
            in_channels=in_dim + 2,
            out_channels=out_dim,
            heads=4,
            concat=False,
            dropout=0.10,
            edge_dim=1,
            beta=GAT_PRIOR_BETA,
            eps=1e-4,
        )

    def forward(
        self,
        feat: torch.Tensor,
        coords: torch.Tensor,
        return_aux: bool = False,
    ):
        sp_feat, edge_index, edge_attr, aux = self.ssn(feat, coords)

        h_sp = self.gat(sp_feat, edge_index, edge_attr=edge_attr)
        h_sp = finite_tensor(F.elu(h_sp), nan=0.0, pos=10.0, neg=-10.0)

        pixel_feat = torch.matmul(aux["Q"], h_sp)
        pixel_feat = finite_tensor(pixel_feat, nan=0.0, pos=10.0, neg=-10.0)

        if return_aux:
            aux["pixel_feat"] = pixel_feat
            return pixel_feat, aux

        return pixel_feat


class MultiScaleAttentionFusion(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()

        self.score_net = nn.Sequential(
            nn.Linear(in_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
    ):
        f1 = finite_tensor(f1, nan=0.0, pos=10.0, neg=-10.0)
        f2 = finite_tensor(f2, nan=0.0, pos=10.0, neg=-10.0)
        f3 = finite_tensor(f3, nan=0.0, pos=10.0, neg=-10.0)

        g = torch.cat(
            [
                f1.mean(dim=0),
                f2.mean(dim=0),
                f3.mean(dim=0),
            ],
            dim=0,
        )

        weights = F.softmax(
            self.score_net(
                finite_tensor(g, nan=0.0, pos=10.0, neg=-10.0)
            ),
            dim=0,
        )

        weights = finite_tensor(weights, nan=1.0 / 3.0, pos=1.0, neg=0.0)
        weights = weights / weights.sum().clamp_min(1e-6)

        fused = weights[0] * f1 + weights[1] * f2 + weights[2] * f3
        fused = finite_tensor(fused, nan=0.0, pos=10.0, neg=-10.0)

        return fused, weights


class FrontBackAttentionFusion(nn.Module):
    def __init__(
        self,
        front_dim: int,
        back_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
    ):
        super().__init__()

        self.front_proj = nn.Linear(front_dim, out_dim)
        self.back_proj = nn.Linear(back_dim, out_dim)

        self.branch_attn = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        mid_dim = max(out_dim // 2, 16)

        self.channel_attn = nn.Sequential(
            nn.Linear(out_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        f_front: torch.Tensor,
        f_back: torch.Tensor,
    ):
        f_front = finite_tensor(f_front, nan=0.0, pos=10.0, neg=-10.0)
        f_back = finite_tensor(f_back, nan=0.0, pos=10.0, neg=-10.0)

        f_front = finite_tensor(
            self.front_proj(f_front),
            nan=0.0,
            pos=10.0,
            neg=-10.0,
        )

        f_back = finite_tensor(
            self.back_proj(f_back),
            nan=0.0,
            pos=10.0,
            neg=-10.0,
        )

        attn = torch.softmax(
            self.branch_attn(torch.cat([f_front, f_back], dim=1)),
            dim=1,
        )

        attn = finite_tensor(attn, nan=0.5, pos=1.0, neg=0.0)
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)

        fused = attn[:, 0:1] * f_front + attn[:, 1:2] * f_back

        ch_weight = self.channel_attn(fused)
        ch_weight = finite_tensor(ch_weight, nan=1.0, pos=1.0, neg=0.0)

        out = fused * ch_weight
        out = finite_tensor(out, nan=0.0, pos=10.0, neg=-10.0)

        return out, attn, ch_weight

class DualBranchAttentionModel(nn.Module):

    def __init__(
        self,
        feat_dim: int = 7,
        grid_res: int = 200,
        cnn_dim: int = 128,
        vae_hidden: int = COORD_VAE_HIDDEN_DIM,
        vae_latent: int = PROJ_DIM,
        dropout_rate: float = DROPOUT,
        model_variant: str = None,
    ):
        super().__init__()
        del cnn_dim

        self.model_variant = str(model_variant or "transformer_cvae_ssn_gat_backend_u")
        allowed = {"transformer_cvae_ssn_gat_backend_u"}
        if self.model_variant not in allowed:
            raise ValueError(f"MODEL_VARIANT 必须为 {sorted(allowed)}，当前={self.model_variant}")

        self.cvae = CoordCVAE(
            feat_dim=feat_dim,
            d_model=TRANSFORMER_D_MODEL,
            coord_dim=2,
            hidden=vae_hidden,
            latent=vae_latent,
            dropout=0.10,
        )
        self.u_norm = nn.LayerNorm(vae_latent)
        self.projection_head = nn.Sequential(
            nn.Linear(vae_latent, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
        )

        self.backend_input_name = "u"
        ssn_in_dim = int(vae_latent)

        self.ssn_scales = list(SSN_SCALES)
        if len(self.ssn_scales) != 3:
            raise ValueError(f"SSN_SCALES 必须正好包含 3 个尺度，当前={self.ssn_scales}")
        self.ssn_branches = nn.ModuleList([
            PaperSSNGATBranch(ssn_in_dim, 64, int(n_sp), grid_res)
            for n_sp in self.ssn_scales
        ])
        self.ms_fusion = MultiScaleAttentionFusion(64)

        self.fb_fusion = FrontBackAttentionFusion(
            front_dim=vae_latent + 32,
            back_dim=64,
            hidden_dim=128,
            out_dim=128,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x_feat: torch.Tensor,
        coords: torch.Tensor,
        inject_noise: bool = False,
        return_aux: bool = False,
    ):
        x_feat = finite_tensor(x_feat, nan=0.0, pos=8.0, neg=-8.0).float()
        coords = finite_tensor(coords, nan=0.0, pos=1.0, neg=0.0).float().clamp(0.0, 1.0)


        cvae_out = self.cvae(x_feat, coords, inject_noise=inject_noise)
        transformer_feat = cvae_out["transformer_feat"]


        u_raw = finite_tensor(cvae_out["u"], nan=0.0, pos=10.0, neg=-10.0)
        if inject_noise and self.training:
            u_raw = u_raw + torch.randn_like(u_raw) * 0.01
        u_feat = finite_tensor(self.u_norm(u_raw), nan=0.0, pos=10.0, neg=-10.0)

        proj_emb = finite_tensor(
            self.projection_head(u_feat),
            nan=0.0,
            pos=10.0,
            neg=-10.0,
        )
        proj_emb = F.normalize(proj_emb, dim=1, eps=1e-6)
        front_fusion_input = finite_tensor(torch.cat([u_feat, proj_emb], dim=1), nan=0.0, pos=10.0, neg=-10.0)


        backend_input = u_feat
        backend_input = finite_tensor(backend_input, nan=0.0, pos=10.0, neg=-10.0)

        multi_feats = [branch(backend_input, coords, return_aux=False) for branch in self.ssn_branches]

        if inject_noise and self.training:
            multi_feats = [f + torch.randn_like(f) * 0.02 for f in multi_feats]

        back_feat, ms_weights = self.ms_fusion(multi_feats[0], multi_feats[1], multi_feats[2])
        fusion_feat, branch_attn, channel_attn = self.fb_fusion(front_fusion_input, back_feat)
        logits = self.classifier(fusion_feat)
        logits = finite_tensor(logits, nan=0.0, pos=20.0, neg=-20.0)

        if not return_aux:
            return logits

        return logits, {
            "model_variant": self.model_variant,
            "backend_input_name": self.backend_input_name,
            "ssn_scales": torch.as_tensor(self.ssn_scales, device=logits.device),
            "ms_weights": ms_weights,
            "branch_attn": branch_attn,
            "channel_attn": channel_attn,
            "transformer_feat": transformer_feat,
            "front_feat": u_feat,
            "front_fusion_input": front_fusion_input,
            "back_feat": back_feat,
            "backend_input": backend_input,
            "proj_emb": proj_emb,
            "proj_u": proj_emb,
            "u": u_feat,
            "mu": cvae_out["mu"],
            "logvar": cvae_out["logvar"],
            "z": cvae_out["z"],
            "recon": cvae_out["recon"],
        }



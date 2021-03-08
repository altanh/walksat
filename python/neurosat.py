# https://arxiv.org/pdf/1903.04671.pdf
# adapted from https://github.com/dselsam/neurocore-public/blob/master/python/neurosat.py

import numpy as np
import torch
from torch import nn
from torch.functional import F


class MLP(nn.Module):
    def __init__(self, in_features, hidden_layers, act, act_final=False):
        super().__init__()
        self.in_features = in_features
        self.hidden_layers = hidden_layers
        self.act = act
        self.act_final = act_final

        d_in = in_features
        layers = []
        for d_out in hidden_layers:
            layers.append(nn.Linear(d_in, d_out, bias=True))
            d_in = d_out
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.act_final or i + 1 < len(self.layers):
                x = self.act(x)
        return x


DEFAULT_EMBEDDING_DIM = 128
DEFAULT_MLP_ARCH = {
    "c_arch": [DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_DIM],
    "l_arch": [DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_DIM],
    "v_arch": [DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_DIM, 1],
}


class NeuroSATConfig:
    def __init__(
        self,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        mlp_arch=DEFAULT_MLP_ARCH,
        num_rounds=4,
        lc_scale=0.1,
        cl_scale=0.1,
    ):
        assert "c_arch" in mlp_arch
        assert "l_arch" in mlp_arch
        assert "v_arch" in mlp_arch
        self.embedding_dim = embedding_dim
        self.mlp_arch = mlp_arch
        self.num_rounds = num_rounds
        self.lc_scale = lc_scale
        self.cl_scale = cl_scale


class NeuroSAT(nn.Module):
    def __init__(self, cfg: NeuroSATConfig):
        super().__init__()
        self.cfg = cfg

        d = cfg.embedding_dim
        c_mlps, l_mlps = [], []
        for _ in range(cfg.num_rounds):
            c_mlps.append(MLP(2 * d, cfg.mlp_arch["c_arch"], F.relu, act_final=False))
            l_mlps.append(MLP(3 * d, cfg.mlp_arch["l_arch"], F.relu, act_final=False))
        self.c_mlps = nn.ModuleList(c_mlps)
        self.l_mlps = nn.ModuleList(l_mlps)
        self.v_mlp = MLP(2 * d, cfg.mlp_arch["v_arch"], F.relu, act_final=False)
        self.c_init_scale = nn.Parameter(torch.tensor(1.0 / np.sqrt(d)))
        self.l_init_scale = nn.Parameter(torch.tensor(1.0 / np.sqrt(d)))
        self.cl_scale = nn.Parameter(torch.tensor(cfg.cl_scale))
        self.lc_scale = nn.Parameter(torch.tensor(cfg.lc_scale))

    def forward(self, G):
        num_clauses, num_lits = G.shape
        num_vars = num_lits // 2

        L = (
            torch.ones((num_lits, self.cfg.embedding_dim), dtype=torch.float32)
            * self.l_init_scale
        )
        C = (
            torch.ones((num_clauses, self.cfg.embedding_dim), dtype=torch.float32)
            * self.c_init_scale
        )
        G_t = torch.transpose(G, 0, 1)

        def flip(lits):
            return torch.cat([lits[num_vars:, :], lits[0:num_vars, :]], axis=0)

        for r in range(self.cfg.num_rounds):
            LC_msgs = torch.sparse.mm(G, L) * self.lc_scale
            C = self.c_mlps[r](torch.cat([C, LC_msgs], dim=-1))
            # TODO: normalize C?
            CL_msgs = torch.sparse.mm(G_t, C) * self.cl_scale
            L = self.l_mlps[r](torch.cat([L, CL_msgs, flip(L)], dim=-1))

        V = torch.cat([L[0:num_vars, :], L[num_vars:, :]], dim=1)
        V_scores = self.v_mlp(V)
        return torch.squeeze(V_scores)

    @staticmethod
    def G_from_indices(indices, num_clauses, num_lits):
        return torch.sparse_coo_tensor(
            indices=indices,
            values=torch.ones((indices.shape[1])),
            size=(num_clauses, num_lits),
        )


if __name__ == "__main__":
    idxs = np.array([[0, 0], [0, 1], [0, 2], [1, 3], [1, 4], [1, 5]])
    n_vars = 3
    n_clauses = 2
    G = NeuroSAT.G_from_indices(idxs.T, n_clauses, n_vars * 2)
    model = NeuroSAT(NeuroSATConfig())

    r = model(G)
    import pdb

    pdb.set_trace()

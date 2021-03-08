import numpy as np
import glob
import os
import torch

from neurosat import NeuroSATConfig, NeuroSAT
from gen_data import OUTPUT_DIR as DATA_DIR


if __name__ == "__main__":
    cfg = NeuroSATConfig(num_rounds=24)
    nsat = NeuroSAT(cfg)
    sgd = torch.optim.Adam(nsat.parameters(), lr=0.001)
    epochs = 20

    X_files = list(glob.glob(DATA_DIR + "/X_uf250*.npy"))
    X_files.sort()
    train_frac = 0.7

    for epoch in range(epochs):
        for X_file in X_files[: int(len(X_files) * train_frac)]:
            sgd.zero_grad()

            Y_file = DATA_DIR + "/Y_" + X_file.split("X_")[1]
            X = np.load(X_file, allow_pickle=False)
            Y = torch.tensor(np.load(Y_file, allow_pickle=False).astype("float32"))
            num_clauses = X[0, -1] + 1
            num_lits = np.max(X[1, :]) + 1

            G = NeuroSAT.G_from_indices(X, num_clauses, num_lits)
            pred = nsat(G)
            loss = torch.nn.BCEWithLogitsLoss()(pred, Y)
            loss.backward()
            sgd.step()

            # print(loss)
        print(((torch.nn.Sigmoid()(pred) > 0.5) != Y).sum())

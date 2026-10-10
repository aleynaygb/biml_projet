#!/usr/bin/env python
"""Expériences multi-seeds : comparaison de modèles + ablation.

Pour chaque seed : nouveau découpage 70/15/15 (identique pour tous les modèles) et nouvelle
initialisation. Sorties : results.json, results_main.tex, results_ablation.tex.

Exemples (depuis le dossier src/) :
    python run_experiments.py --smoke          # test rapide (1 seed, 5 époques)
    python run_experiments.py --seeds 10       # expérience complète
"""
import argparse
import copy
import json
import os
import time

import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import degree, from_networkx

from make_tables import summary_text, write_tables

# ----------------------------------------------------------------------------- features
RAW = ("pop", "latlon")                      # 3 features d'origine
FULL = ("pop", "xyz", "sin", "deg")          # 25 features enrichies
SIN_FREQS = (1, 2, 4, 8, 16)


def zscore(t):
    return (t - t.mean(dim=0, keepdim=True)) / (t.std(dim=0, keepdim=True) + 1e-6)


def build_features(raw, edge_index, parts):
    """raw : [N,3] = population, lat (deg), lon (deg) bruts. Aucun label utilisé."""
    n = raw.shape[0]
    lat, lon = th.deg2rad(raw[:, 1]), th.deg2rad(raw[:, 2])
    feats = []
    if "pop" in parts:
        feats.append(zscore(th.log1p(raw[:, 0:1])))
    if "latlon" in parts:
        feats.append(zscore(raw[:, 1:3]))
    if "xyz" in parts:
        feats.append(th.stack([th.cos(lat) * th.cos(lon), th.cos(lat) * th.sin(lon), th.sin(lat)], dim=1))
    if "sin" in parts:
        feats.append(th.stack([f(k * a) for k in SIN_FREQS for a in (lat, lon) for f in (th.sin, th.cos)], dim=1))
    if "deg" in parts:
        feats.append(zscore(th.log1p(degree(edge_index[0], n)).unsqueeze(1)))
    return th.cat(feats, dim=1)


# ----------------------------------------------------------------------------- modèle
class Net(nn.Module):
    """GNN / MLP à `layers` couches. kind in {gcn, sage, gat, mlp}."""

    def __init__(self, kind, dim_in, dim_out, hidden=256, dropout=0.4, layers=2, residual=False, heads=8):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        self.residual = residual and layers > 1
        self.convs = nn.ModuleList()
        d = dim_in
        for _ in range(layers - 1):
            self.convs.append(self._layer(kind, d, hidden, heads, dropout, last=False))
            d = hidden
        if self.residual:
            d += dim_in  # concaténation de l'entrée avant la dernière couche
        self.convs.append(self._layer(kind, d, dim_out, heads, dropout, last=True))

    @staticmethod
    def _layer(kind, d_in, d_out, heads, dropout, last):
        if kind == "gcn":
            return GCNConv(d_in, d_out)
        if kind == "sage":
            return SAGEConv(d_in, d_out)
        if kind == "gat":
            if last:
                return GATConv(d_in, d_out, heads=1, concat=False, dropout=dropout)
            return GATConv(d_in, d_out // heads, heads=heads, dropout=dropout)
        if kind == "mlp":
            return nn.Linear(d_in, d_out)
        raise ValueError(kind)

    def _apply(self, conv, h, edge_index):
        return conv(h) if self.kind == "mlp" else conv(h, edge_index)

    def forward(self, x, edge_index):
        h = x
        n_hidden = len(self.convs) - 1
        for i, conv in enumerate(self.convs[:-1]):
            h = self._apply(conv, h, edge_index)
            h = F.elu(h) if self.kind == "gat" else F.relu(h)
            if i == n_hidden - 1 and self.residual:
                h = th.cat([h, x], dim=1)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self._apply(self.convs[-1], h, edge_index)
        return F.log_softmax(h, dim=1)


def cfg(**kw):
    base = dict(kind="sage", feats=FULL, hidden=256, dropout=0.4, layers=2, residual=False,
                heads=8, lr=0.01, wd=5e-4, epochs=600, patience=100)
    base.update(kw)
    return base


def without(parts, name):
    return tuple(p for p in parts if p != name)


CONFIGS = {
    # --- features d'origine (reprend les modèles du notebook)
    "gcn_base": cfg(kind="gcn", feats=RAW, hidden=64, dropout=0.0, epochs=300),
    "gcn_tuned_raw": cfg(kind="gcn", feats=RAW, residual=True, lr=0.02, epochs=800, patience=150),
    "gat_raw": cfg(kind="gat", feats=RAW, hidden=64, dropout=0.3, epochs=500),
    "sage_raw": cfg(kind="sage", feats=RAW),
    # --- features enrichies
    "gcn_tuned_full": cfg(kind="gcn", residual=True, lr=0.02, epochs=800, patience=150),
    "gat_full": cfg(kind="gat", hidden=64, dropout=0.3, epochs=500),
    "sage_full": cfg(kind="sage"),
    "mlp_full": cfg(kind="mlp"),
    # --- ablation du modèle final (sage_full)
    "sage_no_sin": cfg(feats=without(FULL, "sin")),
    "sage_no_xyz": cfg(feats=without(FULL, "xyz")),
    "sage_no_deg": cfg(feats=without(FULL, "deg")),
    "sage_no_pop": cfg(feats=without(FULL, "pop")),
    "sage_no_dropout": cfg(dropout=0.0),
    "sage_h64": cfg(hidden=64),
    "sage_l1": cfg(layers=1),
    "sage_l3": cfg(layers=3),
    "gcn_swap": cfg(kind="gcn"),
}


# ----------------------------------------------------------------------------- entraînement
def fit(model, x, ei, y, m, c):
    opt = th.optim.Adam(model.parameters(), lr=c["lr"], weight_decay=c["wd"])
    best, wait, best_state = float("inf"), 0, None
    for _ in range(c["epochs"]):
        model.train()
        opt.zero_grad()
        loss = F.nll_loss(model(x, ei)[m["train"]], y[m["train"]])
        loss.backward()
        opt.step()
        model.eval()
        with th.no_grad():
            val_loss = F.nll_loss(model(x, ei)[m["val"]], y[m["val"]]).item()
        if val_loss < best:
            best, wait, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= c["patience"]:
                break
    model.load_state_dict(best_state)


def metrics(pred, y, m):
    acc = lambda k: (pred[m[k]] == y[m[k]]).float().mean().item()
    f1 = f1_score(y[m["test"]].numpy(), pred[m["test"]].numpy(), average="macro", zero_division=0)
    return {"train_acc": acc("train"), "val_acc": acc("val"), "test_acc": acc("test"), "test_f1": float(f1)}


def run_gnn(c, raw, ei, y, m, n_classes, feat_cache, smoke):
    if c["feats"] not in feat_cache:
        feat_cache[c["feats"]] = build_features(raw, ei, c["feats"])
    x = feat_cache[c["feats"]]
    c = dict(c, epochs=5) if smoke else c
    model = Net(c["kind"], x.shape[1], n_classes, c["hidden"], c["dropout"], c["layers"], c["residual"], c["heads"])
    fit(model, x, ei, y, m, c)
    model.eval()
    with th.no_grad():
        pred = model(x, ei).argmax(dim=1)
    return metrics(pred, y, m)


def run_knn(raw, ei, y, m):
    X = build_features(raw, ei, ("xyz",)).numpy()
    yn = y.numpy()
    mk = {k: v.numpy() for k, v in m.items()}
    best = None
    for k in (1, 3, 5, 7, 10, 15, 20):
        knn = KNeighborsClassifier(n_neighbors=k).fit(X[mk["train"]], yn[mk["train"]])
        va = knn.score(X[mk["val"]], yn[mk["val"]])
        if best is None or va > best[0]:
            best = (va, k, knn)
    knn = best[2]
    pred = th.tensor(knn.predict(X), dtype=th.long)
    out = metrics(pred, y, m)
    out["k"] = best[1]
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="../graphes/airportsAndCoordAndPop.graphml")
    ap.add_argument("--out-dir", default="../results")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--only", nargs="*", help="clés de CONFIGS à exécuter (par défaut : toutes + knn)")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 5 époques : vérifie que tout tourne")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = 1

    G = nx.read_graphml(args.graph, node_type=int)
    data = from_networkx(G, group_node_attrs=["population", "lat", "lon"])
    raw = data.x.clone().float()
    ei = data.edge_index
    y = th.tensor(LabelEncoder().fit_transform(data.country), dtype=th.long)
    n, n_classes = data.num_nodes, int(y.max().item()) + 1
    print(f"{n} nœuds, {ei.shape[1]} arêtes, {n_classes} classes")

    keys = args.only if args.only else list(CONFIGS) + ["knn"]
    results = {k: {"train_acc": [], "val_acc": [], "test_acc": [], "test_f1": []} for k in keys}
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = list(range(args.seeds))

    for seed in seeds:
        g = th.Generator().manual_seed(seed)
        perm = th.randperm(n, generator=g)
        n_tr, n_va = int(0.70 * n), int(0.15 * n)
        m = {}
        for name, idx in (("train", perm[:n_tr]), ("val", perm[n_tr:n_tr + n_va]), ("test", perm[n_tr + n_va:])):
            m[name] = th.zeros(n, dtype=th.bool)
            m[name][idx] = True
        feat_cache = {}
        for key in keys:
            t0 = time.time()
            th.manual_seed(seed)
            res = run_knn(raw, ei, y, m) if key == "knn" else run_gnn(CONFIGS[key], raw, ei, y, m, n_classes, feat_cache, args.smoke)
            for metric in results[key]:
                results[key][metric].append(res[metric])
            print(f"[seed {seed}] {key:<16} val={res['val_acc']*100:5.1f} test={res['test_acc']*100:5.1f} ({time.time()-t0:.0f}s)", flush=True)
        payload = {"seeds": seeds[:seed + 1], "n_nodes": n, "configs": CONFIGS, "results": results}
        with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, default=list, indent=1)   # sauvegarde après chaque seed

    if not args.smoke:
        write_tables(results, args.out_dir)
    print("\n" + summary_text(results))
    print(f"\nRésultats écrits dans {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

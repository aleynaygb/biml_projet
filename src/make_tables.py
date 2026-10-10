"""Agrégation des résultats multi-seeds et génération des tableaux LaTeX.

N'a besoin que de numpy : peut être relancé seul à partir de results.json
    python make_tables.py ../results/results.json ../results
"""
import json
import os
import sys

import numpy as np

# (clé, libellé, groupe) -- ordre d'affichage du tableau principal
MAIN_ROWS = [
    ("knn", "kNN (coordonnées 3D)", "Sans graphe"),
    ("mlp_full", "MLP (features enrichies)", "Sans graphe"),
    ("gcn_base", "GCN (baseline)", "Features d'origine (3)"),
    ("gcn_tuned_raw", "GCN amélioré", "Features d'origine (3)"),
    ("gat_raw", "GAT", "Features d'origine (3)"),
    ("sage_raw", "GraphSAGE", "Features d'origine (3)"),
    ("gcn_tuned_full", "GCN amélioré", "Features enrichies (25)"),
    ("gat_full", "GAT", "Features enrichies (25)"),
    ("sage_full", "GraphSAGE (notre modèle)", "Features enrichies (25)"),
]

# (clé, libellé) -- ablation du modèle final (GraphSAGE + features enrichies)
ABLATION_ROWS = [
    ("sage_full", "Modèle complet"),
    ("sage_no_sin", "sans encodages sinusoïdaux"),
    ("sage_no_xyz", "sans coordonnées 3D"),
    ("sage_no_deg", "sans degré"),
    ("sage_no_pop", "sans population"),
    ("sage_raw", "features d'origine (3 au lieu de 25)"),
    ("mlp_full", "sans graphe (MLP)"),
    ("gcn_swap", "SAGEConv $\\rightarrow$ GCNConv"),
    ("gat_full", "SAGEConv $\\rightarrow$ GATConv (hyperparam. du GAT)"),
    ("sage_no_dropout", "sans dropout"),
    ("sage_h64", "64 unités cachées (au lieu de 256)"),
    ("sage_l1", "1 couche (au lieu de 2)"),
    ("sage_l3", "3 couches (au lieu de 2)"),
]


def _arr(res, key, metric):
    return np.asarray(res[key][metric], dtype=float) * 100.0


def mean_std(res, key, metric):
    a = _arr(res, key, metric)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def _num(x):
    return f"{x:.1f}".replace(".", ",")


def _cell(m, s, bold=False):
    txt = f"{_num(m)} $\\pm$ {_num(s)}"
    return f"\\textbf{{{txt}}}" if bold else txt


def main_table(res):
    cols = [("val_acc", "Acc. val"), ("test_acc", "Acc. test"), ("test_f1", "F1 macro test")]
    rows = [r for r in MAIN_ROWS if r[0] in res]
    best = {c: max(mean_std(res, k, c)[0] for k, _, _ in rows) for c, _ in cols}
    out = ["\\begin{tabular}{l c c c}", "\\hline",
           "Modèle & " + " & ".join(h for _, h in cols) + " \\\\", "\\hline"]
    group = None
    for key, label, grp in rows:
        if grp != group:
            if group is not None:
                out.append("\\hline")
            out.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{grp}}}}} \\\\")
            group = grp
        cells = []
        for c, _ in cols:
            m, s = mean_std(res, key, c)
            cells.append(_cell(m, s, bold=abs(m - best[c]) < 1e-9))
        out.append(f"{label} & " + " & ".join(cells) + " \\\\")
    out += ["\\hline", "\\end{tabular}"]
    return "\n".join(out)


def ablation_table(res):
    rows = [r for r in ABLATION_ROWS if r[0] in res]
    full = _arr(res, "sage_full", "test_acc")
    n = len(full)
    out = ["\\begin{tabular}{l c c c}", "\\hline",
           "Variante & Acc. test & $\\Delta$ vs complet & Victoires \\\\", "\\hline"]
    for key, label in rows:
        m, s = mean_std(res, key, "test_acc")
        if key == "sage_full":
            out.append(f"{label} & {_cell(m, s)} & -- & -- \\\\")
            out.append("\\hline")
            continue
        d = _arr(res, key, "test_acc") - full
        dm, ds = d.mean(), (d.std(ddof=1) if n > 1 else 0.0)
        wins = int((full > _arr(res, key, "test_acc")).sum())
        sign = "+" if dm > 0 else ""
        out.append(f"{label} & {_cell(m, s)} & {sign}{_num(dm)} $\\pm$ {_num(ds)} & {wins}/{n} \\\\")
    out += ["\\hline", "\\end{tabular}"]
    return "\n".join(out)


def summary_text(res):
    lines = []
    if "sage_full" not in res:
        return ""
    full = _arr(res, "sage_full", "test_acc")
    n = len(full)
    lines.append(f"{'modèle':<20} {'val':>14} {'test':>14} {'F1':>14}")
    for key in res:
        v = mean_std(res, key, "val_acc"); t = mean_std(res, key, "test_acc"); f = mean_std(res, key, "test_f1")
        lines.append(f"{key:<20} {v[0]:6.2f}±{v[1]:5.2f} {t[0]:6.2f}±{t[1]:5.2f} {f[0]:6.2f}±{f[1]:5.2f}")
    lines.append("\nDifférences appariées vs sage_full (test acc, points) : moyenne ± écart-type, t-stat, victoires de sage_full")
    for key in res:
        if key == "sage_full":
            continue
        d = _arr(res, key, "test_acc") - full
        sd = d.std(ddof=1) if n > 1 else float("nan")
        t = d.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else float("nan")
        lines.append(f"{key:<20} {d.mean():+6.2f} ± {sd:5.2f}   t={t:6.2f}   {int((full > _arr(res, key, 'test_acc')).sum())}/{n}")
    return "\n".join(lines)


def write_tables(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if any(k in res for k, _, _ in MAIN_ROWS):
        with open(os.path.join(out_dir, "results_main.tex"), "w", encoding="utf-8") as f:
            f.write(main_table(res) + "\n")
    if "sage_full" in res:
        with open(os.path.join(out_dir, "results_ablation.tex"), "w", encoding="utf-8") as f:
            f.write(ablation_table(res) + "\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../results/results.json"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(path) or "."
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    write_tables(payload["results"], out_dir)
    print(summary_text(payload["results"]))
    print(f"\nTableaux écrits dans {out_dir}")

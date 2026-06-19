import sys
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np

# Set project root path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import config as cfg
import src.graph_construction as gc
import src.utils as ut

def local_modularity_weighted(graph: nx.Graph, nodes: set, gamma: float = 1.0) -> float:
    strength_total = sum(dict(graph.degree(weight="weight")).values())
    if strength_total == 0:
        return 0.0

    strength_community = 0.0
    for node in nodes:
        strength_community += graph.degree(node, weight="weight")

    internal_weight = 0.0
    for node in nodes:
        for neighbor in graph.neighbors(node):
            if neighbor in nodes:
                internal_weight += graph[node][neighbor].get("weight", 1.0)

    fraction_real = internal_weight / strength_total
    fraction_expected = (strength_community / strength_total) ** 2
    return fraction_real - (gamma * fraction_expected)

def _mean_or_none(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    if "n_obs" in df.columns:
        valid_df = df[[col, "n_obs"]].dropna()
        if valid_df.empty:
            return None
        total_obs = valid_df["n_obs"].sum()
        if total_obs == 0:
            return None
        return float((valid_df[col] * valid_df["n_obs"]).sum() / total_obs)
    else:
        series = df[col].dropna()
        if series.empty:
            return None
        return float(series.mean())

def _median_or_none(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None

    if "n_obs" in df.columns:
        valid_df = df[[col, "n_obs"]].dropna()
        if valid_df.empty:
            return None

        total_obs = valid_df["n_obs"].sum()
        if total_obs == 0:
            return None

        valid_df = valid_df.sort_values(col).reset_index(drop=True)
        cumsum = valid_df["n_obs"].cumsum()
        cutoff = total_obs / 2.0

        if (cumsum == cutoff).any():
            pos = cumsum[cumsum == cutoff].index[0]
            val1 = valid_df.loc[pos, col]
            val2 = valid_df.loc[pos + 1, col]
            return float((val1 + val2) / 2.0)
        else:
            return float(valid_df.loc[cumsum > cutoff, col].iloc[0])
    else:
        series = df[col].dropna()
        if series.empty:
            return None
        return float(series.median())

def _fmt_number(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.2f}"

def main():
    # Load input data
    nodelist_path = project_root / "data/processed/nodelist_ciuo.csv"
    base_enes_path = project_root / "data/processed/base_enespersonas.csv"

    nodelist_df = pd.read_csv(nodelist_path)
    enes_df = pd.read_csv(base_enes_path)

    # Reconstruct the projection graph
    caes_id = cfg.DATA_ENES_PISAC["col_caes_id"]
    ciuo_id = cfg.DATA_ENES_PISAC["col_ciuo_id"]

    # Rebuild unsorted bipartite graph
    bipartite_graph = gc.build_bipartite_graph(
        enes_df,
        caes_id,
        ciuo_id,
        logscale=cfg.LOGSCALE,
    )
    # Rebuild unsorted custom weighted projection graph
    ciuo_projection = gc.generic_weighted_projected_graph(
        bipartite_graph, 
        target_partition=0,
        weight_function=gc.weighted_hidalgo_proximity_weight
    )

    # Nodelist is already populated with communities (0 to 8, or -1 for none)
    # Filter nodes to keep only those with valid community >= 0
    df_valid = nodelist_df[nodelist_df["community"] >= 0].copy()
    df_valid["community"] = df_valid["community"].apply(lambda c: f"C{int(c)}")

    rows = []
    rows.append(
        "Community & Dominant groups (by count) & Mean Female % & Mean Public Sector % & Age median & Income median & Modularity & Workers (millions) \\\\"
    )

    group_col = cfg.DATA_NODELIST_CIUO["col_letra"] # ciuo1diglabel to get "2. Profesionales", etc.

    for comm_id, group in df_valid.groupby("community"):
        community_nodes = set(group["v183ciuo"].astype(int).tolist())
        local_mod = local_modularity_weighted(ciuo_projection, community_nodes, gamma=1.0)

        dominant_groups_str = "NA"
        if group_col and group_col in group.columns:
            dominant_groups = group[group_col].value_counts().head(3)
            dominant_groups_items = [
                f"{idx} ({count})" for idx, count in dominant_groups.items()
            ]
            dominant_groups_str = (
                "\\makecell[l]{" + " \\\\ ".join(dominant_groups_items) + "}"
            )

        female_mean = _mean_or_none(group, "female_pct")
        public_mean = _mean_or_none(group, "public_sector_pct")
        age_median = _median_or_none(group, "age_median")
        income_median = _median_or_none(group, "income_median")

        # In original nodelist_ciuo, we don't have total_workers_weighted, so we fall back to NA
        workers_millions = None

        rows.append(
            f"{comm_id} & {dominant_groups_str} & {_fmt_number(female_mean)} & {_fmt_number(public_mean)} & {_fmt_number(age_median)} & {_fmt_number(income_median)} & {local_mod:.4f} & {workers_millions} \\\\ \\hline"
        )

    print("=============================================================")
    print("NODELIST WITH COMMUNITIES TABLE")
    print("=============================================================")
    for row in rows:
        print(row)

if __name__ == "__main__":
    main()

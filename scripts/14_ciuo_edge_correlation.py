import config as cfg
import pandas as pd
import networkx as nx
import numpy as np
import src.graph_construction as gc
import src.plotting as pl

def main(enes_df=None, nodelist_ciuo_df=None):
	caes_id = cfg.DATA_ENES_PISAC["col_caes_id"]
	ciuo_id = cfg.DATA_ENES_PISAC["col_ciuo_id"]
	enes_path = cfg.DATA_PROCESSED_PATH / "base_enespersonas.csv"
	ciuo_nodelist_path = cfg.DATA_PROCESSED_PATH / "nodelist_ciuo.csv"

	if enes_df is None:
		enes_df = pd.read_csv(enes_path)
	if nodelist_ciuo_df is None:
		nodelist_ciuo_df = pd.read_csv(ciuo_nodelist_path, index_col=ciuo_id)

	# Accept both nodelist shapes: CIUO as a regular column or already set as index.
	if ciuo_id in nodelist_ciuo_df.columns:
		nodelist_ciuo_df = nodelist_ciuo_df.set_index(ciuo_id)

	missing_cols = [col for col in ["female_pct", "community"] if col not in nodelist_ciuo_df.columns]
	if missing_cols:
		raise KeyError(f"Missing required columns in nodelist_ciuo_df: {missing_cols}")

	feature_map = nodelist_ciuo_df["female_pct"].to_dict()
	color_map = nodelist_ciuo_df["community"].to_dict()
	color_map = {
		k: cfg.COMMUNITY_COLORS_PALETTE[int(v) % len(cfg.COMMUNITY_COLORS_PALETTE)] if v >= 0 else "gray"
		for k, v in color_map.items()
	}

	print("Building bipartite graph...")
	bipartite_graph = gc.build_bipartite_graph(
		enes_df,
		caes_id,
		ciuo_id,
		logscale=cfg.LOGSCALE,
	)

	# CIUO PROJECTION
	print("Building CIUO projection...")
	ciuo_projection = gc.generic_weighted_projected_graph(
		bipartite_graph, 
		target_partition=0,
		weight_function=gc.weighted_hidalgo_proximity_weight
	)
	
	cfg.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
	pl.compute_and_plot_edge_correlation(
		G=ciuo_projection,
		feature_map=feature_map,
		color_map=color_map,
		title=None, 
		output_path=cfg.IMAGE_DIR / "14_ciuo_edge_correlation.png", 
		save=True,
		perfect_line=False,
		figsize=cfg.EDGE_CORRELATION_FIGSIZE,
		font_size=cfg.PLOT_FONT_SIZE,
	)

if __name__ == "__main__":
	main()

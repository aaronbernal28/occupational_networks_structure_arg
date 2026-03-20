from scripts import *
import pandas as pd
import networkx as nx
import numpy as np

def main(enes_df=None, nodelist_ciuo_df=None):
	if enes_df is None:
		enes_df = pd.read_csv(ENES_PATH)
	if nodelist_ciuo_df is None:
		nodelist_ciuo_df = pd.read_csv(CIUO_NODELIST_PATH, index_col=CIUO_ID)

	# Accept both nodelist shapes: CIUO as a regular column or already set as index.
	if CIUO_ID in nodelist_ciuo_df.columns:
		nodelist_ciuo_df = nodelist_ciuo_df.set_index(CIUO_ID)

	missing_cols = [col for col in ["female_pct", "community"] if col not in nodelist_ciuo_df.columns]
	if missing_cols:
		raise KeyError(f"Missing required columns in nodelist_ciuo_df: {missing_cols}")

	feature_map = nodelist_ciuo_df["female_pct"].to_dict()
	color_map = nodelist_ciuo_df["community"].to_dict()
	color_map = {k: COMMUNITY_COLORS_PALETTE[int(v) % len(COMMUNITY_COLORS_PALETTE)] if v>=0 else "gray" for k, v in color_map.items()}

	print("Building bipartite graph...")
	bipartite_graph = gc.build_bipartite_graph(
		enes_df,
		CAES_ID,
		CIUO_ID,
		logscale=LOGSCALE,
	)

	# CIUO PROJECTION
	print("Building CIUO projection...")
	ciuo_projection = gc.generic_weighted_projected_graph(
		bipartite_graph, 
		target_partition=0,
		weight_function=gc.weighted_hidalgo_proximity_weight
	)
	
	IMAGE_DIR.mkdir(parents=True, exist_ok=True)
	pl.compute_and_plot_edge_correlation(
		G=ciuo_projection,
		feature_map=feature_map,
		color_map=color_map,
		title=None, 
		output_path=IMAGE_DIR / "14_ciuo_edge_correlation.png", 
		save=True,
		perfect_line=False
	)

if __name__ == "__main__":
	main()

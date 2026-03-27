import config as cfg
import pandas as pd
import src.communities as comm
import src.graph_construction as gc
import src.metrics as metrics
import src.plotting as pl
import src.utils as utils


def main(enes_df=None, nodelist_caes_df=None, nodelist_ciuo_df=None):
	CAES_PARTITION = 1
	CIUO_PARTITION = 0
	caes_id = cfg.DATA_ENES_PISAC["col_caes_id"]
	ciuo_id = cfg.DATA_ENES_PISAC["col_ciuo_id"]
	caes_ag = cfg.DATA_NODELIST_CAES["col_ag"]
	ciuo_3cat = cfg.DATA_NODELIST_CIUO["col_3cat"]

	enes_path = cfg.DATA_PROCESSED_PATH / "base_enespersonas.csv"
	caes_nodelist_path = cfg.DATA_PROCESSED_PATH / "nodelist_caes.csv"
	ciuo_nodelist_path = cfg.DATA_PROCESSED_PATH / "nodelist_ciuo.csv"

	if enes_df is None:
		enes_df = pd.read_csv(enes_path)
	if nodelist_caes_df is None:
		nodelist_caes_df = pd.read_csv(caes_nodelist_path, index_col=caes_id)
	if nodelist_ciuo_df is None:
		nodelist_ciuo_df = pd.read_csv(ciuo_nodelist_path, index_col=ciuo_id)

	bipartite_graph = gc.build_bipartite_graph(
		enes_df,
		caes_id,
		ciuo_id,
		logscale=cfg.LOGSCALE,
		caes_partition=CAES_PARTITION,
		ciuo_partition=CIUO_PARTITION,
	)

	# Node sizes proportional to absolute workers from processed node lists.
	caes_worker_counts = nodelist_caes_df["n_obs"].to_dict()
	ciuo_worker_counts = nodelist_ciuo_df["n_obs"].to_dict()
	node_size_map_workers = {**caes_worker_counts, **ciuo_worker_counts}

	caes_nodes = [
		node
		for node in bipartite_graph.nodes
		if bipartite_graph.nodes[node].get("bipartite") == CAES_PARTITION
	]
	ciuo_nodes = [
		node
		for node in bipartite_graph.nodes
		if bipartite_graph.nodes[node].get("bipartite") == CIUO_PARTITION
	]

	metric_results = metrics.summarize_graph(bipartite_graph)
	metrics.log_graph_metrics("Bipartite graph", metric_results)
	communities_bipartite, modularity_bipartite = comm.louvain_partition(
		bipartite_graph,
		seed=28,
	)
	num_bipartite_communities = len(set(communities_bipartite.values()))
	print(f"Modularity score: {modularity_bipartite:.4f}")
	print(f"Detected communities: {num_bipartite_communities}")
	print(f"CAES nodes: {len(caes_nodes)}")
	print(f"CIUO nodes: {len(ciuo_nodes)}")

	caes_group_col = caes_ag
	ciuo_group_col = ciuo_3cat
	if caes_group_col not in nodelist_caes_df.columns:
		raise KeyError(f"Missing '{caes_group_col}' column in CAES node list.")
	if ciuo_group_col not in nodelist_ciuo_df.columns:
		raise KeyError(f"Missing '{ciuo_group_col}' column in CIUO node list.")

	caes_group_map = nodelist_caes_df[caes_group_col].to_dict()
	ciuo_group_map = nodelist_ciuo_df[ciuo_group_col].to_dict()

	# Use color columns from CSV files
	caes_color_col = cfg.DATA_NODELIST_CAES["col_ag_color"]
	ciuo_color_col = cfg.DATA_NODELIST_CIUO["col_3cat_color"]
	if caes_color_col not in nodelist_caes_df.columns:
		raise KeyError(f"Missing '{caes_color_col}' column in CAES node list.")
	if ciuo_color_col not in nodelist_ciuo_df.columns:
		raise KeyError(f"Missing '{ciuo_color_col}' column in CIUO node list.")
		
	# Create color palettes from the CSV color columns
	caes_palette = nodelist_caes_df.groupby(caes_group_col)[caes_color_col].first().apply(utils.parse_color).to_dict()
	ciuo_palette = nodelist_ciuo_df.groupby(ciuo_group_col)[ciuo_color_col].first().apply(utils.parse_color).to_dict()

	# Map nodes to their group categories (not letra/1digit)
	label_map_groups = {}
	label_map_groups.update(nodelist_caes_df[caes_group_col].to_dict())
	label_map_groups.update(nodelist_ciuo_df[ciuo_group_col].to_dict())

	color_map_global = {}
	for node in bipartite_graph.nodes:
		if bipartite_graph.nodes[node].get("bipartite") == CAES_PARTITION:
			group = caes_group_map.get(node, "Unknown")
			color_map_global[node] = caes_palette.get(group, "gray")
		else:
			group = ciuo_group_map.get(node, "Unknown")
			color_map_global[node] = ciuo_palette.get(group, "gray")

	colored_output = cfg.IMAGE_DIR / "02_bipartite_colored_groups.png"
	pl.draw_bipartite_by_color(
		bipartite_graph,
		color_map=color_map_global,
		label_map=label_map_groups,
		top_n=None,
		output_path=colored_output,
		title=None,
		save=True,
		figsize=cfg.BIPARTITE_FIGSIZE,
		font_size=cfg.PLOT_FONT_SIZE,
		node_size_map=node_size_map_workers,
		factor_node_size=0.6,
		node_size_exponent=0.8,
	)
	print(f"Saved bipartite groups layout to {colored_output}")


if __name__ == "__main__":
	main()

from scripts import *
import pandas as pd


def main(enes_df=None, nodelist_caes_df=None, nodelist_ciuo_df=None):
	if enes_df is None:
		enes_df = pd.read_csv(ENES_PATH)
	if nodelist_caes_df is None:
		nodelist_caes_df = pd.read_csv(CAES_NODELIST_PATH, index_col=CAES_ID)
	if nodelist_ciuo_df is None:
		nodelist_ciuo_df = pd.read_csv(CIUO_NODELIST_PATH, index_col=CIUO_ID)

	bipartite_graph = gc.build_bipartite_graph(
		enes_df,
		CAES_ID,
		CIUO_ID,
		logscale=LOGSCALE,
	)

	caes_nodes = [
		node
		for node in bipartite_graph.nodes
		if bipartite_graph.nodes[node].get("bipartite") == utils.get_class_index("caes")
	]
	ciuo_nodes = [
		node
		for node in bipartite_graph.nodes
		if bipartite_graph.nodes[node].get("bipartite") == utils.get_class_index("ciuo")
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
	
	caes_color = pl.mean_color(nodelist_caes_df["caeslabel_color"].apply(utils.parse_color).tolist())
	ciuo_color = pl.mean_color(nodelist_ciuo_df["ciuolabel_color"].apply(utils.parse_color).tolist())

	color_map = {
		node: (
			caes_color
			if bipartite_graph.nodes[node].get("bipartite") == utils.get_class_index("caes")
			else ciuo_color
		)
		for node in bipartite_graph.nodes
	}

	colors_hist = {
		"all": pl.mean_color([caes_color, ciuo_color]),
		"caes": caes_color,
		"ciuo": ciuo_color,
	}
	degrees = gc.degree_sequences(bipartite_graph)
	degrees_output = IMAGE_DIR / "02_bipartite_degree_distribution.png"
	pl.plot_degree_histograms(degrees, output_path=degrees_output, colors=colors_hist, save=True)
	print(f"Saved degree distribution to {degrees_output}")

	caes_group_col = CAES_AG_OLD
	ciuo_group_col = CIUO_3CAT
	if caes_group_col not in nodelist_caes_df.columns:
		raise KeyError(f"Missing '{caes_group_col}' column in CAES node list.")
	if ciuo_group_col not in nodelist_ciuo_df.columns:
		raise KeyError(f"Missing '{ciuo_group_col}' column in CIUO node list.")

	caes_group_map = nodelist_caes_df[caes_group_col].to_dict()
	ciuo_group_map = nodelist_ciuo_df[ciuo_group_col].to_dict()

	# Use color columns from CSV files
	caes_color_col = "caesag_color"
	ciuo_color_col = "ciuo3cat_color"
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
		if bipartite_graph.nodes[node].get("bipartite") == utils.get_class_index("caes"):
			group = caes_group_map.get(node, "Unknown")
			color_map_global[node] = caes_palette.get(group, "gray")
		else:
			group = ciuo_group_map.get(node, "Unknown")
			color_map_global[node] = ciuo_palette.get(group, "gray")

	colored_output = IMAGE_DIR / "02_bipartite_colored_groups.png"
	pl.draw_bipartite_by_color(
		bipartite_graph,
		color_map=color_map_global,
		label_map=label_map_groups,
		top_n=None,
		output_path=colored_output,
		title="Bipartite network with CAES and CIUO groups",
		save=True,
		figsize=(10, 8)
	)
	print(f"Saved bipartite groups layout to {colored_output}")


if __name__ == "__main__":
	main()

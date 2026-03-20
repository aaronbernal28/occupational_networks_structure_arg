from scripts import *
import networkx as nx
import pandas as pd


def main(enes_df=None, nodelist_ciuo_df=None):
	if enes_df is None:
		enes_df = pd.read_csv(ENES_PATH)
	if nodelist_ciuo_df is None:
		nodelist_ciuo_df = pd.read_csv(CIUO_NODELIST_PATH, index_col=CIUO_ID)

	bipartite_graph = gc.build_bipartite_graph(
		enes_df,
		CAES_ID,
		CIUO_ID,
		logscale=LOGSCALE,
	)
	ciuo_projection = gc.generic_weighted_projected_graph(
		bipartite_graph, 
		target_partition=0,
		weight_function=gc.weighted_hidalgo_proximity_weight
	)
	density = nx.density(ciuo_projection)
	print(f"CIUO custom projection density: {density:.6f}")

	DATA_PROJECTION_GEPHI_PATH.mkdir(parents=True, exist_ok=True)
	gexf_output = DATA_PROJECTION_GEPHI_PATH / "06_ciuo_projection_custom.gexf"
	#nx.write_gexf(ciuo_projection, gexf_output)
	print(f"Saved CIUO custom projection graph to {gexf_output}")

	group_col = CIUO_3CAT
	if group_col not in nodelist_ciuo_df.columns:
		raise KeyError(f"Missing '{group_col}' column in CIUO node list.")

	group_map = nodelist_ciuo_df[group_col].to_dict()
	
	# Use color column from CSV file
	color_col = CIUO_3CAT_COLOR
	if color_col not in nodelist_ciuo_df.columns:
		raise KeyError(f"Missing '{color_col}' column in CIUO node list.")
	
	group_color_map = nodelist_ciuo_df.groupby(group_col)[color_col].first().apply(utils.parse_color).to_dict()

	IMAGE_DIR.mkdir(parents=True, exist_ok=True)
	output = IMAGE_DIR / "06_custom_ciuo_projection_by_group.png"

	pos = pl.plot_projection_by_group(
		ciuo_projection,
		group_map=group_map,
		group_color_map=group_color_map,
		title=None,
		legend_title="Categories",
		figsize=PROJECTION_FIGSIZE,
		font_size=PLOT_FONT_SIZE,
		output_path=output,
		save=True,
		method="energy",
	)
	nodelist_ciuo_df = dl.insert_positions(nodelist_ciuo_df, pos)
	dl.export_processed(nodelist_ciuo_df, DATA_PROCESSED_PATH, "nodelist_ciuo")
	print(f"Saved CIUO projection to {output}")

	communities_ciuo, modularity_ciuo, best_resolution = comm.best_louvain_partition_random(ciuo_projection, seed=45)
	num_communities = len(set(communities_ciuo.values()))
	print(f"CIUO modularity score: {modularity_ciuo:.4f}")
	print(f"Best resolution: {best_resolution:.3f}")
	print(f"Detected communities: {num_communities}")

	community_colors = {
		comm_id: COMMUNITY_COLORS_PALETTE[comm_id % len(COMMUNITY_COLORS_PALETTE)] for comm_id in set(communities_ciuo.values())
	}
	community_output = IMAGE_DIR / "06_custom_ciuo_louvain.png"
	_ = pl.plot_projection_by_group(
		ciuo_projection,
		group_map=communities_ciuo,
		group_color_map=community_colors,
		title=None,
		legend_title="Communities",
		figsize=PROJECTION_FIGSIZE,
		font_size=PLOT_FONT_SIZE,
		output_path=community_output,
		save=True,
		legend_label_fmt=lambda g: f"C{g}",
		pos=pos
	)
	print(f"Saved CIUO Louvain communities to {community_output}")
	nodelist_ciuo_df["community"] = nodelist_ciuo_df.index.map(communities_ciuo).fillna(-1).astype(int)
	dl.export_processed(nodelist_ciuo_df, DATA_PROCESSED_PATH, "nodelist_ciuo")
	print("Updated nodelist_ciuo.csv with community column.")

	group_col = CIUO_LETRA_OLD
	if group_col not in nodelist_ciuo_df.columns:
		raise KeyError(f"Missing '{group_col}' column in CIUO node list.")

	group_map = nodelist_ciuo_df[group_col].to_dict()

	# Use color column from CSV file
	color_col = CIUO_LETRA_COLOR
	if color_col not in nodelist_ciuo_df.columns:
		raise KeyError(f"Missing '{color_col}' column in CIUO node list.")
	
	group_color_map = nodelist_ciuo_df.groupby(group_col)[color_col].first().apply(utils.parse_color).to_dict()
	
	stacked_output = IMAGE_DIR / "06_custom_ciuo_community_distribution.png"
	pl.plot_stacked_by_group(
		nodelist_ciuo_df,
		group_col=group_col,
		community_map=communities_ciuo,
		title=None,
		output_path=stacked_output,
		group_color_map=group_color_map,
		figsize=STACKED_FIGSIZE,
		font_size=PLOT_FONT_SIZE,
		save=True,
		percentage=False
	)
	print(f"Saved CIUO community distribution to {stacked_output}")

	# Gradient plots: % women and mean age
	features = nc.compute_group_characteristics(enes_df, group_col=CIUO_ID)
	female_map = features["female_pct"].dropna().to_dict()
	age_map = features["age_mean"].dropna().to_dict()
	pub_sector_map = features["public_sector_pct"].dropna().to_dict() if "public_sector_pct" in features else {}

	female_output = IMAGE_DIR / "06_ciuo_female_pct.png"
	pl.plot_projection_gradient(
		ciuo_projection,
		pos=pos,
		node_values=female_map,
		title=None,
		colorbar_label="% Women",
		cmap="coolwarm",
		figsize=PROJECTION_FIGSIZE,
		font_size=PLOT_FONT_SIZE,
		output_path=female_output,
		save=True,
	)
	print(f"Saved CIUO female-pct gradient to {female_output}")

	age_output = IMAGE_DIR / "06_ciuo_age_mean.png"
	pl.plot_projection_gradient(
		ciuo_projection,
		pos=pos,
		node_values=age_map,
		title=None,
		colorbar_label="Mean age (years)",
		cmap="Dark2",
		figsize=PROJECTION_FIGSIZE,
		font_size=PLOT_FONT_SIZE,
		output_path=age_output,
		save=True,
	)
	print(f"Saved CIUO age-mean gradient to {age_output}")
	
	if pub_sector_map:
		pub_sector_output = IMAGE_DIR / "06_ciuo_public_sector_pct.png"
		pl.plot_projection_gradient(
			ciuo_projection,
			pos=pos,
			node_values=pub_sector_map,
			title=None,
			colorbar_label="% Public Sector",
			cmap="PRGn",
			figsize=PROJECTION_FIGSIZE,
			font_size=PLOT_FONT_SIZE,
			output_path=pub_sector_output,
			save=True,
		)
		print(f"Saved CIUO public-sector-pct gradient to {pub_sector_output}")


if __name__ == "__main__":
	main()

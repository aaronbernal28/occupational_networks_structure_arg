import config as cfg
import networkx as nx
import pandas as pd
import src.communities as comm
import src.data_loader as dl
import src.graph_construction as gc
import src.node_characteristics as nc
import src.plotting as pl
import src.utils as utils


def main(enes_df=None, nodelist_ciuo_df=None):
	caes_id = cfg.DATA_ENES_PISAC["col_caes_id"]
	ciuo_id = cfg.DATA_ENES_PISAC["col_ciuo_id"]
	ciuo_3cat = cfg.DATA_NODELIST_CIUO["col_3cat"]
	ciuo_3cat_color = cfg.DATA_NODELIST_CIUO["col_3cat_color"]
	ciuo_letra = cfg.DATA_NODELIST_CIUO["col_letra"]
	ciuo_letra_color = cfg.DATA_NODELIST_CIUO["col_letra_color"]

	enes_path = cfg.DATA_PROCESSED_PATH / "base_enespersonas.csv"
	ciuo_nodelist_path = cfg.DATA_PROCESSED_PATH / "nodelist_ciuo.csv"
	gephi_path = cfg.DATA_RAW_PATH.parent / "gephi"

	if enes_df is None:
		enes_df = pd.read_csv(enes_path)
	if nodelist_ciuo_df is None:
		nodelist_ciuo_df = pd.read_csv(ciuo_nodelist_path, index_col=ciuo_id)

	bipartite_graph = gc.build_bipartite_graph(
		enes_df,
		caes_id,
		ciuo_id,
		logscale=cfg.LOGSCALE,
	)
	ciuo_projection = gc.generic_weighted_projected_graph(
		bipartite_graph, 
		target_partition=0,
		weight_function=gc.weighted_hidalgo_proximity_weight
	)
	density = nx.density(ciuo_projection)
	print(f"CIUO custom projection density: {density:.6f}")

	gephi_path.mkdir(parents=True, exist_ok=True)
	gexf_output = gephi_path / "06_ciuo_projection_custom.gexf"
	#nx.write_gexf(ciuo_projection, gexf_output)
	print(f"Saved CIUO custom projection graph to {gexf_output}")

	group_col = ciuo_3cat
	if group_col not in nodelist_ciuo_df.columns:
		raise KeyError(f"Missing '{group_col}' column in CIUO node list.")

	group_map = nodelist_ciuo_df[group_col].to_dict()
	
	# Use color column from CSV file
	color_col = ciuo_3cat_color
	if color_col not in nodelist_ciuo_df.columns:
		raise KeyError(f"Missing '{color_col}' column in CIUO node list.")
	
	group_color_map = nodelist_ciuo_df.groupby(group_col)[color_col].first().apply(utils.parse_color).to_dict()

	cfg.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
	output = cfg.IMAGE_DIR / "06_custom_ciuo_projection_by_group.png"

	pos = pl.plot_projection_by_group(
		ciuo_projection,
		group_map=group_map,
		group_color_map=group_color_map,
		title=None,
		legend_title="Categories",
		figsize=cfg.PROJECTION_FIGSIZE,
		font_size=cfg.PLOT_FONT_SIZE,
		output_path=output,
		save=True,
		method="energy",
		factor_node_size=0.5,
		node_size_exponent=0.9,
	)
	nodelist_ciuo_df = dl.insert_positions(nodelist_ciuo_df, pos)
	dl.export_processed(nodelist_ciuo_df, cfg.DATA_PROCESSED_PATH, "nodelist_ciuo")
	print(f"Saved CIUO projection to {output}")

	communities_ciuo, modularity_ciuo, best_resolution = comm.best_louvain_partition_random(ciuo_projection, seed=45)
	num_communities = len(set(communities_ciuo.values()))
	print(f"CIUO modularity score: {modularity_ciuo:.4f}")
	print(f"Best resolution: {best_resolution:.3f}")
	print(f"Detected communities: {num_communities}")

	community_colors = {
		comm_id: cfg.COMMUNITY_COLORS_PALETTE[comm_id % len(cfg.COMMUNITY_COLORS_PALETTE)] for comm_id in set(communities_ciuo.values())
	}
	community_output = cfg.IMAGE_DIR / "06_custom_ciuo_louvain.png"
	_ = pl.plot_projection_by_group(
		ciuo_projection,
		group_map=communities_ciuo,
		group_color_map=community_colors,
		title=None,
		legend_title="Communities",
		figsize=cfg.PROJECTION_FIGSIZE,
		font_size=cfg.PLOT_FONT_SIZE,
		output_path=community_output,
		save=True,
		legend_label_fmt=lambda g: f"C{g}",
		pos=pos,
		factor_node_size=0.5,
		node_size_exponent=0.9,
	)
	print(f"Saved CIUO Louvain communities to {community_output}")
	nodelist_ciuo_df["community"] = nodelist_ciuo_df.index.map(communities_ciuo).fillna(-1).astype(int)
	dl.export_processed(nodelist_ciuo_df, cfg.DATA_PROCESSED_PATH, "nodelist_ciuo")
	print("Updated nodelist_ciuo.csv with community column.")

	group_col = ciuo_letra
	if group_col not in nodelist_ciuo_df.columns:
		raise KeyError(f"Missing '{group_col}' column in CIUO node list.")

	group_map = nodelist_ciuo_df[group_col].to_dict()

	# Use color column from CSV file
	color_col = ciuo_letra_color
	if color_col not in nodelist_ciuo_df.columns:
		raise KeyError(f"Missing '{color_col}' column in CIUO node list.")
	
	group_color_map = nodelist_ciuo_df.groupby(group_col)[color_col].first().apply(utils.parse_color).to_dict()
	
	stacked_output = cfg.IMAGE_DIR / "06_custom_ciuo_community_distribution.png"
	pl.plot_stacked_by_group(
		nodelist_ciuo_df,
		group_col=group_col,
		community_map=communities_ciuo,
		title=None,
		output_path=stacked_output,
		group_color_map=group_color_map,
		figsize=cfg.STACKED_FIGSIZE,
		font_size=cfg.PLOT_FONT_SIZE,
		save=True,
		percentage=False
	)
	print(f"Saved CIUO community distribution to {stacked_output}")

	# Gradient plots: % women and mean age
	features = nc.compute_group_characteristics(enes_df, group_col=ciuo_id)
	female_map = features["female_pct"].dropna().to_dict()
	age_map = features["age_mean"].dropna().to_dict()
	pub_sector_map = features["public_sector_pct"].dropna().to_dict() if "public_sector_pct" in features else {}

	female_output = cfg.IMAGE_DIR / "06_ciuo_female_pct.png"
	pl.plot_projection_gradient(
		ciuo_projection,
		pos=pos,
		node_values=female_map,
		title=None,
		colorbar_label="% Women",
		cmap="coolwarm",
		figsize=cfg.PROJECTION_FIGSIZE,
		font_size=cfg.PLOT_FONT_SIZE,
		output_path=female_output,
		save=True,
		factor_node_size=0.5,
		node_size_exponent=0.9,
	)
	print(f"Saved CIUO female-pct gradient to {female_output}")

	age_output = cfg.IMAGE_DIR / "06_ciuo_age_mean.png"
	pl.plot_projection_gradient(
		ciuo_projection,
		pos=pos,
		node_values=age_map,
		title=None,
		colorbar_label="Mean age (years)",
		cmap="Dark2",
		figsize=cfg.PROJECTION_FIGSIZE,
		font_size=cfg.PLOT_FONT_SIZE,
		output_path=age_output,
		save=True,
		factor_node_size=0.5,
		node_size_exponent=0.9,
	)
	print(f"Saved CIUO age-mean gradient to {age_output}")
	
	if pub_sector_map:
		pub_sector_output = cfg.IMAGE_DIR / "06_ciuo_public_sector_pct.png"
		pl.plot_projection_gradient(
			ciuo_projection,
			pos=pos,
			node_values=pub_sector_map,
			title=None,
			colorbar_label="% Public Sector",
			cmap="PRGn",
			figsize=cfg.PROJECTION_FIGSIZE,
			font_size=cfg.PLOT_FONT_SIZE,
			output_path=pub_sector_output,
			save=True,
			factor_node_size=0.5,
			node_size_exponent=0.9,
		)
		print(f"Saved CIUO public-sector-pct gradient to {pub_sector_output}")


if __name__ == "__main__":
	main()

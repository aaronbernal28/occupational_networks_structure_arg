from scripts import *
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
	

def compute_and_plot_edge_correlation(G: nx.Graph, feature_map: dict, color_map: dict, title: str, output_path: Path, save: bool = True):
	# Only keep nodes that have the feature
	valid_nodes = set(feature_map.keys())
	
	x_vals_map = {}
	y_vals_map = {u: [] for u in valid_nodes}
	edge_weights = {u: [] for u in valid_nodes}
	
	for u, v, data in G.edges(data=True):
		if u in valid_nodes and v in valid_nodes:
			# Undirected, we add both (u,v) and (v,u) to make it symmetric
			w = data.get("weight", 0.0)
			
			x_vals_map[u] = feature_map[u]
			y_vals_map[u].append(feature_map[v])
			edge_weights[u].append(w)
			
			x_vals_map[v] = feature_map[v]
			y_vals_map[v].append(feature_map[u])
			edge_weights[v].append(w)

	x_vals = []
	y_vals = []
	for u in x_vals_map:
		# For each node, we take its feature value and the average of its neighbors' feature values (weighted if desired)
		if not y_vals_map[u]:
			continue
		x_vals.append(x_vals_map[u])
		y_vals.append(np.average(y_vals_map[u], weights=edge_weights[u]))

	if len(x_vals) < 2:
		print(f"Warning: Not enough valid points to compute correlation for {title}.")
		return

	# Plot
	plt.figure(figsize=(9, 8))
	
	# Scatter plot of node feature vs average neighbor feature, colored by community
	sns.scatterplot(x=x_vals, y=y_vals, alpha=0.7, edgecolor="black", c=[color_map.get(u, "gray") for u in x_vals_map.keys()])
	plt.plot([0, 100], [0, 100], "k--", label="y=x (Perfect Assortativity)", alpha=0.5)
	
	# Add a legend for the communities
	unique_colors = set(color_map.values()) - {"gray"}  # Exclude gray if it's used for unassigned nodes
	for color in COMMUNITY_COLORS_PALETTE:
		if color in unique_colors:
			plt.scatter([], [], c=color, label=f"Community {color}", edgecolor="black")
	
	# Add regression line on top to show trend
	sns.regplot(x=x_vals, y=y_vals, scatter=False, color="red", line_kws={"linestyle": "--", "alpha": 0.5}, label="Trend", x_ci="ci")

	# Compute correlation
	pearson_r, p_value = stats.pearsonr(x_vals, y_vals)

	print(f"--- {title} ---")
	print(f"Assortativity Coefficient (Pearson r): {pearson_r:.4f} (p-value: {p_value:.4e})")
	if pearson_r > 0 and p_value < 0.05:
		print("Positive correlation: Gender homophily exists. Occupations with similar gender compositions cluster together.")
	elif pearson_r < 0 and p_value < 0.05:
		print("Negative correlation: Heterophily exists. Occupations connect primarily to those of opposite gender compositions.")
	else:
		print("Zero correlation: Gender is randomly distributed across the network structure.")

	plt.title(f"{title}\nAssortativity (Pearson r): {pearson_r:.3f}")
	plt.xlabel(f"% Women per node (Node i)")
	plt.ylabel(f"% Women per weighted average of neighbors (Node i)")
	#plt.xlim(0, 100)
	#plt.ylim(0, 100)
	sns.despine()
	plt.legend()

	# Generate Assortativity Scatter Plot (Node value vs Average Neighbor Value)
	plt.tight_layout()
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()

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
		class_name="ciuo",
		weight_function=gc.weighted_hidalgo_proximity_weight
	)
	
	IMAGE_DIR.mkdir(parents=True, exist_ok=True)
	compute_and_plot_edge_correlation(
		G=ciuo_projection,
		feature_map=feature_map,
		color_map=color_map,
		title="Gender Correlation (CIUO Network)", 
		output_path=IMAGE_DIR / "14_ciuo_edge_correlation.png", 
		save=True
	)

if __name__ == "__main__":
	main()

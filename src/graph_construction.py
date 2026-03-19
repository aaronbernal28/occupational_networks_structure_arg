"""
Graph construction helpers for bipartite and projected networks.
"""
from typing import Dict, Iterable, Set, Tuple
import src.utils as ut
import networkx as nx
import pandas as pd
import numpy as np


def build_bipartite_graph(enes_df: pd.DataFrame, caes_id: str, ciuo_id: str, logscale=True) -> nx.Graph:
	"""
	Build the bipartite graph from the merged ENES dataframe.
	"""
	# Define node sets
	caes_nodes = set(enes_df[caes_id].unique())
	ciuo_nodes = set(enes_df[ciuo_id].unique())

	assert caes_nodes & ciuo_nodes == set(), "CAES and CIUO IDs must be disjoint."

	# Build bipartite graph
	graph = nx.Graph()
	graph.add_nodes_from(caes_nodes, bipartite=ut.get_class_index(caes_id))
	graph.add_nodes_from(ciuo_nodes, bipartite=ut.get_class_index(ciuo_id))

	edges = (
		enes_df.groupby([caes_id, ciuo_id])
		.size()
		.apply(lambda x: x if not logscale else float(np.log1p(x)))
		.reset_index(name="weight")
	)

	# Add edges with weights using vectorization
	graph.add_weighted_edges_from(edges[[caes_id, ciuo_id, "weight"]].itertuples(index=False, name=None))

	assert nx.is_bipartite(graph), "Constructed graph is not bipartite."
	return graph


def projected_graph(graph: nx.Graph, class_name: str) -> nx.Graph:
	"""Unweighted projection onto caes or ciuo node sets."""
	assert class_name.lower() in ["caes", "ciuo"], "class_name must be 'caes' or 'ciuo'."
	nodes = [node for node in graph.nodes if graph.nodes[node].get("bipartite") == ut.get_class_index(class_name)]
	return nx.bipartite.projected_graph(graph, nodes)


def weighted_projected_graph(graph: nx.Graph, class_name: str) -> nx.Graph:
	"""Weighted projection onto the provided node set."""
	assert class_name.lower() in ["caes", "ciuo"], "class_name must be 'caes' or 'ciuo'."
	nodes = [node for node in graph.nodes if graph.nodes[node].get("bipartite") == ut.get_class_index(class_name)]
	return nx.bipartite.weighted_projected_graph(graph, nodes)


def generic_weighted_projected_graph(graph: nx.Graph, class_name: str, weight_function = None) -> nx.Graph:
	"""Weighted projection onto the provided node set using a custom weight function."""
	assert class_name.lower() in ["caes", "ciuo"], "class_name must be 'caes' or 'ciuo'."
	nodes = [node for node in graph.nodes if graph.nodes[node].get("bipartite") == ut.get_class_index(class_name)]
	return nx.bipartite.generic_weighted_projected_graph(graph, nodes, weight_function)


def hidalgo_proximity_weight(G: nx.Graph, u: int, v: int) -> float:
	"""Hidalgo, C. A., Klinger, B., Barabási, A. L., & Hausmann, R. (2007). The product space conditions the development of nations."""
	# Get the shared neighbors
	shared_features_len = len(set(G[u]).intersection(G[v]))
	
	# If no overlap, return 0 to save time
	if shared_features_len == 0:
		return 0
	
	# Get degrees
	degree_u = G.degree[u]
	degree_v = G.degree[v]
	
	# Calculate Conditional Probabilities
	prob_u_given_v = shared_features_len / degree_v
	prob_v_given_u = shared_features_len / degree_u
	
	# Return the Minimum (The Hidalgo/Hausmann standard)
	return min(prob_u_given_v, prob_v_given_u)


def dot_product_weight(G: nx.Graph, u: int, v: int) -> float:
	"""Newman, M. E. J. (2001). Scientific collaboration networks. II. Shortest paths, weighted networks, and centrality.
	Zhou, T., Ren, J., Medo, M., & Zhang, Y. C. (2007). Bipartite network projection and personal recommendation.
	"""
	shared_nodes = set(G[u]).intersection(G[v])
	return sum(G[u][node].get('weight', 1) * G[v][node].get('weight', 1) for node in shared_nodes)


def cosine_similarity_weight(G: nx.Graph, u: int, v: int) -> float:
	norm_weight_u = 0
	norm_weight_v = 0
	shared_nodes = set(G[u]) & set(G[v])

	for node in shared_nodes:
		# Get weights of edges (u, node) and (v, node)
		w_u = G[u][node].get('weight', 1) 
		w_v = G[v][node].get('weight', 1)
		
		norm_weight_u += w_u ** 2
		norm_weight_v += w_v ** 2

	return dot_product_weight(G, u, v) / (np.sqrt(norm_weight_u) * np.sqrt(norm_weight_v)) if norm_weight_u > 0 and norm_weight_v > 0 else 0


def weighted_hidalgo_proximity_weight(G: nx.Graph, u: int, v: int, weight: str = "weight") -> float:
	"""
	Calculates the 'Weighted Hidalgo' proximity (Minimum Conditional Probability) preserving intensity.
	"""
	# Get the shared neighbors
	# Note: For very dense graphs, iterating the smaller neighborhood is faster
	nb_u = set(G[u])
	nb_v = set(G[v])
	shared_neighbors = list(nb_u & nb_v)
	
	# If no overlap, return 0 to save time
	if len(shared_neighbors) == 0:
		return 0.0
	
	# 1. Calculate Weighted Overlap (Dot Product)
	# Sum of (weight_u_k * weight_v_k) for all shared neighbors k
	dot_product = sum(
		G[u][k].get(weight, 1.0) * G[v][k].get(weight, 1.0) 
		for k in shared_neighbors
	)
	
	# 2. Calculate "Weighted Degrees" (Squared Norms)
	# We must use squared weights so the denominator matches the scale of the numerator (weights * weights)
	# This ensures the probability never exceeds 1.0.
	norm_sq_u = sum(d.get(weight, 1.0) ** 2 for _, d in G[u].items())
	norm_sq_v = sum(d.get(weight, 1.0) ** 2 for _, d in G[v].items())
	
	# Avoid division by zero
	if norm_sq_u == 0 or norm_sq_v == 0:
		return 0.0

	# 3. Calculate Conditional Probabilities (Weighted)
	# "Given v, how much of its total 'energy' overlaps with u?"
	prob_u_given_v = dot_product / norm_sq_v
	prob_v_given_u = dot_product / norm_sq_u
	
	# Return the Minimum (The Hidalgo/Hausmann standard)
	return min(prob_u_given_v, prob_v_given_u)


def degree_sequences(graph: nx.Graph) -> Dict[str, list]:
	"""Return degree lists for all nodes and each partition."""
	degrees_all = list(dict(graph.degree()).values())
	degrees_caes = [graph.degree(node) for node in graph.nodes if graph.nodes[node].get("bipartite") == ut.get_class_index("caes")]
	degrees_ciuo = [graph.degree(node) for node in graph.nodes if graph.nodes[node].get("bipartite") == ut.get_class_index("ciuo")]
	return {
		"all": degrees_all,
		"caes": degrees_caes,
		"ciuo": degrees_ciuo,
	}


def disparity_filter_backbone(
	graph: nx.Graph,
	alpha: float = 0.05,
	mode: str = "or",
	keep_isolates: bool = False,
) -> nx.Graph:
	"""Return a disparity-filter backbone from a weighted undirected graph.

	Parameters
	----------
	graph: Weighted undirected projection graph.
	alpha: Significance threshold. Lower values retain fewer, stronger/significant edges.
	mode:
		"or" keeps an edge if significant for at least one endpoint. (Serrano 2009)
		"and" keeps an edge only if significant for both endpoints.
	keep_isolates: If True, keep all original nodes even if no edge survives.
	"""
	if alpha <= 0 or alpha >= 1:
		raise ValueError("alpha must be in the open interval (0, 1).")
	if mode not in {"or", "and"}:
		raise ValueError("mode must be either 'or' or 'and'.")
	if graph.is_directed():
		raise ValueError("disparity_filter_backbone expects an undirected graph.")

	strengths = {
		node: sum(data.get("weight", 1.0) for _, _, data in graph.edges(node, data=True))
		for node in graph.nodes()
	}
	degrees = dict(graph.degree())

	backbone = nx.Graph()
	if keep_isolates:
		backbone.add_nodes_from(graph.nodes(data=True))

	for u, v, data in graph.edges(data=True):
		weight = float(data.get("weight", 1.0))
		if weight <= 0:
			continue

		def edge_alpha(node: int, node_degree: int, node_strength: float) -> float:
			if node_degree <= 1 or node_strength <= 0:
				return 0.0
			p_ij = weight / node_strength
			p_ij = min(max(p_ij, 0.0), 1.0)
			return (1.0 - p_ij) ** (node_degree - 1)

		alpha_u = edge_alpha(u, degrees.get(u, 0), strengths.get(u, 0.0))
		alpha_v = edge_alpha(v, degrees.get(v, 0), strengths.get(v, 0.0))

		keep_edge = (alpha_u < alpha or alpha_v < alpha) if mode == "or" else (alpha_u < alpha and alpha_v < alpha)
		if keep_edge:
			backbone.add_edge(u, v, **data)

	return backbone

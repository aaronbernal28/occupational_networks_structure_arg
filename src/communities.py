"""
Community detection utilities.
"""
from typing import Dict, List, Tuple

import networkx as nx
from networkx.algorithms.community import louvain_communities, girvan_newman, modularity
import numpy as np


def louvain_partition(graph: nx.Graph, resolution: float = 1.0, seed: int = 28) -> Tuple[Dict[int, int], float]:
	"""
	Run Louvain and return the partition map plus modularity.
	"""
	communities_list = louvain_communities(graph, weight="weight", resolution=resolution, seed=seed)
	
	# Convert to dict format: node -> community_id
	communities = {node: i for i, comm in enumerate(communities_list) for node in comm}
	
	# Calculate modularity score
	score = modularity(graph, communities_list, weight="weight")
	return communities, score


def best_louvain_partition_search(graph: nx.Graph, seed: int = 28, max_iter: int = 8) -> Tuple[Dict[int, int], float, float]:
	"""
	Find the best Louvain partition by exploring resolution values in basic optimization approach.

	Args:
		graph: The graph to partition
		seed: Random seed for reproducibility
		max_iter: Maximum iterations per exploration phase (default 8)
	
	Returns:
		Tuple of (best_partition, best_modularity, best_resolution)
	"""
	base_resolution = 1.0
	step = 0.1
	
	# Evaluate base resolution
	base_partition, base_score = louvain_partition(graph, resolution=base_resolution, seed=seed)
	best_partition = base_partition
	best_score = base_score
	best_resolution = base_resolution
	
	# Explore both directions
	lower_res = base_resolution - step
	upper_res = base_resolution + step
	
	lower_partition, lower_score = louvain_partition(graph, resolution=lower_res, seed=seed)
	upper_partition, upper_score = louvain_partition(graph, resolution=upper_res, seed=seed)
	
	# Determine best direction (-1, 0, +1)
	if lower_score > best_score and lower_score >= upper_score:
		direction = -1
		best_partition = lower_partition
		best_score = lower_score
		best_resolution = lower_res
	elif upper_score > best_score:
		direction = 1
		best_partition = upper_partition
		best_score = upper_score
		best_resolution = upper_res
	else:
		direction = 0
	
	# Continue in the chosen direction with 0.1 steps
	if direction != 0:
		current_resolution = best_resolution
		iter_count = 0
		while iter_count < max_iter or next_resolution <= 0.1:
			next_resolution = current_resolution + (direction * step)
			if next_resolution <= 0:
				break
			
			next_partition, next_score = louvain_partition(graph, resolution=next_resolution, seed=seed)
			
			if next_score > best_score:
				best_partition = next_partition
				best_score = next_score
				best_resolution = next_resolution
				current_resolution = next_resolution
				iter_count += 1
			else:
				# Modularity decreased, switch to finer steps
				break
	
	# Refine with 0.02 steps
	fine_step = 0.02
	current_resolution = best_resolution
	
	# Try the opposite direction
	fine_direction = -direction

	if fine_direction != 0:
		temp_resolution = current_resolution
		iter_count = 0
		while iter_count < max_iter or next_resolution <= 0.02:
			next_resolution = temp_resolution + (fine_direction * fine_step)
			if next_resolution <= 0:
				break
			
			next_partition, next_score = louvain_partition(graph, resolution=next_resolution, seed=seed)
			
			if next_score > best_score:
				best_partition = next_partition
				best_score = next_score
				best_resolution = next_resolution
				temp_resolution = next_resolution
				iter_count += 1
			else:
				# Modularity decreased in this direction, stop
				break
	
	return best_partition, best_score, best_resolution


def best_louvain_partition_random(graph: nx.Graph, seed: int = 28, n_samples: int = 20, 
								   min_resolution: float = 0.5, max_resolution: float = 5.0) -> Tuple[Dict[int, int], float, float]:
	"""
	Find the best Louvain partition by random sampling of resolution values.
	Args:
		graph: The graph to partition
		seed: Random seed for reproducibility
		n_samples: Number of random resolution values to sample (default 20)
		min_resolution: Minimum resolution value (default 0.5)
		max_resolution: Maximum resolution value (default 5.0)
	
	Returns:
		Tuple of (best_partition, best_modularity, best_resolution)
	"""
	np.random.seed(seed)
	
	# Sample resolution values uniformly
	resolutions = np.random.uniform(min_resolution, max_resolution, n_samples)
	
	best_partition = None
	best_score = -1.0
	best_resolution = None
	
	for resolution in resolutions:
		seed += 1  # Increment seed for variability
		partition, score = louvain_partition(graph, resolution=resolution, seed=seed)
		
		if score > best_score:
			best_partition = partition
			best_score = score
			best_resolution = resolution
	
	return best_partition, best_score, best_resolution


def girvan_newman_partition(graph: nx.Graph, max_levels: int = 20, ) -> Tuple[Dict[int, int], float]:
	"""
	Run Girvan-Newman and return the best partition up to max_levels plus modularity.
	"""
	return None


def best_partition(graph: nx.Graph, algorithm: any, parameters: List[Dict[str, any]]) -> Tuple[Dict[int, int], float]:
	"""
	Compute the best partition using the provided community detection algorithm.
	"""
	best_score = -1.0
	best_partition = None
	for params in parameters:
		partition, score = algorithm(graph, **params)
		if score > best_score:
			best_score = score
			best_partition = partition
	return best_partition, best_score
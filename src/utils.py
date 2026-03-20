import ast
from functools import lru_cache

@lru_cache(maxsize=1000)
def parse_color_from_string(color_str: str):
	parsed = ast.literal_eval(color_str)
	if hasattr(parsed, '__iter__') and not isinstance(parsed, str):
		return tuple(float(x) for x in parsed)
	return parsed

def parse_color(color_value):
	"""Parse color value from CSV (can be string tuple or already a tuple)."""
	if isinstance(color_value, str):
		return parse_color_from_string(color_value)
	# If it's already a tuple/list/array, ensure it's a regular tuple
	if hasattr(color_value, '__iter__') and not isinstance(color_value, str):
		return tuple(float(x) for x in color_value)
	return color_value


def get_class_index(col_name: str) -> int:
	"""Determine if the column corresponds to CAES or CIUO IDs for bipartite graph construction."""
	return int("caes" in col_name.lower())


def original_ciuo_id(id: int, max_caes_id: int) -> int:
	"""
	Recover original CIUO ID from disambiguated ID.
	"""
	return id - max_caes_id


def desambiated_caes_id(id: int) -> int:
	return id


def desambiated_ciuo_id(id: int, max_caes_id: int) -> int:
	"""
	Recover original CIUO ID from disambiguated ID.
	"""
	return id + max_caes_id

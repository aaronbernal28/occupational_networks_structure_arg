"""
Data loading and cleaning utilities for the ENES occupational network analysis.
"""

from pathlib import Path
from typing import Dict
import src.utils as ut
import pandas as pd
from src.plotting import (
	color_map_caes,
	color_map_ciuo,
	color_letra_map_caes,
	color_1digit_map_ciuo,
	color_agrupation_map_caes,
	color_ciuo3cat_map_ciuo
)


def load_enes_base(enes_path: Path, caes_id: str, ciuo_id: str) -> pd.DataFrame:
	"""Load and clean the base ENES data (person-level responses)."""
	try:
		# Try with semicolon first (original format)
		enes_df = pd.read_csv(enes_path, sep=';')
		if caes_id not in enes_df.columns:
			# If fails, try with comma (2021 format)
			enes_df = pd.read_csv(enes_path, sep=',')
	except Exception:
		enes_df = pd.read_csv(enes_path, sep=',')

	enes_df = enes_df.dropna(subset=[ciuo_id, caes_id])   
	enes_df[caes_id] = enes_df[caes_id].astype(int)
	enes_df[ciuo_id] = enes_df[ciuo_id].astype(int)

	enes_df[caes_id] = enes_df[caes_id].apply(lambda x: ut.desambiated_caes_id(x))
	enes_df[ciuo_id] = enes_df[ciuo_id].apply(lambda x: ut.desambiated_ciuo_id(x))
	return enes_df


def load_nodelist_caes(caes_path: Path, caes_id: str) -> pd.DataFrame:
	"""Load CAES node metadata and normalize labels."""
	caes_df = pd.read_csv(caes_path)
	caes_df[caes_id] = caes_df[caes_id].astype(int)
	caes_df[caes_id] = caes_df[caes_id].apply(lambda x: ut.desambiated_caes_id(x))
	caes_df = caes_df.set_index(caes_id)
	caes_df["caesletra"] = caes_df["caesletra"].apply(lambda x: x.split(";")[0])

	color_map = color_map_caes(caes_df.index.to_list())
	caes_df['caeslabel_color'] = caes_df.index.map(color_map)
	
	color_map = color_letra_map_caes(caes_df)
	caes_df['caesletra_color'] = caes_df['caesletra'].map(color_map)

	color_map = color_agrupation_map_caes(caes_df)
	caes_df['caesag_color'] = caes_df['caesag'].map(color_map)


	caes_df["caesletra_old"] = caes_df["caesletra"]
	caes_df["caesag_old"] = caes_df["caesag"]
	caes_df["caesletra"] = caes_df["caesletra"].apply(lambda x: x.split(".")[0])
	caes_df["caesag"] = caes_df["caesag"].apply(lambda x: x.split(".")[0])
	return caes_df


def load_nodelist_ciuo(ciuo_path: Path, ciuo_id: str) -> pd.DataFrame:
	"""Load CIUO node metadata and normalize labels."""
	ciuo_df = pd.read_csv(ciuo_path)
	ciuo_df[ciuo_id] = ciuo_df[ciuo_id].astype(int)
	ciuo_df[ciuo_id] = ciuo_df[ciuo_id].apply(lambda x: ut.desambiated_ciuo_id(x))
	ciuo_df = ciuo_df.set_index(ciuo_id)

	color_map = color_map_ciuo(ciuo_df.index.to_list())
	ciuo_df['ciuolabel_color'] = ciuo_df.index.map(color_map)

	color_map = color_1digit_map_ciuo(ciuo_df)
	ciuo_df['ciuo1diglabel_color'] = ciuo_df['ciuo1diglabel'].map(color_map)

	color_map = color_ciuo3cat_map_ciuo(ciuo_df)
	ciuo_df['ciuo3cat_color'] = ciuo_df['ciuo3cat'].map(color_map)

	ciuo_df["ciuo1diglabel_old"] = ciuo_df["ciuo1diglabel"]
	ciuo_df["ciuo1diglabel"] = ciuo_df["ciuo1diglabel"].apply(lambda x: x.split(".")[0])
	return ciuo_df


def merge_enes_with_metadata(enes_df: pd.DataFrame, caes_df: pd.DataFrame, ciuo_df: pd.DataFrame, caes_id: str, ciuo_id: str) -> pd.DataFrame:
	"""Attach CAES and CIUO labels to the ENES responses."""
	merged = enes_df.merge(caes_df, left_on=caes_id, right_index=True, how="inner")
	merged = merged.merge(ciuo_df, left_on=ciuo_id, right_index=True, how="inner")
	return merged


def load_dataset(enes_path: Path, caes_path: Path, ciuo_path: Path, caes_id: str, ciuo_id: str) -> Dict[str, pd.DataFrame]:
	"""
	Load all required dataframes and return them as a dict.
	"""
	
	if not enes_path.exists() or not caes_path.exists() or not ciuo_path.exists():
		raise FileNotFoundError(f"CSV files not found at {enes_path}, {caes_path}, or {ciuo_path}. Place the required files under data/raw/ or provide custom paths.")

	try:
		enes_raw = load_enes_base(enes_path, caes_id, ciuo_id)
		if "v188" in enes_raw.columns:
			enes_raw["sector_publico"] = (enes_raw["v188"] == 1)
		if "M3_7" in enes_raw.columns:
			enes_raw["sector_publico"] = (enes_raw["M3_7"] == 1)
		enes_raw = enes_raw[[col for col in [caes_id, ciuo_id, "v108", "v109", "ITI", "nivel_ed", "estado", "cat_ocup", "v206a", "f_calib3", "region", "sector_publico"] if col in enes_raw.columns]]
	except Exception as e:
		raise RuntimeError(f"Error loading ENES base data from {enes_path}: {e}") from e
	
	try:
		caes_df = load_nodelist_caes(caes_path, caes_id)
		ciuo_df = load_nodelist_ciuo(ciuo_path, ciuo_id)
	except Exception as e:
		raise RuntimeError(f"Error loading node list data from {caes_path} or {ciuo_path}: {e}") from e
	
	try:
		enes = merge_enes_with_metadata(enes_raw, caes_df, ciuo_df, caes_id, ciuo_id)
	except Exception as e:
		raise RuntimeError(f"Error merging ENES data with metadata: {e}") from e
	
	if enes.empty:
		raise ValueError("Merged ENES dataset is empty after joining with CAES and CIUO metadata. Check for mismatched IDs.")
	
	if set(enes[caes_id].unique()) & set(enes[ciuo_id].unique()) != set():
		raise ValueError("ID desambiguation failed: overlapping CAES and CIUO IDs detected.")

	if "encuesta" not in enes.columns:
		enes["encuesta"] = 2019  # Add default survey year if missing

	return {"enes": enes, "caes_nodes": caes_df, "ciuo_nodes": ciuo_df}

def export_processed(enes_df: pd.DataFrame, processed_path: Path, name: str) -> Path:
	"""
	Persist the merged dataset to CSV for reuse by scripts.
	"""
	processed_path.mkdir(parents=True, exist_ok=True)
	enes_df.to_csv(processed_path / f"{name}.csv", index=True)
	return processed_path / f"{name}.csv"


def insert_positions(nodelist_df: pd.DataFrame, positions: dict[str, list[float]]) -> pd.DataFrame:
	"""
	Insert precomputed positions into the node list dataframe for consistent plotting.
	"""
	pos_df = pd.DataFrame.from_dict(positions, orient="index", columns=["x", "y"])

	result = nodelist_df.drop(columns=["x", "y"], errors="ignore")
	return result.join(pos_df[["x", "y"]], how="left")

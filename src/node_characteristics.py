"""Utilities to compute descriptive characteristics for nodelists."""

from typing import Iterable

import pandas as pd


def _to_numeric(series: pd.Series) -> pd.Series:
	return pd.to_numeric(series, errors="coerce")


def compute_group_characteristics(
	enes_df: pd.DataFrame,
	group_col: str,
	age_col: str = "v108",
	sex_col: str = "v109",
	income_col: str = "ITI",
	education_col: str = "nivel_ed",
	status_col: str = "estado",
	category_col: str = "cat_ocup",
	hours_col: str = "v206a",
	weight_col: str = "f_calib3",
	region_col: str = "region",
	public_sector_col: str = "sector_publico",
) -> pd.DataFrame:
	"""Aggregate descriptive characteristics by group column."""
	if group_col not in enes_df.columns:
		raise KeyError(f"Missing '{group_col}' in ENES dataframe.")

	cols_to_keep = [group_col] + [
		col for col in [age_col, sex_col, income_col, education_col, status_col, category_col, hours_col, weight_col, region_col, public_sector_col]
		if col in enes_df.columns
	]
	data = enes_df[cols_to_keep].copy()

	for col in [age_col, sex_col, income_col, education_col, status_col, category_col, hours_col, weight_col, region_col, public_sector_col]:
		if col in data.columns:
			data[col] = _to_numeric(data[col])

	grouped = data.groupby(group_col, dropna=True)
	features = pd.DataFrame(index=grouped.size().index)
	features["n_obs"] = grouped.size()

	if region_col in data.columns:
		# Calculate percentage of workers in each region
		valid_region = data[data[region_col].notna()]
		valid_grouped = valid_region.groupby(group_col, dropna=True)
		for r in range(1, 9):
			features[f"region_{r}_pct"] = valid_grouped[region_col].apply(lambda s: (s == r).mean() * 100)

	if weight_col in data.columns:
		valid_weight = data[data[weight_col].notna()]
		valid_weight_grouped = valid_weight.groupby(group_col, dropna=True)
		features["total_workers_weighted"] = valid_weight_grouped[weight_col].sum()

	if age_col in data.columns:
		valid_age = data[data[age_col].notna()]
		valid_age_grouped = valid_age.groupby(group_col, dropna=True)
		features["age_mean"] = valid_age_grouped[age_col].mean()
		features["age_min"] = valid_age_grouped[age_col].min()
		features["age_q1"] = valid_age_grouped[age_col].quantile(0.25)
		features["age_median"] = valid_age_grouped[age_col].median()
		features["age_q3"] = valid_age_grouped[age_col].quantile(0.75)
		features["age_max"] = valid_age_grouped[age_col].max()
		features["age_std"] = valid_age_grouped[age_col].std()

	if income_col in data.columns:
		valid_income = data[data[income_col].notna()]
		valid_income_grouped = valid_income.groupby(group_col, dropna=True)
		features["income_mean"] = valid_income_grouped[income_col].mean()
		features["income_min"] = valid_income_grouped[income_col].min()
		features["income_q1"] = valid_income_grouped[income_col].quantile(0.25)
		features["income_median"] = valid_income_grouped[income_col].median()
		features["income_q3"] = valid_income_grouped[income_col].quantile(0.75)
		features["income_max"] = valid_income_grouped[income_col].max()
		features["income_std"] = valid_income_grouped[income_col].std()

	if hours_col in data.columns:
		valid_hours = data[data[hours_col].notna()]
		valid_hours_grouped = valid_hours.groupby(group_col, dropna=True)
		features["mean_hours_worked"] = valid_hours_grouped[hours_col].mean()

	if sex_col in data.columns:
		valid_sex = data[data[sex_col].isin([1, 2])]
		valid_grouped = valid_sex.groupby(group_col, dropna=True)
		features["female_pct"] = valid_grouped[sex_col].apply(lambda s: (s == 2).mean() * 100)
		features["male_pct"] = valid_grouped[sex_col].apply(lambda s: (s == 1).mean() * 100)

	if education_col in data.columns:
		# Calculate percentage with higher education
		valid_edu = data[data[education_col].notna()]
		valid_grouped = valid_edu.groupby(group_col, dropna=True)
		features["higher_education_pct"] = valid_grouped[education_col].apply(lambda s: (s >= 6).mean() * 100)

	if category_col in data.columns:
		# Assuming 3 is self-employed (cuenta propia) and 2 is salaried
		valid_cat = data[data[category_col].notna()]
		valid_grouped = valid_cat.groupby(group_col, dropna=True)
		features["self_employed_pct"] = valid_grouped[category_col].apply(lambda s: (s == 3).mean() * 100)
		features["salaried_pct"] = valid_grouped[category_col].apply(lambda s: (s == 2).mean() * 100)

	if public_sector_col in data.columns:
		valid_pub = data[data[public_sector_col].notna()]
		valid_grouped = valid_pub.groupby(group_col, dropna=True)
		features["public_sector_pct"] = valid_grouped[public_sector_col].apply(lambda s: (s == 1).mean() * 100)

	numeric_cols = features.select_dtypes(include=["number"]).columns
	features[numeric_cols] = features[numeric_cols].round(2)
	return features


def attach_group_characteristics(
	nodelist_df: pd.DataFrame,
	features_df: pd.DataFrame,
	keep_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
	"""Join computed characteristics onto a nodelist indexed by node id."""
	if keep_columns is not None:
		features_df = features_df[[col for col in keep_columns if col in features_df.columns]]

	result = nodelist_df.join(features_df, how="left")
	if "n_obs" in result.columns:
		result["n_obs"] = result["n_obs"].fillna(0).astype("Int64")
	return result

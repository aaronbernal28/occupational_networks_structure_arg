from pathlib import Path

import config as cfg
import pandas as pd
import src.data_loader as dl
import src.node_characteristics as nc
import src.plotting as pl

def main():
	caes_id = cfg.DATA_ENES_PISAC["col_caes_id"]
	ciuo_id = cfg.DATA_ENES_PISAC["col_ciuo_id"]
	base_sex_col = cfg.DATA_ENES_PISAC["col_sex_id"]
	base_public_col = cfg.DATA_ENES_PISAC["col_public_worker"]
	base_income_col = cfg.DATA_ENES_PISAC["col_total_income"]
	caes_letra = cfg.DATA_NODELIST_CAES["col_letra"]
	caes_ag = cfg.DATA_NODELIST_CAES["col_ag"]
	ciuo_letra = cfg.DATA_NODELIST_CIUO["col_letra"]
	ciuo_3cat = cfg.DATA_NODELIST_CIUO["col_3cat"]

	raw_enes_path = cfg.DATA_ENES_PISAC["source"]
	raw_caes_path = cfg.DATA_NODELIST_CAES["source"]
	raw_ciuo_path = cfg.DATA_NODELIST_CIUO["source"]
	raw_enes_2021_path = cfg.DATA_EXTRA[0]["source"] if cfg.DATA_EXTRA else None

	datasets = dl.load_dataset(
		raw_enes_path,
		raw_caes_path,
		raw_ciuo_path,
		caes_id,
		ciuo_id,
		cfg.MAX_CAES_ID,
		caes_letra,
		caes_ag,
		ciuo_letra,
		ciuo_3cat,
		cfg.DATA_NODELIST_CAES["col_label_color"],
		cfg.DATA_NODELIST_CAES["col_letra_color"],
		cfg.DATA_NODELIST_CAES["col_ag_color"],
		cfg.DATA_NODELIST_CIUO["col_label_color"],
		cfg.DATA_NODELIST_CIUO["col_letra_color"],
		cfg.DATA_NODELIST_CIUO["col_3cat_color"],
	)
	if raw_enes_2021_path is not None and Path(raw_enes_2021_path).exists():
		extra_cfg = cfg.DATA_EXTRA[0]
		extra_caes_id = extra_cfg["col_caes_id"]
		extra_ciuo_id = extra_cfg["col_ciuo_id"]
		extra_sex_col = extra_cfg["col_sex_id"]
		extra_public_col = extra_cfg["col_public_worker"]
		extra_income_col = extra_cfg["col_total_income"]

		extra_enes = dl.load_enes_base(
			raw_enes_2021_path,
			extra_caes_id,
			extra_ciuo_id,
			max_caes_id=cfg.MAX_CAES_ID,
		)
		rename_map = {
			extra_caes_id: caes_id,
			extra_ciuo_id: ciuo_id,
			extra_sex_col: base_sex_col,
			extra_public_col: base_public_col,
		}
		if extra_income_col is not None:
			rename_map[extra_income_col] = base_income_col
		extra_enes = extra_enes.rename(columns={k: v for k, v in rename_map.items() if k in extra_enes.columns})
		extra_enes["encuesta"] = 2021
		if base_public_col in extra_enes.columns:
			extra_enes["sector_publico"] = extra_enes[base_public_col] == 1

		extra_enes = dl.merge_enes_with_metadata(
			extra_enes,
			datasets["caes_nodes"],
			datasets["ciuo_nodes"],
			caes_id,
			ciuo_id,
		)
		datasets["enes"] = pd.concat([datasets["enes"], extra_enes], ignore_index=True, axis=0)

	caes_features = nc.compute_group_characteristics(datasets["enes"], caes_id)
	ciuo_features = nc.compute_group_characteristics(datasets["enes"], ciuo_id)

	datasets["caes_nodes"] = nc.attach_group_characteristics(datasets["caes_nodes"], caes_features)
	datasets["ciuo_nodes"] = nc.attach_group_characteristics(datasets["ciuo_nodes"], ciuo_features)

	output1 = dl.export_processed(datasets["enes"], cfg.DATA_PROCESSED_PATH, "base_enespersonas")
	output2 = dl.export_processed(datasets["caes_nodes"], cfg.DATA_PROCESSED_PATH, "nodelist_caes")
	output3 = dl.export_processed(datasets["ciuo_nodes"], cfg.DATA_PROCESSED_PATH, "nodelist_ciuo")
	print(f"Saved merged dataset to {output1}, {output2}, and {output3}")

	color_map_caes = datasets["caes_nodes"][cfg.DATA_NODELIST_CAES["col_label_color"]].to_dict()
	color_map_ciuo = datasets["ciuo_nodes"][cfg.DATA_NODELIST_CIUO["col_label_color"]].to_dict()

	# Generate exploratory data analysis plots
	print(f"Generating exploratory plots in {cfg.IMAGE_DIR}...")
	if "caeslabel" in datasets["caes_nodes"].columns:
		pl.plot_top_n_bar(
			df=datasets["caes_nodes"],
			label_col="caeslabel",
			val_col="n_obs",
			color_col=cfg.DATA_NODELIST_CAES["col_label_color"],
			title=None,
			xlabel="Total Workers",
			figsize=cfg.TOP_N_BAR_FIGSIZE,
			font_size=cfg.PLOT_FONT_SIZE,
			output_path=cfg.IMAGE_DIR / "00_top_caes_workers.png",
			top_n=15,
			save=True,
		)

	if "ciuolabel" in datasets["ciuo_nodes"].columns:
		pl.plot_top_n_bar(
			df=datasets["ciuo_nodes"],
			label_col="ciuolabel",
			val_col="n_obs",
			color_col=cfg.DATA_NODELIST_CIUO["col_label_color"],
			title=None,
			xlabel="Total Workers",
			figsize=cfg.TOP_N_BAR_FIGSIZE,
			font_size=cfg.PLOT_FONT_SIZE,
			output_path=cfg.IMAGE_DIR / "00_top_ciuo_workers.png",
			top_n=15,
			save=True,
		)

	print("Exploratory plots generated successfully.")

if __name__ == "__main__":
	main()
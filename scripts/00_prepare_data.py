from pathlib import Path

import config as cfg
import src.data_loader as dl
import src.node_characteristics as nc
import src.plotting as pl

def main():
	datasets = dl.load_dataset(
		cfg.DATA_ENES_PISAC,
		cfg.DATA_NODELIST_CAES,
		cfg.DATA_NODELIST_CIUO,
		cfg.DATA_EXTRA,
	)

	caes_id = cfg.DATA_ENES_PISAC["col_caes_id"]
	ciuo_id = cfg.DATA_ENES_PISAC["col_ciuo_id"]

	caes_features = nc.compute_group_characteristics(datasets["enes"], caes_id)
	ciuo_features = nc.compute_group_characteristics(datasets["enes"], ciuo_id)

	datasets["caes_nodes"] = nc.attach_group_characteristics(datasets["caes_nodes"], caes_features)
	datasets["ciuo_nodes"] = nc.attach_group_characteristics(datasets["ciuo_nodes"], ciuo_features)

	output1 = dl.export_processed(datasets["enes"], cfg.DATA_PROCESSED_PATH, "base_enespersonas")
	output2 = dl.export_processed(datasets["caes_nodes"], cfg.DATA_PROCESSED_PATH, "nodelist_caes")
	output3 = dl.export_processed(datasets["ciuo_nodes"], cfg.DATA_PROCESSED_PATH, "nodelist_ciuo")
	print(f"Saved merged dataset to {output1}, {output2}, and {output3}")

	# Generate exploratory data analysis plots
	print(f"Generating exploratory plots in {cfg.IMAGE_DIR}...")
	if "caeslabel" in datasets["caes_nodes"].columns:
		pl.plot_top_n_bar(
			df=datasets["caes_nodes"],
			label_col="caeslabel",
			val_col="n_obs",
			color_col=None,
			title=None,
			xlabel="Cantidad de trabajadores",
			figsize=cfg.TOP_N_BAR_FIGSIZE,
			font_size=cfg.PLOT_FONT_SIZE,
			output_path=cfg.IMAGE_DIR / "00_top_caes_workers.png",
			top_n=10,
			save=True,
		)

	if "ciuolabel" in datasets["ciuo_nodes"].columns:
		pl.plot_top_n_bar(
			df=datasets["ciuo_nodes"],
			label_col="ciuolabel",
			val_col="n_obs",
			color_col=None,
			title=None,
			xlabel="Cantidad de trabajadores",
			figsize=cfg.TOP_N_BAR_FIGSIZE,
			font_size=cfg.PLOT_FONT_SIZE,
			output_path=cfg.IMAGE_DIR / "00_top_ciuo_workers.png",
			top_n=10,
			save=True,
		)

	print("Exploratory plots generated successfully.")
	return datasets

if __name__ == "__main__":
	main()
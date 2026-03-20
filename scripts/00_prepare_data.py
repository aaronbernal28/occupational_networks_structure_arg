from scripts import *
import pandas as pd

def main():
	datasets = dl.load_dataset(
		RAW_ENES_PATH, 
		RAW_CAES_NODELIST_PATH, 
		RAW_CIUO_NODELIST_PATH, 
		CAES_ID,
		CIUO_ID,
		MAX_CAES_ID,
		CAES_LETRA,
		CEAS_AG,
		CAES_LETRA_OLD,
		CAES_AG_OLD,
		CIUO_LETRA,
		CIUO_3CAT,
		CIUO_LETRA_OLD,
		CAES_LABEL_COLOR,
		CAES_LETRA_COLOR,
		CAES_AG_COLOR,
		CIUO_LABEL_COLOR,
		CIUO_LETRA_COLOR,
		CIUO_3CAT_COLOR,
		)
	datasets_2021 = dl.load_dataset(
		RAW_ENES_2021_PATH, 
		RAW_CAES_NODELIST_PATH, 
		RAW_CIUO_NODELIST_PATH, 
		CAES_ID,
		CIUO_ID,
		MAX_CAES_ID,
		CAES_LETRA,
		CEAS_AG,
		CAES_LETRA_OLD,
		CAES_AG_OLD,
		CIUO_LETRA,
		CIUO_3CAT,
		CIUO_LETRA_OLD,
		CAES_LABEL_COLOR,
		CAES_LETRA_COLOR,
		CAES_AG_COLOR,
		CIUO_LABEL_COLOR,
		CIUO_LETRA_COLOR,
		CIUO_3CAT_COLOR,
	)
	datasets["enes"] = pd.concat([datasets["enes"], datasets_2021["enes"]], ignore_index=True, axis=0)

	caes_features = nc.compute_group_characteristics(datasets["enes"], CAES_ID)
	ciuo_features = nc.compute_group_characteristics(datasets["enes"], CIUO_ID)

	datasets["caes_nodes"] = nc.attach_group_characteristics(datasets["caes_nodes"], caes_features)
	datasets["ciuo_nodes"] = nc.attach_group_characteristics(datasets["ciuo_nodes"], ciuo_features)

	output1 = dl.export_processed(datasets["enes"], DATA_PROCESSED_PATH, "base_enespersonas")
	output2 = dl.export_processed(datasets["caes_nodes"], DATA_PROCESSED_PATH, "nodelist_caes")
	output3 = dl.export_processed(datasets["ciuo_nodes"], DATA_PROCESSED_PATH, "nodelist_ciuo")
	print(f"Saved merged dataset to {output1}, {output2}, and {output3}")

	color_map_caes = datasets["caes_nodes"][CAES_LABEL_COLOR].to_dict()
	color_map_ciuo = datasets["ciuo_nodes"][CIUO_LABEL_COLOR].to_dict()

	# Generate exploratory data analysis plots
	print(f"Generating exploratory plots in {IMAGE_DIR}...")
	if "caeslabel" in datasets["caes_nodes"].columns:
		pl.plot_top_n_bar(
			df=datasets["caes_nodes"],
			label_col="caeslabel",
			val_col="n_obs",
			color_col=CAES_LABEL_COLOR,
			title=None,
			xlabel="Total Workers",
			figsize=TOP_N_BAR_FIGSIZE,
			font_size=PLOT_FONT_SIZE,
			output_path=IMAGE_DIR / "00_top_caes_workers.png",
			top_n=15,
			save=True,
		)

	if "ciuolabel" in datasets["ciuo_nodes"].columns:
		pl.plot_top_n_bar(
			df=datasets["ciuo_nodes"],
			label_col="ciuolabel",
			val_col="n_obs",
			color_col=CIUO_LABEL_COLOR,
			title=None,
			xlabel="Total Workers",
			figsize=TOP_N_BAR_FIGSIZE,
			font_size=PLOT_FONT_SIZE,
			output_path=IMAGE_DIR / "00_top_ciuo_workers.png",
			top_n=15,
			save=True,
		)

	print("Exploratory plots generated successfully.")

if __name__ == "__main__":
	main()
from scripts import *
import pandas as pd

def main():
	datasets = dl.load_dataset(
		RAW_ENES_PATH, 
		RAW_CAES_NODELIST_PATH, 
		RAW_CIUO_NODELIST_PATH, 
		CAES_ID,
		CIUO_ID
		)
	datasets_2021 = dl.load_dataset(
		RAW_ENES_2021_PATH, 
		RAW_CAES_NODELIST_PATH, 
		RAW_CIUO_NODELIST_PATH, 
		CAES_ID,
		CIUO_ID
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

	# Generate exploratory data analysis plots
	print(f"Generating exploratory plots in {IMAGE_DIR}...")
	pl.plot_exploratory_analysis(datasets["caes_nodes"], datasets["ciuo_nodes"], IMAGE_DIR)
	print("Exploratory plots generated successfully.")

if __name__ == "__main__":
	main()
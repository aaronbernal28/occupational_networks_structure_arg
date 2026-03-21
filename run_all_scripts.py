import importlib
import time

import config as cfg
import pandas as pd

run_00_prepare_data = importlib.import_module('scripts.00_prepare_data').main
run_02_bipartite = importlib.import_module('scripts.02_bipartite').main
run_06_ciuo_projection_custom = importlib.import_module('scripts.06_ciuo_projection_custom').main
run_14_ciuo_edge_correlation = importlib.import_module('scripts.14_ciuo_edge_correlation').main

# Global variables for dataframes
enes_df = None
nodelist_caes_df = None
nodelist_ciuo_df = None

def run_step(name, func, *args, **kwargs):
	print(f"\n{name}...")
	start_time = time.time()
	func(*args, **kwargs)
	elapsed_time = time.time() - start_time
	print(f"Finished in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
	total_start_time = time.time()

	run_step("Running data preparation", run_00_prepare_data)
	
	# Load data once after preparation
	print("\nLoading datasets...")
	start_time = time.time()
	enes_df = pd.read_csv(cfg.DATA_PROCESSED_PATH / "base_enespersonas.csv")
	nodelist_caes_df = pd.read_csv(cfg.DATA_PROCESSED_PATH / "nodelist_caes.csv", index_col=cfg.DATA_ENES_PISAC["col_caes_id"])
	nodelist_ciuo_df = pd.read_csv(cfg.DATA_PROCESSED_PATH / "nodelist_ciuo.csv", index_col=cfg.DATA_ENES_PISAC["col_ciuo_id"])
	print(f"Finished loading datasets in {time.time() - start_time:.2f} seconds.")
	
	run_step("Running bipartite graph construction", run_02_bipartite, enes_df=enes_df, nodelist_caes_df=nodelist_caes_df, nodelist_ciuo_df=nodelist_ciuo_df)
	
	run_step("Analyzing CIUO projection with custom weights", run_06_ciuo_projection_custom, enes_df=enes_df, nodelist_ciuo_df=nodelist_ciuo_df)
	nodelist_ciuo_df = pd.read_csv(cfg.DATA_PROCESSED_PATH / "nodelist_ciuo.csv", index_col=cfg.DATA_ENES_PISAC["col_ciuo_id"])
	run_step("Analyzing CIUO gender correlation", run_14_ciuo_edge_correlation, enes_df=enes_df, nodelist_ciuo_df=nodelist_ciuo_df)

	total_elapsed_time = time.time() - total_start_time
	print(f"\nAll scripts executed successfully in {total_elapsed_time:.2f} seconds.")
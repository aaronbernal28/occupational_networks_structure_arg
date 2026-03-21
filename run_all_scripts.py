import importlib
import time

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
	result = func(*args, **kwargs)
	elapsed_time = time.time() - start_time
	print(f"Finished in {elapsed_time:.2f} seconds.")
	return result

if __name__ == "__main__":
	total_start_time = time.time()

	datasets = run_step("Running data preparation", run_00_prepare_data)
	if datasets is None:
		raise RuntimeError("scripts/00_prepare_data.main() must return datasets for in-memory pipeline execution.")

	enes_df = datasets["enes"]
	nodelist_caes_df = datasets["caes_nodes"]
	nodelist_ciuo_df = datasets["ciuo_nodes"]
	
	run_step("Running bipartite graph construction", run_02_bipartite, enes_df=enes_df, nodelist_caes_df=nodelist_caes_df, nodelist_ciuo_df=nodelist_ciuo_df)
	
	updated_ciuo_df = run_step("Analyzing CIUO projection with custom weights", run_06_ciuo_projection_custom, enes_df=enes_df, nodelist_ciuo_df=nodelist_ciuo_df)
	if updated_ciuo_df is not None:
		nodelist_ciuo_df = updated_ciuo_df

	run_step("Analyzing CIUO gender correlation", run_14_ciuo_edge_correlation, enes_df=enes_df, nodelist_ciuo_df=nodelist_ciuo_df)

	total_elapsed_time = time.time() - total_start_time
	print(f"\nAll scripts executed successfully in {total_elapsed_time:.2f} seconds.")
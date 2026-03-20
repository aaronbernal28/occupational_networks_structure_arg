from pathlib import Path

# Define paths and identifiers
DATA_RAW_PATH = Path(__file__).parent / Path("data/raw")
DATA_PROCESSED_PATH = Path(__file__).parent / Path("data/processed")
DATA_PROJECTION_GEPHI_PATH = Path(__file__).parent / Path("data/gephi")

RAW_ENES_PATH = DATA_RAW_PATH / "base_enespersonas.csv"
RAW_CAES_NODELIST_PATH = DATA_RAW_PATH / "nodelist_caes.csv"
RAW_CIUO_NODELIST_PATH = DATA_RAW_PATH / "nodelist_ciuo.csv"

RAW_ENES_2021_PATH = DATA_RAW_PATH / "base_enespersonas_2021.csv"
RAW_METADATA_ENES_PATH = DATA_RAW_PATH / "metadata_base_enespersonas.csv"

ENES_PATH = DATA_PROCESSED_PATH / "base_enespersonas.csv"
CAES_NODELIST_PATH = DATA_PROCESSED_PATH / "nodelist_caes.csv"
CIUO_NODELIST_PATH = DATA_PROCESSED_PATH / "nodelist_ciuo.csv"

CAES_ID = "v182caes"
CIUO_ID = "v183ciuo"

CAES_LETRA = "caesletra"
CIUO_LETRA = "ciuo1diglabel"

CAES_LETRA_OLD = "caesletra_old"
CIUO_LETRA_OLD = "ciuo1diglabel_old"

CEAS_AG = "caesag"
CIUO_3CAT = "ciuo3cat"

CAES_AG_OLD = "caesag_old"

CAES_LABEL = "caeslabel"
CIUO_LABEL = "ciuolabel"

CAES_LABEL_COLOR = "caeslabel_color"
CIUO_LABEL_COLOR = "ciuolabel_color"
CAES_LETRA_COLOR = "caesletra_color"
CIUO_LETRA_COLOR = "ciuo1diglabel_color"
CAES_AG_COLOR = "caesag_color"
CIUO_3CAT_COLOR = "ciuo3cat_color"

TOTAL_INCOME = "ITI" # Monto de ingreso total individual 

MAX_CAES_ID = 10000  # Threshold to disambiguate CAES and CIUO IDs

IMAGE_DIR = Path(__file__).parent / Path("images")

TOP_N_BAR_FIGSIZE = (8, 6)
EDGE_CORRELATION_FIGSIZE = (6, 6)
BIPARTITE_FIGSIZE = (12, 6)
PROJECTION_FIGSIZE = (8, 8)
STACKED_FIGSIZE = (12, 3)
PLOT_FONT_SIZE = 11

LOGSCALE = False

COMMUNITY_COLORS_PALETTE = [f"C{i}" for i in range(10)] + ["black", "gold", "darkred", "navy", "darkcyan"]
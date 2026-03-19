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

TOTAL_INCOME = "ITI" # Monto de ingreso total individual 

MAX_CAES_ID = 10000  # Threshold to disambiguate CAES and CIUO IDs

IMAGE_DIR = Path(__file__).parent / Path("images")

LOGSCALE = False

COMMUNITY_COLORS_PALETTE = [f"C{i}" for i in range(10)] + ["black", "gold", "darkred", "navy", "darkcyan"]
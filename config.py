from pathlib import Path

# Define paths and identifiers
DATA_RAW_PATH = Path(__file__).parent / Path("data/raw")
DATA_PROCESSED_PATH = Path(__file__).parent / Path("data/processed")

DATA_ENES_PISAC = {
    "source": DATA_RAW_PATH / "base_enespersonas.csv",
    "url": None, #"https://datasets.datos.mincyt.gob.ar/dataset/b0c0ae96-4028-4d0e-919a-c6830110f6b6/resource/421a7e29-7d15-4b69-a2e8-d7242b58b770/download/base_enespersonas.csv",
    "col_id": ["nocues", "nhog"],
    "col_caes_id": "v182caes",
    "col_ciuo_id": "v183ciuo",
    "col_sex_id": "v109",
    "col_public_worker": "v188",
    "col_total_income": "ITI",
}

DATA_EXTRA = [
    {
        "source": DATA_RAW_PATH / "base_enespersonas_2021.csv",
        "url": None,
        "year": 2021,
        "col_id": "CUEST",
        "col_caes_id": "CAES_num",
        "col_ciuo_id": "CIUO_encuestado",
        "col_sex_id": "SEXO",
        "col_public_worker": "M3_7",
        "col_total_income": None,
    }
]

DATA_NODELIST_CAES = {
    "source": DATA_RAW_PATH / "nodelist_caes.csv",
    "col_id": "v182caes",
    "col_label": "caeslabel",
    "col_letra": "caesletra",
    "col_ag": "caesag",
    "col_label_color": "caeslabel_color",
    "col_letra_color": "caesletra_color",
    "col_ag_color": "caesag_color",
}

DATA_NODELIST_CIUO = {
    "source": DATA_RAW_PATH / "nodelist_ciuo.csv",
    "col_id": "v183ciuo",
    "col_label": "ciuolabel",
    "col_letra": "ciuo1diglabel",
    "col_3cat": "ciuo3cat",
    "col_label_color": "ciuolabel_color",
    "col_letra_color": "ciuo1diglabel_color",
    "col_3cat_color": "ciuo3cat_color",
}

MAX_CAES_ID = 10000  # Threshold to disambiguate CAES and CIUO IDs

IMAGE_DIR = Path(__file__).parent / Path("images")

TOP_N_BAR_FIGSIZE = (8, 6)
EDGE_CORRELATION_FIGSIZE = (6, 6)
BIPARTITE_FIGSIZE = (14, 6)
PROJECTION_FIGSIZE = (8, 8)
STACKED_FIGSIZE = (12, 4)
PLOT_FONT_SIZE = 14

LOGSCALE = False

COMMUNITY_COLORS_PALETTE = [f"C{i}" for i in range(10)] + ["black", "gold", "darkred", "navy", "darkcyan"]
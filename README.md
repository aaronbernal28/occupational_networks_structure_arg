# Occupational Networks: Structure and Gender Gap.

### Authors
* **Joaquín Carrascosa** ¹
* **Aaron Bernal Huanca** ²
* **Viktoriya Semeshenko** ³
* **Carlos Sarraute** ⁴

### Affiliations
1. **CONICET - Universidad de Buenos Aires** Instituto de Investigaciones Gino Germani (IIGG-UBA)  
   Email: [jcarrascosa@sociales.uba.ar](mailto:jcarrascosa@sociales.uba.ar)

2. **Licenciatura en Ciencia de Datos, FCEyN** Universidad de Buenos Aires Email: [ahuanca@dc.uba.ar](mailto:ahuanca@dc.uba.ar)

3. **CONICET - Universidad de Buenos Aires** Instituto Interdisciplinario de Economía Política de Buenos Aires (IIEP)

4. **Disruptive Research Institute** Email: [csarraute@disruptiveresearch.org](mailto:csarraute@disruptiveresearch.org)


**Subject:** Social Network Science

## General Description

Aquí iria el abstract del paper

## Project Structure

```text
occupational_and_branch_of_activity_networks/
├── config.py            # Configuration file with paths and parameters
├── requirements.txt     # Python dependencies
├── run_all_scripts.py   # Execute complete pipeline
├── data/
│   ├── raw/             # Raw survey data and node lists
│   └── processed/       # Processed and merged datasets
├── images/              # Generated visualizations
├── notebooks/           # Exploratory analysis notebooks
├── scripts/             # Analysis pipeline
│   ├── 00_prepare_data.py         # Merge and prepare datasets
│   ├── 01_heatmap.py              # Generate adjacency heatmaps
│   ├── 02_bipartite.py            # Visualize bipartite network
│   ├── 03_caes_projection.py      # CAES weighted projection
│   ├── 04_ciuo_projection.py      # CIUO weighted projection
│   ├── 05_caes_projection_custom.py   # CAES custom weight functions
│   ├── 06_ciuo_projection_custom.py   # CIUO custom weight functions
│   ├── 07_caes_tda.py             # TDA on CAES network
│   ├── 08_ciuo_tda.py             # TDA on CIUO network
│   ├── 09_caes_backbone_disparity.py  # Backbone extraction for CAES
│   ├── 10_ciuo_backbone_disparity.py  # Backbone extraction for CIUO
│   └── so on for other scripts...
├── src/                 # Source code modules
│   ├── communities.py   # Community detection algorithms
│   ├── data_loader.py   # Data loading and preprocessing
│   ├── graph_construction.py  # Bipartite and projection methods
│   ├── metrics.py       # Network metrics and statistics
│   ├── plotting.py      # Visualization functions
│   ├── topology.py      # Topological data analysis
│   └── utils.py         # Utility functions
```

## Methodology

1. **Data Preparation**: Loads and merges ENES survey data with classification node lists for CAES (economic activities) and CIUO (occupations).
2. **Bipartite Network Construction**: Creates a weighted bipartite graph where edges connect occupations to economic activity branches based on employment relationships.
3. **Network Projections**: Generates unipartite projections using multiple weighting functions:
	- Standard weighted projection
	- Hidalgo proximity (Hausmann et al., 2007)
	- Dot product weight (Newman, 2001)
	- Cosine similarity weight
	- Weighted Higaldo proximity
4. **Community Detection**: Applies the Louvain algorithm to identify communities in projected networks.
5. **Backbone Extraction**: Uses the disparity filter method to extract the most significant edges.
6. **Topological Analysis**: Computes persistent homology on distance matrices derived from network weights.
7. **Visualization**: Generates heatmaps, network layouts, degree distributions, persistence diagrams, and community structure plots.

## Key Dependencies

- **Network Analysis**: NetworkX, scikit-learn
- **TDA Libraries**: Ripser, Persim, GUDHI for persistent homology
- **Scientific Computing**: NumPy, Pandas, SciPy
- **Visualization**: Matplotlib, Seaborn

## Installation

Python 3.12.9 is recommended (my current environment).

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Usage Workflow

1. **Prepare and merge datasets**:
	```bash
	python -m scripts.00_prepare_data
	```

2. **Run the complete analysis pipeline**:
	```bash
	python run_all_scripts.py
	```

3. **Run individual analysis scripts** (optional):
	```bash
	# Generate heatmap of bipartite adjacency matrix
	python -m scripts.01_heatmap
	
	# Generate visualization of bipartite network
	python -m scripts.02_bipartite
	```

4. **Explore results**: Generated visualizations are saved in the `images/` directory.

5. **Get logcat output**: To see the detailed output of each script, run:
	```bash
	python -m run_all_scripts > logcat.txt
	```

## Data Description

- **ENES PISAC 2019**: https://datos.gob.ar/sq/dataset/mincyt-pisac---programa-investigacion-sobre-sociedad-argentina-contemporanea
- **ESAyPP 2021**: Encuesta sobre Estructura Social y Políticas Publicas 2021
- **CAES Node List**: Classification of Economic Activities (branch of activity codes and labels)
- **CIUO Node List**: Classification of Occupations 1.0 Argentina (occupation codes and labels)

## References

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

This work analyzes the structure of the Argentine labor market through a complex networks approach that links occupations (CIUO-08) and branches of economic activity (CAES 1.0). Using microdata from the ENES (2019) and ESAyPP (2021) surveys, we build a high-resolution bipartite network and its corresponding unipartite projections to uncover interdependence patterns not visible with traditional methods. The results reveal a modular organization where industrial sectors and social services occupy differentiated regions of the network, while commerce acts as a bridge node that connects both spaces.

In the occupational network, we observe a marked segmentation between manual and non-manual occupations, along with the presence of transversal occupations that connect multiple economic sectors. A central finding is that gender segregation has a topological correlate: occupations with similar gender compositions tend to cluster in the network (homophily), indicating that inequality is written into the relational structure of the labor market. These results show the potential of network analysis to understand the structural organization of the labor market and its patterns of segmentation.

## Project Structure

```text
occupational_networks_structure_arg/
├── config.py                          # Configuration file with paths and parameters
├── requirements.txt                   # Python dependencies
├── run_all_scripts.py                 # Execute complete pipeline
├── data/
│   ├── raw/                           # Raw survey data and node lists
│   │   ├── base_enespersonas_2021.csv
│   │   ├── nodelist_caes.csv
│   │   └── nodelist_ciuo.csv
│   ├── processed/                     # Processed and merged datasets
│   │   ├── base_enespersonas.csv
│   │   ├── nodelist_caes.csv
│   │   └── nodelist_ciuo.csv
│   └── gephi/                         # Optional exports for Gephi
├── images/                            # Generated visualizations
├── scripts/                           # Analysis pipeline
│   ├── 00_prepare_data.py             # Merge data, compute characteristics, EDA plots
│   ├── 02_bipartite.py                # Visualize bipartite network by group
│   ├── 06_ciuo_projection_custom.py   # CIUO projection + communities + gradients
│   └── 14_ciuo_edge_correlation.py    # Edge correlation plot (gender/communities)
├── src/                               # Source code modules
│   ├── communities.py                 # Community detection algorithms
│   ├── data_loader.py                 # Data loading and preprocessing
│   ├── graph_construction.py          # Bipartite and projection methods
│   ├── metrics.py                     # Network metrics and statistics
│   ├── node_characteristics.py        # Group-level feature computation
│   ├── plotting.py                    # Visualization functions
│   └── utils.py                       # Utility functions
```

## Methodology

1. **Data Preparation**: Loads ENES/ESAyPP microdata and CAES/CIUO node lists, computes group-level characteristics, and exports processed datasets.
2. **Exploratory Plots**: Generates top-N bar charts for CAES and CIUO employment counts.
3. **Bipartite Network Construction**: Builds the CAES-CIUO bipartite graph, logs summary metrics, and plots a colored bipartite layout by group.
4. **CIUO Projection (Custom Weights)**: Creates a CIUO projection using weighted Hidalgo proximity, plots the network by CIUO categories, and stores node positions.
5. **Community Detection**: Runs Louvain with resolution search on the CIUO projection and visualizes community structure and distributions by CIUO letters.
6. **Gradient Visualizations**: Produces CIUO projection gradients for % women, mean age, and (if present) % public sector.
7. **Edge Correlation**: Relates CIUO edge weights to gender composition and community coloring.

## Key Dependencies

- **Network Analysis**: NetworkX
- **Scientific Computing**: NumPy, Pandas, SciPy
- **Visualization**: Matplotlib, Seaborn
- **File I/O**: openpyxl

## Installation

Python 3.12.3 is recommended.

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
	python -m run_all_scripts.py
	```

3. **Run individual analysis scripts** (optional):
	```bash
	# Build bipartite graph and group-colored layout
	python -m scripts.02_bipartite
	
	# CIUO projection with custom weights + community/gradient plots
	python -m scripts.06_ciuo_projection_custom
	
	# Edge correlation plot on CIUO projection
	python -m scripts.14_ciuo_edge_correlation
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

### Getting Started

Before running the code, the required packages are listed in `requirements.txt`. It is recommended to create a virtual environment for this project.

The analysis pipeline can be found in the following notebooks:
- **`demo_analyze_pipeline.ipynb`**: Contains the analysis workflow using sample data from May (note: some data were removed to reduce database size)
- **`original_analyze_pipeline.ipynb`**: Shows the complete output from the original dataset

The `src/` directory contains core functions:
- **`poi_joint.py`**: Generates stop-pair connections based on different gap settings
- **`statistical_learning.py`**: Implementations for causal analysis

### Data Preprocessing

The complete data cleaning procedure is not included in this repository as it is not in the project scope and was not performed in a fully reproducible manner. And the data access requires a personal API key which I cannot share here. However, the `preprocessing/` directory contains code for:
- Downloading GTFS static information
- Fetching real-time TripUpdates feeds

For questions about data handling or reproducibility, please feel free to contact me.

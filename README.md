# Heart Disease Prediction

This repository provides a small reproducible project for building, tuning, and evaluating machine learning models to predict heart disease from a standard clinical dataset.

The code includes data loading, preprocessing, model training (Logistic Regression, Random Forest, K-Nearest Neighbors), evaluation (metrics, confusion matrices, ROC curves), and an interactive Jupyter notebook that walks through the full experiment.

---

## Quick links

- Dataset (raw CSV used by the project): (https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- Notebook (interactive): heart_disease_prediction.ipynb

---

## Project overview

The goal of this project is to demonstrate a simple machine learning pipeline for binary classification (presence vs absence of heart disease) using commonly available clinical features. The repository contains:

- Scripts for preprocessing, training, evaluation, and an end-to-end pipeline.
- A Jupyter notebook with exploratory data analysis (EDA) and reproducible steps.
- Visual outputs (saved to `outputs/`) including ROC curves, confusion matrices, and feature-importance plots.


## Files in this repository

- README.md — This file
- heart.csv — Dataset (CSV)
- heart_disease_prediction.ipynb — Jupyter notebook with step-by-step EDA and modeling
- preprocess.py — Data loading and preprocessing utilities
- train.py — Model training and hyperparameter tuning (GridSearchCV)
- evaluate.py — Model evaluation functions and plotting helpers
- run_pipeline.py — Intended end-to-end pipeline (download, preprocessing, training, evaluation)


## Requirements

Recommended Python version: 3.8+ (tested with 3.10/3.12)

Install the minimal dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn

(If you prefer, create a virtual environment first.)


## How to run

Two recommended ways to run the project:

1) Jupyter notebook (recommended)

- Open `heart_disease_prediction.ipynb` in Jupyter or Colab:
  - Colab link: https://colab.research.google.com/github/GWTM505/Heart-Disease-Prediction/blob/main/heart_disease_prediction.ipynb
- The notebook contains EDA, deduplication analysis, preprocessing, model training/tuning, and visualizations. It is the easiest way to reproduce the results interactively.

2) Run the pipeline script (non-interactive)

- The repository includes `run_pipeline.py` which is intended to download the dataset, run EDA, preprocess, train, and evaluate models and save outputs under `outputs/`.

Usage:

python run_pipeline.py

Notes / potential fix when running the script:
- `run_pipeline.py` currently imports modules using `from src.preprocess import ...` and similar imports (`src.train`, `src.evaluate`). In this repository the helper scripts live at the repository root (e.g. `preprocess.py`, `train.py`, `evaluate.py`), so the `src.` prefix will raise ImportError unless you make one of the following adjustments:
  - Quick local fix: edit `run_pipeline.py` and replace the `src.` imports with direct imports: `from preprocess import load_data, get_data_summary, preprocess_pipeline` and `from train import train_and_tune_models` and `from evaluate import evaluate_models, plot_feature_importance`.
  - Or run Python with the repository added to PYTHONPATH and move the helper modules into a `src/` package directory (create `src/__init__.py` and move files to `src/`).


## Outputs

Running the notebook or pipeline will create an `outputs/` directory containing:

- target_distribution.png
- correlation_matrix.png
- roc_curves_comparison.png
- confusion_matrix_<model>.png (one per model)
- feature_importance_random_forest.png
- feature_importance_logistic_regression.png
- model_comparison_deduplicated.csv
- model_comparison_duplicated.csv

These files are created by the evaluation and EDA functions.


## Notes on deduplication and data leakage

- The dataset used here contains duplicate rows in the original CSV. The notebook and `preprocess.py` demonstrate deduplication and warn about potential data leakage when duplicates are kept before splitting.
- The pipeline runs both deduplicated and duplicated experiments to illustrate the effect on performance.


## Contributing

Contributions are welcome. Suggestions:

- Add a `requirements.txt` or `environment.yml` for reproducible environments.
- Convert helper scripts into a proper Python package (move files into `src/`, add a setup.cfg/pyproject.toml) so `run_pipeline.py` imports work without modification.
- Add unit tests for preprocessing and model training steps.


## License

This repository does not include a license file. If you intend to share or reuse the code publicly, consider adding an open-source license (for example MIT or Apache-2.0).


## Contact

If you have questions or need help reproducing the results, open an issue in this repository.

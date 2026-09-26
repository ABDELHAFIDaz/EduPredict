# EduPredict

Predicting student exam scores from study habits, past performance and socio-familial context, to help identify students at risk of failing early and support the pedagogical team.

## Project structure

```
EduPredict/
├── app.py                        # Streamlit app
├── data/
│   ├── dataset.csv               # not tracked in git, see "Data" below
│   └── dictionnaire_des_donnees.md
├── models/
│   ├── final_pipeline.joblib     # trained pipeline (scaler + model)
│   └── expected_columns.json     # column order expected by the pipeline
├── notebooks/
│   └── edupredict.ipynb          # full analysis, training and evaluation
├── src/edupredict/
│   ├── __init__.py
│   └── preprocessing.py          # encoding + pipeline logic, shared by the notebook and the app
├── pyproject.toml
├── uv.lock
└── README.md
```

## Data

The dataset isn't committed to this repository. Download `dataset.csv` from the project resources and place it in `data/`:

```
EduPredict/data/dataset.csv
```

The column descriptions are in `data/dictionnaire_des_donnees.md`.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

This installs Python 3.12 and all dependencies from `uv.lock`.

## Running the notebook

```bash
uv run jupyter lab
```

Open `notebooks/edupredict.ipynb` and run the cells from top to bottom. It covers data loading, exploratory analysis, cleaning, encoding, model training, hyperparameter tuning, evaluation and model export.

## Running the app

```bash
uv run streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`. Enter a student's profile using the sliders and dropdowns, then click **Predict** to get an estimated exam score. Students predicted below 60 are flagged as at risk.

## Models

Four regression models were compared: Linear Regression, Random Forest, SVR and XGBoost. Linear Regression was selected as the final model, it performs on par with tuned SVR while being simpler and fully interpretable. Full comparison and reasoning are in the notebook.

## Notes

XGBoost was excluded from the final comparison; see the notebook for details.
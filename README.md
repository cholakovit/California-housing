TASK: A property analytics team wants to estimate typical home values across different neighborhoods so they can prioritize expansion markets, support pricing discussions, and focus sales efforts on the most promising areas; they will use available neighborhood characteristics to produce consistent value estimates, compare two internal approaches on the same sample, and select the one that gives the most reliable business signal for day-to-day decision-making.

----------------------------------------------------

# Linear regression (California housing)

Small demo: load the **California housing** tabular dataset from the Hugging Face Hub, fit **ordinary least squares** linear regression on numeric features, and report test metrics and coefficients.

## Task

- **Goal:** predict **median house value** (`MedHouseVal`) for a census block from eight numeric attributes (median income, housing age, rooms, bedrooms, population, occupancy, latitude, longitude).
- **Model:** `sklearn.linear_model.LinearRegression` inside a pipeline with `StandardScaler` so features are normalized before fitting; coefficients are interpreted **per one standard deviation** of each input.
- **Split:** random 80% / 20% train–test (`random_state=42`). (Geographic autocorrelation is ignored here; this is a teaching baseline.)
- **Metrics on the test set:** R², RMSE, MAE. Target units are **hundreds of thousands of USD** (as in the original dataset).

## Data

- Source: [`gvlassis/california_housing`](https://huggingface.co/datasets/gvlassis/california_housing) via `datasets.load_dataset`.
- The script uses the Hub **`train`** split only, then applies its own `train_test_split`.

## Requirements

- Python **3.14+** (see `pyproject.toml`). The repo pins **`3.14`** in `.python-version`; [uv](https://docs.astral.sh/uv/) can install it with `uv python install 3.14` if you do not have it yet.
- [uv](https://docs.astral.sh/uv/) recommended.

## Setup

```bash
uv sync
```

Optional: copy `.env.example` to `.env` and set `HF_TOKEN` for higher Hugging Face Hub rate limits. Public datasets work without a token.

## Run

```bash
uv run python main.py
uv run python main_boost.py
```

`main_boost.py` uses **`GradientBoostingRegressor`** (gradient boosting in scikit-learn) on the **same** train/test split (`random_state=42`) so you can compare R² / RMSE / MAE with `main.py`.

## Project layout

| Path              | Role                                                    |
|-------------------|---------------------------------------------------------|
| `main.py`         | OLS `LinearRegression` + `StandardScaler`, coefficients |
| `main_boost.py`   | Histogram gradient boosting, feature importances        |
| `pyproject.toml` | Dependencies and Python version       |
| `.env.example` | Example env vars (no secrets committed)   |

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from scipy.stats import loguniform, randint
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split

TARGET = "MedHouseVal"


def enrich_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    ar = X["AveRooms"].clip(lower=1e-6)
    ao = X["AveOccup"].clip(lower=1e-6)
    X["BedRmRatio"] = X["AveBedrms"] / ar
    X["RoomsPerOccup"] = X["AveRooms"] / ao
    X["LogPop"] = np.log1p(X["Population"])
    lat0 = X["Latitude"].mean()
    lon0 = X["Longitude"].mean()
    X["DistCenter"] = np.sqrt(
        (X["Latitude"] - lat0) ** 2 + (X["Longitude"] - lon0) ** 2
    )
    return X


def main() -> None:
    load_dotenv()
    ds = load_dataset("gvlassis/california_housing", split="train")
    df = ds.to_pandas()
    base_cols = [c for c in df.columns if c != TARGET]
    X_raw = df[base_cols]
    y = df[TARGET].values.astype(np.float64)
    X = enrich_features(X_raw)
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    base = HistGradientBoostingRegressor(random_state=42)
    model = TransformedTargetRegressor(
        regressor=base,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    param_distributions = {
        "regressor__max_depth": randint(3, 12),
        "regressor__learning_rate": loguniform(0.02, 0.2),
        "regressor__max_iter": randint(200, 600),
        "regressor__min_samples_leaf": randint(15, 80),
        "regressor__l2_regularization": loguniform(1e-5, 1.0),
        "regressor__max_leaf_nodes": randint(31, 127),
    }
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=40,
        cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    y_pred = best.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("HistGradientBoosting + CV tuning + log1p target + extra features")
    print(f"CV best RMSE (mean): {-search.best_score_:.4f} (100k USD)")
    print(f"test R2:  {r2:.4f}")
    print(f"test RMSE: {rmse:.4f} (100k USD)")
    print(f"test MAE:  {mae:.4f} (100k USD)")
    print("best params:")
    for k, v in sorted(search.best_params_.items()):
        print(f"  {k}: {v}")

    perm = permutation_importance(
        best,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    print("permutation_importance (mean decrease, test set):")
    order = np.argsort(perm.importances_mean)[::-1]
    for i in order:
        m = perm.importances_mean[i]
        s = perm.importances_std[i]
        print(f"  {feature_cols[i]}: {m:.4f} (+/- {s:.4f})")


if __name__ == "__main__":
    main()

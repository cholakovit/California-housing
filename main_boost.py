from dotenv import load_dotenv
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

TARGET = "MedHouseVal"


def main() -> None:
    load_dotenv()
    ds = load_dataset("gvlassis/california_housing", split="train")
    df = ds.to_pandas()
    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols]
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("GradientBoostingRegressor (sklearn)")
    print(f"test R2:  {r2:.4f}")
    print(f"test RMSE: {rmse:.4f} (100k USD)")
    print(f"test MAE:  {mae:.4f} (100k USD)")

    imp = model.feature_importances_
    print("feature_importances_:")
    for name, w in sorted(
        zip(feature_cols, imp), key=lambda t: t[1], reverse=True
    ):
        print(f"  {name}: {w:.4f}")


if __name__ == "__main__":
    main()

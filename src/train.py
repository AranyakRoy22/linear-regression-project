"""Train and evaluate a linear regression model on the diabetes dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def main() -> None:
    dataset = load_diabetes(as_frame=True)
    features = dataset.data
    target = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "mean_absolute_error": mean_absolute_error(y_test, predictions),
        "root_mean_squared_error": mean_squared_error(y_test, predictions) ** 0.5,
        "r2_score": r2_score(y_test, predictions),
    }

    OUTPUTS.mkdir(exist_ok=True)
    pd.Series(metrics, name="value").to_csv(OUTPUTS / "metrics.csv", header=True)

    results = pd.DataFrame({"actual": y_test, "predicted": predictions})
    results.to_csv(OUTPUTS / "predictions.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.75, color="#176b87")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--", color="#d1495b")
    plt.xlabel("Actual disease progression")
    plt.ylabel("Predicted disease progression")
    plt.title("Linear Regression: Actual vs. Predicted")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "actual_vs_predicted.png", dpi=150)
    plt.close()

    print("Linear regression evaluation")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"\nSaved outputs to: {OUTPUTS}")


if __name__ == "__main__":
    main()

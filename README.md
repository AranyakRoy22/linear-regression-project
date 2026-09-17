# Linear Regression on the Diabetes Dataset

A reproducible machine learning project that trains and evaluates a linear regression model using scikit-learn's built-in diabetes dataset. The project also includes a custom `r2_score` implementation in `src/metrics.py` for learning and transparency.

## Project Highlights

- Loads a public dataset without a separate download step
- Splits data into training and test sets
- Trains a `LinearRegression` model
- Uses a custom `r2_score` function from `src/metrics.py`
- Reports MAE, RMSE, and R²
- Saves predictions, metrics, and an actual-vs-predicted plot

## Project Structure

```text
linear_regression_project/
├── README.md
├── LICENSE
├── requirements.txt
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── metrics.csv
│   └── predictions.csv
├── src/
│   ├── __init__.py
│   ├── metrics.py
│   └── train.py
└── tests/
    └── test_train.py
```

## Setup

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

From the project root:

```bash
python src/train.py
```

The script creates these files in `outputs/`:

- `metrics.csv`
- `predictions.csv`
- `actual_vs_predicted.png`

## Custom Metric

The custom regression metric is implemented in `src/metrics.py`:

```python
import numpy as np


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot
```

This computes the coefficient of determination using the standard formula:

$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

## Example Results

Results can vary slightly with library versions. With the current split and random seed, the model should achieve an R² score around `0.45` on the held-out test set.

## Dataset

The dataset contains ten baseline medical measurements for 442 patients and a quantitative measure of disease progression one year later. It is bundled with scikit-learn and is used here for educational purposes.

## Next Steps

Possible extensions include comparing regularized models such as Ridge and Lasso, adding cross-validation, and tuning the train/test split.

## License

This project is available under the MIT License.

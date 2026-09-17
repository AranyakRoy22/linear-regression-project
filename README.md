# Linear Regression on the Diabetes Dataset

A reproducible machine learning project that trains and evaluates a linear regression model using scikit-learn's built-in diabetes dataset.

## Project Highlights

- Loads a public dataset without a separate download step
- Splits data into training and test sets
- Trains a `LinearRegression` model
- Reports MAE, RMSE, and R²
- Saves predictions, metrics, and an actual-vs-predicted plot

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

## Example Results

Results can vary slightly with library versions. With the current split and random seed, the model should achieve an R² score around `0.45` on the held-out test set.

## Dataset

The dataset contains ten baseline medical measurements for 442 patients and a quantitative measure of disease progression one year later. It is bundled with scikit-learn and is used here for educational purposes.

## Next Steps

Possible extensions include comparing regularized models such as Ridge and Lasso, adding cross-validation, and tuning the train/test split.

## License

This project is available under the MIT License.

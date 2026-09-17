from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def test_model_trains_and_predicts() -> None:
    dataset = load_diabetes(as_frame=True)
    x_train, x_test, y_train, _ = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42
    )

    model = LinearRegression().fit(x_train, y_train)
    predictions = model.predict(x_test)

    assert len(predictions) == len(x_test)
    assert predictions.shape == (len(x_test),)

import numpy as np
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression

def get_selectors_chart(hist: pd.DataFrame, nearest):
    # Return an invisible selectors chart for Altair hover.
    return alt.Chart(hist).mark_point().encode(
        x='Datetime:T',
        y='Close:Q',
        opacity=alt.value(0),
    ).add_params(
        nearest
    )

def get_points_chart(line, nearest):
    """Return points chart for Altair hover."""
    return line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )


def linRegVis(hist: pd.DataFrame) -> pd.DataFrame:
    time_in_hours = (hist['Datetime'] - hist['Datetime'].iloc[0]).dt.total_seconds() / 3600.0
    X = time_in_hours.values.reshape(-1, 1)
    y = np.log(hist['Close'] / hist['Close'].iloc[0])
    model = LinearRegression().fit(X, y)

    hist = hist.copy()
    hist['PredictedLogClose'] = model.predict(X)
    hist['PredictedClose'] = hist['Close'].iloc[0] * np.exp(hist['PredictedLogClose'])

    plot_data = pd.concat([
        hist[['Datetime', 'Close']].assign(Type='Actual'),
        hist[['Datetime', 'PredictedClose']].rename(columns={'PredictedClose': 'Close'}).assign(Type='Regression')
    ])
    return plot_data, model, hist['Close'].iloc[0]
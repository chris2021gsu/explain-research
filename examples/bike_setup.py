"""Shared setup for the bike example: model, Explainer, and two forecasts to compare.

    from bike_setup import ex, test, last_week, this_week
"""

import lightgbm as lgb
from sklearn.datasets import fetch_openml

from explainkit import Explainer

df = fetch_openml(data_id=42712, as_frame=True, data_home=str(__import__("pathlib").Path(__file__).parents[1] / "data")).frame
features = [c for c in df.columns if c != "count"]

# Train on 2011, use on 2012: a realistic "forecast the future" setup.
train, test = df[df["year"] == 0], df[df["year"] == 1].reset_index(drop=True)
model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, random_state=0, verbose=-1)
model.fit(train[features], train["count"])

ex = Explainer(
    model, test, target_name="bike rentals per hour", y="count",
    feature_groups={"weather conditions": ["weather", "temp", "feel_temp", "humidity", "windspeed"],
                    "day type": ["holiday", "workingday", "weekday"],
                    "time of day": ["hour"], "time of year": ["season", "month"]},
    feature_descriptions={"year": "0 = 2011, 1 = 2012", "weekday": "0 = Sunday ... 6 = Saturday",
                          "feel_temp": "apparent temperature, Celsius", "humidity": "0 to 1"},
    value_labels={
        "weekday": dict(enumerate(["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])),
        "month": dict(enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)),
        "hour": lambda h: f"{h}:00",
        "holiday": {"True": "holiday", "False": "not a holiday"},
        "workingday": {"True": "working day", "False": "day off"},
        "temp": lambda t: f"{t:.0f}°C", "feel_temp": lambda t: f"feels like {t:.0f}°C",
        "humidity": lambda h: f"{h:.0%} humidity", "windspeed": lambda w: f"wind {w:.0f} km/h",
    },
    llm="gemini/gemini-2.5-flash",
)

# Two forecasts to compare: the same Friday-evening hour, one week apart.
friday_5pm = test[(test["weekday"] == 5) & (test["hour"] == 17)]
last_week, this_week = friday_5pm.iloc[[20]], friday_5pm.iloc[[21]]

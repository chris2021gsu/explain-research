"""explainkit quickstart: explain a house-price model in plain English.

pip install "git+https://github.com/chris2021gsu/explain-research.git"
python quickstart.py
"""

import lightgbm as lgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from explainkit import Explainer

# --- 1. Your existing model (any model with .predict() works) ---
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
y = y * 100_000  # dataset is in $100k units; use dollars
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = lgb.LGBMRegressor(n_estimators=300, random_state=0, verbose=-1).fit(X_train, y_train)

# --- 2. Attach explainkit ---
ex = Explainer(
    model, X_test, target_name="dollars", y=y_test,
    feature_groups={"location": ["Latitude", "Longitude"], "house size": ["AveRooms", "AveBedrms"]},
    value_labels={"MedInc": lambda v: f"income ${v * 10:.0f}k",
                  "HouseAge": lambda v: f"{v:.0f} yrs old",
                  "AveOccup": lambda v: f"{v:.1f} people/home"},
)

house = X_test.iloc[[0]]
neighbor = X_test.iloc[[1]]

# --- 3. Ask questions: every answer prints in plain English ---
print(ex.explain_prediction(house))                         # Why is this house valued at X?
print(ex.prediction_context(house))                         # Is that high or low?
print(ex.explain_change(before=neighbor, after=house))      # Why is it worth more/less than that one?
print(ex.what_if(house, {"MedInc": 8.0}))                   # What if the area's income were $80k?
print(ex.top_drivers())                                     # What matters most overall?
print(ex.model_errors())                                    # How accurate is the model?

# --- 4. The exact numbers are always there too ---
r = ex.explain_change(neighbor, house)
print(r["before"], r["after"], r["change"])

# --- 5. Any question in your own words (needs an LLM key, e.g. GEMINI_API_KEY) ---
# print(ex.ask("Why is this house worth more than its neighbor?", row=house, compare_to=neighbor))

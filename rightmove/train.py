import pickle
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import r2_score


all_data = pd.read_parquet("../data/rightmove_processed.parquet")

features = ["MEDIAN_LISTING_PRICE_PER_SQM"]
target = "Next 5yr yoy rent growth"

all_data = all_data.dropna(subset=features + [target])

train_data = all_data.query("year < 2019")
test_data = all_data.query("year == 2019")


model = XGBRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0, verbosity=0
)


x_train = train_data[features]
y_train = train_data[target]


model.fit(x_train, y_train)

test_predictions = model.predict(test_data[features])
test_actuals = test_data[target]

r2 = r2_score(test_actuals, test_predictions)

with open("metrics.json", "w") as f:
    f.write(f'{{"r2": {r2}}}')


# Save the model
pickle.dump(model, open("../model/model.pkl", "wb"))

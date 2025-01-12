import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from tqdm import tqdm

dtype_dict = {
    'Column9': 'str',
    'Column10': 'str',
    'Column11': 'str',
    'Column14': 'str',
    'Column15': 'str',
    'Column16': 'str',
    'Column17': 'str'
}


data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/electricity.csv", dtype=dtype_dict, low_memory=False)
columns_to_convert = [
    "ForecastWindProduction", "SystemLoadEA", "SMPEA",
    "ORKTemperature", "ORKWindspeed", "CO2Intensity",
    "ActualWindProduction", "SystemLoadEP2", "SMPEP2"
]
for col in columns_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna(subset=["SMPEP2"])
x = data[["Day", "Month", "ForecastWindProduction", "SystemLoadEA",
          "SMPEA", "ORKTemperature", "ORKWindspeed", "CO2Intensity",
          "ActualWindProduction", "SystemLoadEP2"]]
y = data["SMPEP2"]


imputer = SimpleImputer(strategy='mean')
x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)
y = y.fillna(y.mean())


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)


models = {
    "Random Forest": RandomForestRegressor(n_estimators=10, n_jobs=-1, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Bagging": BaggingRegressor(n_estimators=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=10, random_state=42),
    "SVM": SVR(kernel='rbf')
}


results = {}
predictions = {}

for model_name, model in tqdm(models.items(), desc="Training Models"):
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)

    mae = mean_absolute_error(ytest, y_pred)
    mse = mean_squared_error(ytest, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(ytest, y_pred)

    results[model_name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
    predictions[model_name] = model.predict(np.array([[10, 12, 54.10, 4241.05, 49.56, 9.0, 14.8, 491.32, 54.0, 4426.84]]))

results_df = pd.DataFrame(results).T
print("Model Performance:")
print(results_df)


best_model = results_df.loc[results_df['R2'].idxmax()]
print("\nBest Model:")
print(best_model)


print("\nPredicted SMPEP2 values for the new data point:")
for model_name, prediction in predictions.items():
    print(f"{model_name}: {prediction[0]}")


numeric_data = data.select_dtypes(include=[np.number])
correlations = numeric_data.corr(method='pearson')
plt.figure(figsize=(16, 12))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.title("Correlation Matrix")
plt.show()

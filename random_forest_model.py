from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from preprocess_dataset_for_training import create_training_data
import polars as pl
import time

joined_df = pl.read_parquet("data/joined_df.parquet")
X, y = create_training_data(joined_df)


param_grid = {
    "n_estimators": [10],
    "criterion": ["absolute_error"],
    "max_depth": [6],
    "max_features": ["sqrt"],
}
cv = KFold(n_splits=2, shuffle=True)
for target in y.columns:

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid=param_grid,
        return_train_score=True,
        cv=cv,
    ).fit(X, y[target])
    print(target, grid_search.cv_results_)

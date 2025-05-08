from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from preprocess_dataset_for_training import create_training_data
import polars as pl
import time

keep_cols = (
    "Időpont",
    "Negatív Mérlegköri kiegyenlítő energia egységára (HUF/kWh)",
    "Pozitív Mérlegköri kiegyenlítő energia egységára (HUF/kWh)",
    "Rendszer-irány (kWh)",
    "Naperőművek becsült termelése (aktuális)",
    "Naperőművek becsült termelése (intraday)",
    "Naperőművek becsült termelése (dayahead)",
    "HU-UK",
    "HU-SK",
    # "HU-SI", too many NaN values
    "HU-RS",
    "HU-AT",
    "HU-HR",
    "HU-RO",
    "HU-AT menetrend",
    "HU-HR menetrend",
    # "HU-SI menetrend (RIR NT)", too many NaN values
    "HU-SK menetrend",
    "HU-RS menetrend",
    "HU-UK menetrend",
    "HU-RO menetrend",
    "Bruttó terv erőművi termelés",
    "Bruttó tény erőművi termelés",
    "Bruttó hitelesített rendszerterhelés tény",
    "Bruttó rendszerterhelés becslés (dayahead)",
    "Szélerőművek becsült termelése (aktuális)",
    "Szélerőművek becsült termelése (dayahead)",
    "Szélerőművek becsült termelése (intraday)",
)
joined_df = pl.read_parquet("data/joined_df.parquet").select(keep_cols)
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
        estimator=RandomForestRegressor(n_jobs=-1),
        param_grid=param_grid,
        return_train_score=True,
        cv=cv,
    ).fit(X, y[target])
    print(target, grid_search.cv_results_)

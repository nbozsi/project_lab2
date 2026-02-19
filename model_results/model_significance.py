from scipy import stats
import numpy as np

base = dict(
    mae=np.load("model_results/mae_all_joined_df_no_weather.npy"),
    rmse=np.load("model_results/rmse_all_joined_df_no_weather.npy"),
    r2=np.load("model_results/r2_all_joined_df_no_weather.npy"),
)

weather = dict(
    mae=np.load("model_results/mae_all_joined_df_with_weather.npy"),
    rmse=np.load("model_results/rmse_all_joined_df_with_weather.npy"),
    r2=np.load("model_results/r2_all_joined_df_with_weather.npy"),
)

LAG = 0
targets = ("neg", "pos", "sys")
for LAG in range(5):
    for err in ("mae", "rmse", "r2"):
        # alternative='greater' tests if mean(A) > mean(B)
        # This is what you want if you're looking for Model B to have LOWER error.
        for i in range(3):
            t_stat, p_value = stats.ttest_rel(base[err][LAG * 3 + i], weather[err][LAG * 3 + i], alternative="greater")
            if p_value < 0.05:
                print(f"{LAG = }, {targets[i] = } {err = }: Model with weather is significantly better than without. ({p_value})")
            else:
                # print(f"No significant difference found. ({p_value})")
                pass

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, r2_score, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="modeling")
parser.add_argument('--results_path', default="results/process_result.txt")
parser.add_argument('--plot_save_dir', default="modeling_results")
args = parser.parse_args()

data = pd.read_csv(args.results_path, delim_whitespace=True)
features = ['N', 'N1', 'N2', 'N3', 'W', 'S1', 'S2', 'S3', 'C', 'F2', 'F3']
X = data[features]
y_multi = data[['T', 'T10', 'T25', 'T50', 'T75', 'T90']].values

y_binned = pd.qcut(data['T'], q=10, labels=False)  # Bin into 10 quantiles
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

numeric_features = ['N', 'N1', 'N2', 'N3', 'W', 'S1', 'S2', 'S3']
categorical_features = ['C', 'F2', 'F3']

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

mlp_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', MLPRegressor(random_state=42, max_iter=10000))])

def target_scorer(metric_func, target_idx):
    def score_func(y_true, y_pred):
        return metric_func(y_true[:, target_idx], y_pred[:, target_idx])
    return make_scorer(score_func, greater_is_better=(metric_func != mean_squared_error))


scorers = {}
for i in range(y_multi.shape[1]):
    scorers[f'r2_target_{i}'] = target_scorer(r2_score, i)
    def rmse_func(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    scorers[f'rmse_target_{i}'] = target_scorer(rmse_func, i)

cv_results = cross_validate(
    mlp_pipeline,
    X,
    y_multi,  # Use continuous target for evaluation
    cv=cv.split(X, y_binned),  # Use binned target for stratification
    scoring=scorers,
    return_train_score=False
)

for i in range(y_multi.shape[1]):
    r2_scores = cv_results[f'test_r2_target_{i}']
    rmse_scores = cv_results[f'test_rmse_target_{i}']
    print(f"Target {y_multi[i]}:")
    print(f"  R² scores per fold: {r2_scores}")
    print(f"  Mean R²: {r2_scores.mean():.4f}")
    print(f"  RMSE scores per fold: {rmse_scores}")
    print(f"  Mean RMSE: {rmse_scores.mean():.4f}")
    print()

## Plot the results
r2_means = []
r2_stds = []
rmse_means = []
rmse_stds = []

for i in range(y_multi.shape[1]):
    r2_scores = cv_results[f'test_r2_target_{i}']
    rmse_scores = cv_results[f'test_rmse_target_{i}']
    r2_means.append(np.mean(r2_scores))
    r2_stds.append(np.std(r2_scores))
    rmse_means.append(np.mean(rmse_scores))
    rmse_stds.append(np.std(rmse_scores))

sns.set_theme(style="whitegrid", font_scale=1.2)

targets = np.arange(y_multi.shape[1])
target_labels = ["median", "10-th", "25-th", "50-th", "75-th", "90-th"]

# ----- Plot for R² -----
plt.figure(figsize=(6, 5))
plt.errorbar(targets, r2_means, yerr=r2_stds, fmt='o-', color='#1f77b4', 
             ecolor='black', capsize=5, linewidth=2, markersize=8)
plt.title(r'$R^2$ Performance per Target', fontsize=14)
plt.xlabel('Target Index', fontsize=12)
plt.ylabel(r'$R^2$ Score', fontsize=12)
plt.xticks(targets, target_labels)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim([0.95, 1.0]) 

# Save the R² plot
plt.tight_layout()
plt.savefig(f"{args.plot_save_dir}/model_performance_r2.png", dpi=300)
plt.show()

# ----- Plot for RMSE -----
plt.figure(figsize=(6, 5))
plt.errorbar(targets, rmse_means, yerr=rmse_stds, fmt='o-', color='#d62728', 
             ecolor='black', capsize=5, linewidth=2, markersize=8)
plt.title('RMSE Performance per Target', fontsize=14)
plt.xlabel('Target Index', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.xticks(targets, target_labels)
plt.grid(True, linestyle='--', alpha=0.7)

# Save the RMSE plot
plt.tight_layout()
plt.savefig(f"{args.plot_save_dir}/model_performance_rmse.png", dpi=300)
plt.show()
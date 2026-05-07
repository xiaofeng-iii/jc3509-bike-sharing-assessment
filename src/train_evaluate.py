"""训练评估脚本: Linear Regression vs Random Forest Regressor

输入:
  - data/processed/day_processed.csv (731×33)
  - data/processed/split_indices.json

输出:
  - results/metrics.csv
  - results/lr_coefficients.csv
  - results/rf_importances.csv
  - results/figures/*.png
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 路径配置
BASE = Path(__file__).parent.parent
DATA_CSV = BASE / "data" / "processed" / "day_processed.csv"
SPLIT_JSON = BASE / "data" / "processed" / "split_indices.json"
RESULTS_DIR = BASE / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

def load_data():
    """加载数据并按索引划分"""
    df = pd.read_csv(DATA_CSV)
    print(f"数据形状: {df.shape}")

    with open(SPLIT_JSON) as f:
        splits = json.load(f)

    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    test_idx = splits['test_idx']

    X = df.drop(columns=['cnt'])
    y = df['cnt']

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    print(f"训练集: {len(train_idx)} 样本")
    print(f"验证集: {len(val_idx)} 样本")
    print(f"测试集: {len(test_idx)} 样本")
    print(f"特征数: {X.shape[1]}")

    return X_train, y_train, X_val, y_val, X_test, y_test, X.columns.tolist()

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    """评估模型在三个数据集上的性能"""
    results = []

    for dataset_name, X, y in [('train', X_train, y_train),
                                ('val', X_val, y_val),
                                ('test', X_test, y_test)]:
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        r2 = r2_score(y, y_pred)

        results.append({
            'Model': model_name,
            'Dataset': dataset_name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R²': round(r2, 2)
        })

        print(f"  {dataset_name:5s}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")

    return results

def train_linear_regression(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """训练 Linear Regression 并评估"""

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("模型训练完成")

    results = evaluate_model(lr, X_train, y_train, X_val, y_val, X_test, y_test, 'Linear Regression')

    # 提取系数
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lr.coef_,
        'Abs_Coefficient': np.abs(lr.coef_)
    })
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False).head(15)
    coef_df['Coefficient'] = coef_df['Coefficient'].round(2)
    coef_df['Abs_Coefficient'] = coef_df['Abs_Coefficient'].round(2)


    return lr, results, coef_df

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """训练 Random Forest（含超参调优）"""
    print("\n=== 训练 Random Forest ===")

    # 超参搜索
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20]
    }

    best_rmse = float('inf')
    best_params = None

    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            rf = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=max_d,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            val_pred = rf.predict(X_val)
            val_rmse = mean_squared_error(y_val, val_pred, squared=False)

            print(f"  n_estimators={n_est}, max_depth={max_d}: val_RMSE={val_rmse:.2f}")

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_params = {'n_estimators': n_est, 'max_depth': max_d}

    print(f"\n最佳超参: {best_params}")
    print(f"最佳验证 RMSE: {best_rmse:.2f}")

    # 用最佳超参在 train+val 上重训
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    rf_final = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        random_state=42,
        n_jobs=-1
    )
    rf_final.fit(X_train_val, y_train_val)
    print("最终模型训练完成")

    # 评估
    # val 已被 RF 最终模型见过，因此标注为 'train_seen_val' 以避免被误读为独立验证性能
    results = []
    for dataset_name, X, y in [('train', X_train, y_train),
                                ('train_seen_val', X_val, y_val),
                                ('test', X_test, y_test)]:
        y_pred = rf_final.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        r2 = r2_score(y, y_pred)

        results.append({
            'Model': 'Random Forest',
            'Dataset': dataset_name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R²': round(r2, 2)
        })

        print(f"  {dataset_name:15s}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")

    # 提取特征重要性
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_final.feature_importances_
    })
    imp_df = imp_df.sort_values('Importance', ascending=False).head(15)
    imp_df['Importance'] = imp_df['Importance'].round(3)

    print(f"\nTop-15 特征重要性:")
    print(imp_df.to_string(index=False))

    return rf_final, results, imp_df, best_params

def save_results(lr_results, rf_results, lr_coef, rf_imp, rf_best_params):
    """保存所有结果文件"""
    # 保存 metrics.csv
    all_results = lr_results + rf_results
    metrics_df = pd.DataFrame(all_results)
    metrics_path = RESULTS_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # 保存 lr_coefficients.csv
    lr_coef_path = RESULTS_DIR / "lr_coefficients.csv"
    lr_coef.to_csv(lr_coef_path, index=False)

    # 保存 rf_importances.csv
    rf_imp_path = RESULTS_DIR / "rf_importances.csv"
    rf_imp.to_csv(rf_imp_path, index=False)

    # 保存 rf_best_params.json
    params_path = RESULTS_DIR / "rf_best_params.json"
    with open(params_path, 'w') as f:
        json.dump(rf_best_params, f, indent=2)

    print("结果已保存")

def plot_prediction_scatter(lr_model, rf_model, X_test, y_test):
    """绘制预测 vs 真实散点图"""

    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    lr_r2 = r2_score(y_test, lr_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test, lr_pred, alpha=0.6, s=30, color='steelblue')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('True Values (cnt)')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'Linear Regression (R²={lr_r2:.3f})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_test, rf_pred, alpha=0.6, s=30, color='darkorange')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('True Values (cnt)')
    axes[1].set_ylabel('Predicted Values')
    axes[1].set_title(f'Random Forest (R²={rf_r2:.3f})')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = FIGURES_DIR / "prediction_scatter.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ {save_path}")

def plot_feature_importance(lr_coef, rf_imp):
    """绘制特征重要性对比图"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    lr_top = lr_coef.head(15).sort_values('Abs_Coefficient')
    axes[0].barh(lr_top['Feature'], lr_top['Abs_Coefficient'], color='steelblue')
    axes[0].set_xlabel('Absolute Coefficient')
    axes[0].set_title('Linear Regression - Top 15 Features')
    axes[0].grid(axis='x', alpha=0.3)

    rf_top = rf_imp.head(15).sort_values('Importance')
    axes[1].barh(rf_top['Feature'], rf_top['Importance'], color='darkorange')
    axes[1].set_xlabel('Feature Importance')
    axes[1].set_title('Random Forest - Top 15 Features')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ {save_path}")

def plot_residuals(lr_model, rf_model, X_test, y_test):
    """绘制残差分布图"""
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    lr_residuals = y_test - lr_pred
    rf_residuals = y_test - rf_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(lr_pred, lr_residuals, alpha=0.6, s=30, color='steelblue')
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals (True - Predicted)')
    axes[0].set_title('Linear Regression - Residuals')
    axes[0].grid(alpha=0.3)

    axes[1].scatter(rf_pred, rf_residuals, alpha=0.6, s=30, color='darkorange')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals (True - Predicted)')
    axes[1].set_title('Random Forest - Residuals')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = FIGURES_DIR / "residuals.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def main():
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_data()

    lr_model, lr_results, lr_coef = train_linear_regression(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    )

    rf_model, rf_results, rf_imp, rf_best_params = train_random_forest(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    )

    save_results(lr_results, rf_results, lr_coef, rf_imp, rf_best_params)

    plot_prediction_scatter(lr_model, rf_model, X_test, y_test)
    plot_feature_importance(lr_coef, rf_imp)
    plot_residuals(lr_model, rf_model, X_test, y_test)

    print_summary(lr_results, rf_results, rf_best_params)

if __name__ == "__main__":
    main()

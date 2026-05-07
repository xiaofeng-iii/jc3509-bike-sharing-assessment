"""day.csv 预处理脚本

输入: data/raw/day.csv
输出:
  - data/processed/day_processed.csv (731×33)
  - data/processed/split_indices.json
"""
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# 路径配置
BASE = Path(__file__).parent.parent
INPUT = BASE / "data" / "raw" / "day.csv"
OUTPUT_DIR = BASE / "data" / "processed"
OUTPUT_CSV = OUTPUT_DIR / "day_processed.csv"
OUTPUT_JSON = OUTPUT_DIR / "split_indices.json"

def main():
    df = pd.read_csv(INPUT)

    cols_to_drop = ['instant', 'dteday', 'casual', 'registered', 'temp']
    df = df.drop(columns=cols_to_drop)

    cat_cols = ['season', 'weathersit', 'mnth', 'weekday']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    cols = [c for c in df.columns if c != 'cnt'] + ['cnt']
    df = df[cols]

    features = df.drop(columns=['cnt'])
    print(f"特征值范围: [{features.min().min():.4f}, {features.max().max():.4f}]")
    out_of_range = ((features < 0) | (features > 1)).sum().sum()
    if out_of_range > 0:
        print(f"警告: {out_of_range} 个特征值超出 [0, 1] 范围")
    else:
        print("所有特征值在 [0, 1] 范围内")

    indices = df.index.tolist()
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42
    )

    assert len(set(train_idx) & set(val_idx)) == 0, "训练集与验证集有重叠"
    assert len(set(train_idx) & set(test_idx)) == 0, "训练集与测试集有重叠"
    assert len(set(val_idx) & set(test_idx)) == 0, "验证集与测试集有重叠"
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(indices), "索引总数不匹配"
    print("划分验证通过")

    split_data = {
        "train_idx": sorted(train_idx),
        "val_idx": sorted(val_idx),
        "test_idx": sorted(test_idx)
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(split_data, f, indent=2)

    df.to_csv(OUTPUT_CSV, index=False)

    final_df = pd.read_csv(OUTPUT_CSV)
    with open(OUTPUT_JSON) as f:
        splits = json.load(f)

    checks = [
        ("形状 (731, 33)", final_df.shape == (731, 33)),
        ("无 NaN", final_df.isnull().sum().sum() == 0),
        ("特征范围 [0,1]", ((final_df.drop(columns=['cnt']) >= 0) & (final_df.drop(columns=['cnt']) <= 1)).all().all()),
        ("索引总数 731", len(splits['train_idx']) + len(splits['val_idx']) + len(splits['test_idx']) == 731),
        ("最后一列是 cnt", final_df.columns[-1] == 'cnt')
    ]

    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {name}")

    if all_passed:
        print("\n所有验证通过，预处理完成")
    else:
        print("\n部分验证失败，请检查")

if __name__ == "__main__":
    main()

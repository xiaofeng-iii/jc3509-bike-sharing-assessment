"""day.csv preprocessing script

Input: data/raw/day.csv
Output:
  - data/processed/day_processed.csv (731×33)
  - data/processed/split_indices.json
"""
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Path configuration
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
    print(f"Feature value range: [{features.min().min():.4f}, {features.max().max():.4f}]")
    out_of_range = ((features < 0) | (features > 1)).sum().sum()
    if out_of_range > 0:
        print(f"Warning: {out_of_range} feature values out of [0, 1] range")
    else:
        print("All feature values within [0, 1] range")

    indices = df.index.tolist()
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42
    )

    assert len(set(train_idx) & set(val_idx)) == 0, "Train and val sets overlap"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train and test sets overlap"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val and test sets overlap"
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(indices), "Index count mismatch"
    print("Split validation passed")

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
        ("Shape (731, 33)", final_df.shape == (731, 33)),
        ("No NaN", final_df.isnull().sum().sum() == 0),
        ("Feature range [0,1]", ((final_df.drop(columns=['cnt']) >= 0) & (final_df.drop(columns=['cnt']) <= 1)).all().all()),
        ("Index count 731", len(splits['train_idx']) + len(splits['val_idx']) + len(splits['test_idx']) == 731),
        ("Last column is cnt", final_df.columns[-1] == 'cnt')
    ]

    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {name}")

    if all_passed:
        print("\nAll validations passed, preprocessing complete")
    else:
        print("\nSome validations failed, please check")

if __name__ == "__main__":
    main()

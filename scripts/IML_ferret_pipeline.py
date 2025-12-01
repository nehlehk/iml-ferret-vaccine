"""
Unified IML vs Baseline Models Pipeline
=======================================

This script compares an interpretable TreeFARMS model against three standard
"black-box" baselines on a ferret vaccination multi-omics dataset:

    - TreeFARMS  (optimal sparse decision trees over a Rashomon set)
    - Random Forest
    - LASSO Logistic Regression (L1-penalised)
    - Shallow Decision Tree

For each method, the pipeline:

    1. Respects the ferret-level grouping by using 5-fold cross-validation
       where all samples from a given ferret are kept in the same fold.
    2. Trains models on the same train/test splits.
    3. Computes test accuracy per fold.
    4. Aggregates feature importance across folds.
    5. Saves:
         - model_accuracy_summary.csv
         - top20_features_<Model>.csv
         - accuracy_comparison.png

TreeFARMS uses thresholded (binary) features computed once on the full dataset.
Baselines are trained on the original continuous omics features.

Author: Nahleh Kargarfard (2025)
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from treefarms import TREEFARMS
from treefarms.model.threshold_guess import compute_thresholds, cut


# =====================================================================
# 0. Configuration
# =====================================================================

# Path to the processed ferret multi-omics dataset
DATA_PATH = Path(
    "data/task6_vax2_processed_all_2filters.csv"
    # Replace with your actual path if needed, e.g.:
    # "/datasets/work/hb-covid-fda/work/local/iml/vaccine_data/Processed_data_task6/task6_vax2_processed_all_2filters.csv"
)

# Directory where all outputs will be written
OUTPUT_DIR = Path("results/ml_comparison_pipeline")

# Ferret-level cross-validation settings
N_SPLITS = 5
RANDOM_STATE = 20

# TreeFARMS configuration
USE_THRESHOLDS = True
TREEFARMS_CONFIG = {
    "regularization": 0.008,
    "rashomon_bound_multiplier": 0.05,
}


# =====================================================================
# 1. Helper functions
# =====================================================================

def load_data(path: Path) -> pd.DataFrame:
    """Load the processed vaccination dataset."""
    df = pd.read_csv(path, sep=",")
    # Extract numeric ID from sample ID for ferret mapping
    df["Num"] = df.iloc[:, 0].str.split("_").str[-1].astype(int)
    return df


def get_ferret_ids(df: pd.DataFrame) -> List[int]:
    """
    Infer the number of ferrets from the dataset structure.

    In this dataset there are 10 samples per ferret, so the number of
    ferrets is len(df) // 10.
    """
    total_ferrets = len(df) // 10
    return list(range(1, total_ferrets + 1))


def get_ferret_data(
    df: pd.DataFrame,
    ferret_number: int,
    start_id: int = 145,
    step: int = 16,
) -> pd.DataFrame:
    """
    Subset the dataframe to rows belonging to a given ferret.

    This follows the original mapping logic used in the project,
    based on encoded numeric IDs in the sample names.
    """
    target_ids = [start_id + (ferret_number - 1) + i * step for i in range(16)]
    ferret_ids = [
        f"INO_21_4311_{id_num}" for id_num in target_ids if id_num in df["Num"].values
    ]
    return df[df.iloc[:, 0].isin(ferret_ids)]


def get_feature_importances(model, columns: List[str]) -> pd.Series:
    """Extract feature importances or (absolute) coefficients."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=columns)
    if hasattr(model, "coef_"):
        # LASSO: use absolute coefficient magnitude as importance
        return pd.Series(np.abs(model.coef_[0]), index=columns)
    # Fallback: all zeros if nothing is available
    return pd.Series(0.0, index=columns)


def aggregate_feature_importance(feature_list: List[pd.Series]) -> pd.Series:
    """
    Aggregate importances from multiple folds by taking the mean importance
    per feature across folds.

    Missing features (not selected in some folds) are treated as zero.
    """
    if not feature_list:
        return pd.Series(dtype=float)
    df_imp = pd.concat(feature_list, axis=1).fillna(0.0)
    return df_imp.mean(axis=1)


def map_treefarms_features(
    tree_features: List[object], feature_names: List[str]
) -> List[str]:
    """
    Map TreeFARMS feature indices / names to actual column names.

    TreeFARMS trees may store features as indices (ints) or strings like
    'feature_12'. This function normalises them to the corresponding
    feature name in `feature_names`.
    """
    mapped = []
    for f in tree_features:
        # case 1: already a valid index
        if isinstance(f, int):
            if 0 <= f < len(feature_names):
                mapped.append(feature_names[f])
            continue

        f_str = str(f)

        # case 2: "feature_12" style
        if f_str.startswith("feature_"):
            try:
                idx = int(f_str.split("_")[1])
                if 0 <= idx < len(feature_names):
                    mapped.append(feature_names[idx])
                continue
            except ValueError:
                pass  # fall through

        # case 3: assume it's already a usable name
        mapped.append(f_str)
    return mapped


# =====================================================================
# 2. Main pipeline
# =====================================================================

def run_pipeline() -> None:
    """Run TreeFARMS vs baseline models and save results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 2.1 Load data and set up CV
    # -----------------------------
    df = load_data(DATA_PATH)
    ferret_ids = get_ferret_ids(df)

    full_df = df.drop(columns=["Num"])
    y_full = full_df.iloc[:, 1]
    X_full = full_df.iloc[:, 2:]
    original_feature_names = X_full.columns.tolist()

    # -----------------------------
    # 2.2 Thresholding for TreeFARMS
    # -----------------------------
    if USE_THRESHOLDS:
        X_full_bin, thresholds, header, _ = compute_thresholds(
            X_full.copy(), y_full, 40, 1
        )
        print(f"[INFO] Using thresholded binary features: {X_full_bin.shape}")
    else:
        X_full_bin = X_full.copy()
        thresholds, header = None, original_feature_names

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # -----------------------------
    # 2.3 Models and result holders
    # -----------------------------
    model_names = ["TreeFARMS", "RandomForest", "LASSO", "DecisionTree"]

    accuracies: Dict[str, List[float]] = {m: [] for m in model_names}
    errors: Dict[str, List[float]] = {m: [] for m in model_names}
    feature_imps: Dict[str, List[pd.Series]] = {m: [] for m in model_names}

    # -----------------------------
    # 2.4 Cross-validation loop
    # -----------------------------
    for fold_idx, (train_index, test_index) in enumerate(kf.split(ferret_ids), start=1):
        print(f"\n[INFO] ===== Fold {fold_idx} / {N_SPLITS} =====")

        train_ferrets = [ferret_ids[i] for i in train_index]
        test_ferrets = [ferret_ids[i] for i in test_index]

        # Build fold-specific data
        train_df = pd.concat([get_ferret_data(df, f) for f in train_ferrets])
        test_df = pd.concat([get_ferret_data(df, f) for f in test_ferrets])

        train_df = train_df.drop(columns=["Num"])
        test_df = test_df.drop(columns=["Num"])

        y_train = train_df.iloc[:, 1]
        y_test = test_df.iloc[:, 1]
        X_train = train_df.iloc[:, 2:]
        X_test = test_df.iloc[:, 2:]

        # Ensure consistent column order
        X_train = X_train[original_feature_names]
        X_test = X_test[original_feature_names]

        # Binary version for TreeFARMS
        if USE_THRESHOLDS:
            X_train_bin = cut(X_train.copy(), thresholds)[header]
            X_test_bin = cut(X_test.copy(), thresholds)[header]
            bin_feature_names = list(X_train_bin.columns)
        else:
            X_train_bin = X_train.copy()
            X_test_bin = X_test.copy()
            bin_feature_names = original_feature_names

        # -------------------------
        # 2.4.1 Baseline models
        # -------------------------

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        accuracies["RandomForest"].append(acc_rf)
        errors["RandomForest"].append(1.0 - acc_rf)
        feature_imps["RandomForest"].append(
            get_feature_importances(rf, original_feature_names)
        )

        # Decision Tree (shallow)
        dt = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        acc_dt = accuracy_score(y_test, y_pred_dt)
        accuracies["DecisionTree"].append(acc_dt)
        errors["DecisionTree"].append(1.0 - acc_dt)
        feature_imps["DecisionTree"].append(
            get_feature_importances(dt, original_feature_names)
        )

        # LASSO Logistic Regression
        lasso = LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=2000
        )
        lasso.fit(X_train, y_train)
        y_pred_lasso = lasso.predict(X_test)
        acc_lasso = accuracy_score(y_test, y_pred_lasso)
        accuracies["LASSO"].append(acc_lasso)
        errors["LASSO"].append(1.0 - acc_lasso)
        feature_imps["LASSO"].append(
            get_feature_importances(lasso, original_feature_names)
        )

        # -------------------------
        # 2.4.2 TreeFARMS (IML)
        # -------------------------
        forest = TREEFARMS(TREEFARMS_CONFIG)
        forest.fit(X_train_bin, y_train)

        n_trees = forest.get_tree_count()
        print(f"[INFO] TreeFARMS Rashomon set size: {n_trees} trees")

        test_accs = []
        all_feats_this_fold: List[str] = []

        for i in range(n_trees):
            tree = forest[i]
            # Accuracy for this tree
            acc_test_tree = tree.score(X_test_bin, y_test)
            test_accs.append(acc_test_tree)

            # Feature usage for this tree
            mapped_feats = map_treefarms_features(
                tree.features(), bin_feature_names
            )
            all_feats_this_fold.extend(mapped_feats)

        # Average accuracy over all near-optimal trees
        fold_acc_treefarms = float(np.mean(test_accs))
        accuracies["TreeFARMS"].append(fold_acc_treefarms)
        errors["TreeFARMS"].append(1.0 - fold_acc_treefarms)

        # Feature "importance" = frequency of use across the Rashomon set
        feature_imps["TreeFARMS"].append(pd.Series(Counter(all_feats_this_fold)))

    # =================================================================
    # 3. Significance tests and summaries
    # =================================================================

    # Accuracy comparison: TreeFARMS vs each baseline
    tree_acc = np.array(accuracies["TreeFARMS"])
    print("\n[INFO] ===== Accuracy Comparison (TreeFARMS vs baselines) =====")
    for name in ["RandomForest", "LASSO", "DecisionTree"]:
        acc_arr = np.array(accuracies[name])
        t_p = ttest_rel(tree_acc, acc_arr).pvalue
        w_p = wilcoxon(tree_acc, acc_arr).pvalue
        print(
            f"{name:>12} | mean={acc_arr.mean():.3f}, sd={acc_arr.std():.3f}, "
            f"t-test p={t_p:.4f}, wilcoxon p={w_p:.4f}"
        )

    # Accuracy summary table
    summary = pd.DataFrame(
        {
            "Model": model_names,
            "MeanAccuracy": [np.mean(accuracies[m]) for m in model_names],
            "StdDev": [np.std(accuracies[m]) for m in model_names],
        }
    )
    summary.to_csv(OUTPUT_DIR / "model_accuracy_summary.csv", index=False)
    print("\n[INFO] Saved model_accuracy_summary.csv")

    # =================================================================
    # 4. Top-20 features per model
    # =================================================================

    for name in model_names:
        mean_imp = aggregate_feature_importance(feature_imps[name])
        if mean_imp.empty:
            continue
        top20 = mean_imp.sort_values(ascending=False).head(20)
        top20.to_csv(OUTPUT_DIR / f"top20_features_{name}.csv")
        print(f"[INFO] Saved top20_features_{name}.csv")

    # =================================================================
    # 5. Accuracy plot
    # =================================================================

    plt.figure(figsize=(7, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]  # one colour per model

    plt.bar(
        summary["Model"],
        summary["MeanAccuracy"],
        yerr=summary["StdDev"],
        capsize=5,
        color=colors,
        edgecolor="black",
    )
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("Cross-validated Accuracy Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Simple legend explaining the error bars
    plt.errorbar(
        [], [], yerr=[], capsize=5, fmt="none", ecolor="black", label="± 1 SD"
    )
    plt.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=300)
    plt.close()
    print("[INFO] Saved accuracy_comparison.png")

    print("\n[INFO] Analysis complete. Results written to:", OUTPUT_DIR.resolve())


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    run_pipeline()

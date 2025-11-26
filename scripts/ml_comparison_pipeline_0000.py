"""
Unified IML vs. Baseline Models Pipeline
----------------------------------------
Performs cross-validated training of TreeFARMS, Random Forest, XGBoost,
LASSO (Logistic Regression with L1), and a simple Decision Tree on the same
ferret vaccination dataset. Outputs accuracy metrics, significance tests,
and top-20 feature importance tables.

Author: Nahleh Kargarfard, 2025
"""

# =============================================================
# 0. Imports and Setup
# =============================================================
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from treefarms import TREEFARMS  
from model.threshold_guess import compute_thresholds, cut
from treefarms.model.threshold_guess import compute_thresholds, cut
from treefarms.model.model_set import ModelSetContainer
from sklearn.inspection import permutation_importance
import re



def get_ferret_data(df, ferret_number, start_id=145, step=16):
    target_ids = [start_id + (ferret_number - 1) + i * step for i in range(16)]
    ferret_ids = [f"INO_21_4311_{id_num}" for id_num in target_ids if id_num in df['Num'].values]
    return df[df.iloc[:, 0].isin(ferret_ids)]

def replace_features(rule_string, real_features):
    def repl(match):
        idx = int(match.group(1))
        if idx < len(real_features):
            return real_features[idx]
        return match.group(0)
    return re.sub(r'feature_(\d+)', repl, rule_string)

# Paths
rules_dir = Path("/datasets/work/hb-covid-fda/work/local/iml/vaccine_data/pipeline_ferret/ml_comparison_results")
rules_dir.mkdir(exist_ok=True, parents=True)

# =============================================================
# 1. Load Data
# =============================================================
df = pd.read_csv("/datasets/work/hb-covid-fda/work/local/iml/vaccine_data/Processed_data_task6/task6_vax2_processed_all_2filters.csv", sep=",")

df['Num'] = df.iloc[:,0].str.split('_').str[-1].astype(int)
ferret_ids = list(range(1, len(df)//10 + 1))

y_full = df.iloc[:, 1]
X_full = df.iloc[:, 2:]
header = X_full.columns.tolist()

# =============================================================
# 2. Thresholding (for interpretability comparability)
# =============================================================
use_thresholds = True

if use_thresholds:
    X_full_guessed, thresholds, header, _ = compute_thresholds(X_full.copy(), y_full, 40, 1)
    print(f"Using thresholded binary features: {X_full_guessed.shape}")
else:
    X_full_guessed = X_full.copy()

kf = KFold(n_splits=5, shuffle=True, random_state=20)

# =============================================================
# 3. Helper Functions
# =============================================================
def get_feature_importances(model, columns):
    """Extract feature importances or coefficients."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=columns)
    elif hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_[0]), index=columns)
    else:
        return pd.Series(0, index=columns)

def aggregate_feature_importance(feature_list):
    """Aggregate importances from multiple folds."""
    df_imp = pd.concat(feature_list, axis=1)
    return df_imp.mean(axis=1)

# =============================================================
# 4. Model Setup
# =============================================================
models = {
    "TreeFARMS": "treefarms",
    "RandomForest": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    "LASSO": LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42)
}

accuracies = {m: [] for m in models}
errors = {m: [] for m in models}
feature_importances = {m: [] for m in models}

# ---- TreeFARMS ----
accuracies_test = []
accuracies_train = []
all_paths = []
all_features = []

# =============================================================
# 5. Cross-Validation Loop
# =============================================================
for fold_idx, (train_index, test_index) in enumerate(kf.split(ferret_ids)):
    print(f"\n===== Fold {fold_idx+1} =====")
    train_ferrets = [ferret_ids[i] for i in train_index]
    test_ferrets = [ferret_ids[i] for i in test_index]

    train_df = pd.concat([get_ferret_data(df, f) for f in train_ferrets])
    test_df = pd.concat([get_ferret_data(df, f) for f in test_ferrets])

    y_train, y_test = train_df.iloc[:, 1], test_df.iloc[:, 1]
    X_train, X_test = train_df.iloc[:, 2:], test_df.iloc[:, 2:]

    if use_thresholds:
        X_train = cut(X_train.copy(), thresholds)[header]
        X_test = cut(X_test.copy(), thresholds)[header]

    # ---- TreeFARMS ----
    
    forest = TREEFARMS({"regularization": 0.008, "rashomon_bound_multiplier": 0.05})
    forest.fit(X_train, y_train)
    treesNum = forest.get_tree_count()

    fold_train_accs = []
    fold_test_accs = []
    real_features = X_train.columns.tolist()

    for i in range(treesNum):
        tree = forest[i]
        acc_train = tree.score(X_train, y_train)
        acc_test = tree.score(X_test, y_test)
        fold_train_accs.append(acc_train)
        fold_test_accs.append(acc_test)

        mapped_features = []
        for f in tree.features():
            if isinstance(f, int):
                idx = f
            elif str(f).startswith("feature_"):
                idx = int(str(f).split("_")[1])
            else:
                mapped_features.append(f)
                continue
            if idx < len(real_features):
                mapped_features.append(real_features[idx])
        all_features.extend(mapped_features)

        tree_rules = str(tree)
        tree_rules_fixed = replace_features(tree_rules, real_features)
        out_file = rules_dir / f"fold{fold_idx}_tree{i}.txt"
        with open(out_file, "w") as f:
            f.write(tree_rules_fixed)

        lines = tree_rules_fixed.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("if") or line.startswith("else if"):
                condition_real = line
                for feat in real_features:
                    if feat in condition_real:
                        all_features.append(feat)
            elif line.startswith("predicted"):
                prediction = line
                path = f"{condition_real} --> {prediction}"
                all_paths.append(path)

    accuracies_train.append(sum(fold_train_accs) / len(fold_train_accs))
    accuracies_test.append(sum(fold_test_accs) / len(fold_test_accs))

    with open(rules_dir / "rashomon_summary.csv", "a") as f:
        if fold_idx == 0:
            f.write("Fold,Train_Acc,Test_Acc,Num_Trees,Num_Features\n")
        f.write(f"{fold_idx+1},{sum(fold_train_accs)/len(fold_train_accs):.4f},{sum(fold_test_accs)/len(fold_test_accs):.4f},{forest.get_tree_count()},{X_train.shape[1]}\n")





    # train_accs = [forest[i].score(X_train, y_train) for i in range(forest.get_tree_count())]
    # test_accs  = [forest[i].score(X_test, y_test) for i in range(forest.get_tree_count())]
    # accuracies["TreeFARMS"].append(np.mean(train_accs))
    # errors["TreeFARMS"].append(1 - accuracies["TreeFARMS"][-1])

    # # Feature frequency across near-optimal trees
    # all_feats = [f for i in range(forest.get_tree_count()) for f in forest[i].features()]
    # feature_importances["TreeFARMS"].append(pd.Series(Counter(all_feats)))

    # ---- Other Models ----
    for name, model in models.items():
        if name == "TreeFARMS":
            continue
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies[name].append(accuracy_score(y_test, y_pred))
        errors[name].append(1 - accuracies[name][-1])
        feature_importances[name].append(get_feature_importances(model, X_train.columns))

# =============================================================
# 6. Significance Tests (TreeFARMS vs Others)
# =============================================================
# Summary stats
path_counts = Counter(all_paths)
feature_counts = Counter(all_features)

with open(rules_dir / f"rashomon_path_overlap.csv", "w") as f:
    f.write("Path,Count\n")
    for path, count in path_counts.most_common():
        f.write(f"\"{path}\",{count}\n")

with open(rules_dir / f"rashomon_feature_stability.csv", "w") as f:
    f.write("Feature,Count\n")
    for feat, count in feature_counts.most_common():
        f.write(f"{feat},{count}\n")

print("\nAll done!")
print(f"Average train accuracy: {sum(accuracies_train)/len(accuracies_train):.4f}")
print(f"Average test accuracy: {sum(accuracies_test)/len(accuracies_test):.4f}")


print("\n===== Accuracy Comparison =====")
treefarms_acc = np.array(accuracies["TreeFARMS"])
for name in models:
    if name == "TreeFARMS":
        continue
    acc_arr = np.array(accuracies[name])
    t_p = ttest_rel(treefarms_acc, acc_arr).pvalue
    w_p = wilcoxon(treefarms_acc, acc_arr).pvalue
    print(f"{name}: mean={acc_arr.mean():.3f}, sd={acc_arr.std():.3f}, t-test p={t_p:.4f}, wilcoxon p={w_p:.4f}")

# =============================================================
# 7. Export Top Features
# =============================================================
for name in feature_importances:
    mean_imp = aggregate_feature_importance(feature_importances[name])
    top20 = mean_imp.sort_values(ascending=False).head(20)
    top20.to_csv(rules_dir / f"top20_features_{name}.csv")
    print(f"\nTop 5 {name} features:\n{top20.head(5)}")


# =============================================================
# 8. Save Accuracy Summary
# =============================================================
summary = pd.DataFrame({
    "Model": list(models.keys()),
    "MeanAccuracy": [np.mean(accuracies[m]) for m in models],
    "StdDev": [np.std(accuracies[m]) for m in models],
})
summary.to_csv(rules_dir / "model_accuracy_summary.csv", index=False)

# =============================================================
# 9. Optional Visualization
# =============================================================
plt.figure(figsize=(7,5))
plt.bar(summary["Model"], summary["MeanAccuracy"], yerr=summary["StdDev"], capsize=5)
plt.ylabel("Accuracy")
plt.title("Cross-validated Accuracy Comparison")
plt.tight_layout()
plt.savefig(rules_dir / "accuracy_comparison.png", dpi=300)
plt.show()

print("\nAnalysis complete. Results saved in /results directory.")

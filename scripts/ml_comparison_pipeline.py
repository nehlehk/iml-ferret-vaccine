import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from model.threshold_guess import compute_thresholds, cut
from treefarms.model.threshold_guess import compute_thresholds, cut
from treefarms import TREEFARMS
from treefarms.model.model_set import ModelSetContainer
from collections import Counter, defaultdict
import re

# Load your dataset here
df = pd.read_csv("/datasets/work/hb-covid-fda/work/local/iml/vaccine_data/Processed_data_task6/task6_vax2_processed_all_2filters.csv", sep=",")

# Add helper column for ferret logic
df['Num'] = df.iloc[:, 0].str.split('_').str[-1].astype(int)

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


# def aggregate_feature_importance(feature_list):
#     """Aggregate importances from multiple folds."""
#     df_imp = pd.concat(feature_list, axis=1)
#     return df_imp.mean(axis=1)

# Rashomon output dir
rules_dir = Path("/datasets/work/hb-covid-fda/work/local/iml/vaccine_data/ML_comparison_pipeline/trees")
rules_dir.mkdir(exist_ok=True, parents=True)
output_dir = Path("/datasets/work/hb-covid-fda/work/local/iml/vaccine_data/ML_comparison_pipeline")
output_dir.mkdir(exist_ok=True, parents=True)

# KFold setup
total_ferrets = len(df) // 10
ferret_ids = list(range(1, total_ferrets + 1))
kf = KFold(n_splits=5, shuffle=True, random_state=20)


models = {
    "TreeFARMS",
    "RandomForest" ,
    "LASSO",
    "DecisionTree"
}

accuracies = {m: [] for m in models}
errors = {m: [] for m in models}
feature_importances = {m: [] for m in models}

accuracies_test = []
accuracies_train = []
all_paths = []
all_features = []

# Prepare data for threshold estimation
full_df = df.drop(columns=['Num'])
y_full = full_df.iloc[:, 1]
X_full = full_df.iloc[:, 2:]
h = list(X_full.columns)
X_full = pd.DataFrame(X_full, columns=h)

# Compute thresholds once on the entire dataset
X_full_guessed, thresholds, header, _ = compute_thresholds(X_full.copy(), y_full, 40, 1)

# Main CV loop
for fold_idx, (train_index, test_index) in enumerate(kf.split(ferret_ids)):
    train_ferrets = [ferret_ids[i] for i in train_index]
    test_ferrets = [ferret_ids[i] for i in test_index]

    train_df = pd.concat([get_ferret_data(df, f) for f in train_ferrets])
    test_df = pd.concat([get_ferret_data(df, f) for f in test_ferrets])

    train_df = train_df.drop(columns=['Num'])
    test_df = test_df.drop(columns=['Num'])

    y_train = train_df.iloc[:, 1]
    y_test = test_df.iloc[:, 1]
    X_train = train_df.iloc[:, 2:]
    X_test = test_df.iloc[:, 2:]
    h = list(train_df.columns[2:])

    X_train = pd.DataFrame(X_train, columns=h)
    X_test = pd.DataFrame(X_test, columns=h)

    X_train_guessed = cut(X_train.copy(), thresholds)[header]
    X_test_guessed = cut(X_test.copy(), thresholds)[header]

    #--------------------- Random Foresst -----------------------------
    RF = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    accuracies["RandomForest"].append(accuracy_score(y_test, y_pred))
    errors["RandomForest"].append(1 - accuracies["RandomForest"][-1])
    importances = RF.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': h,
        'Importance': importances
    })
    # Sort and take top 20
    top10 = importance_df.sort_values(by='Importance', ascending=False).head(10)
    # feature_importances["RandomForest"].append(top20)
    top10.to_csv(output_dir / f"top10_features_{"RandomForest"}.csv")

    #--------------------- /Random Foresst ----------------------------

    #--------------------- Descion tree -----------------------------
    DT = DecisionTreeClassifier(max_depth=5, random_state=42)
    DT.fit(X_train, y_train)
    y_pred = DT.predict(X_test)
    accuracies["DecisionTree"].append(accuracy_score(y_test, y_pred))
    errors["DecisionTree"].append(1 - accuracies["DecisionTree"][-1])
    importances = DT.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': h,
        'Importance': importances
    })
    # Sort and take top 20
    top10 = importance_df.sort_values(by='Importance', ascending=False).head(10)
    # feature_importances["DecisionTree"].append(top20)
    top10.to_csv(output_dir / f"top10_features_{"DecisionTree"}.csv")
    #--------------------- /Descion tree ----------------------------

     #--------------------- Lasso -----------------------------
    LR = LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracies["LASSO"].append(accuracy_score(y_test, y_pred))
    errors["LASSO"].append(1 - accuracies["LASSO"][-1])
    # Get feature coefficients
    coefficients = LR.coef_[0]

    # Compute absolute importance (since coefficients can be positive or negative)
    importance_df = pd.DataFrame({
        'Feature': h,
        'Coefficient': coefficients,
        'Importance': np.abs(coefficients)
    }).sort_values(by='Importance', ascending=False)
    # Sort and take top 20
    top10 = importance_df.sort_values(by='Importance', ascending=False).head(10)
    # feature_importances["LASSO"].append(top20)
    top10.to_csv(output_dir / f"top10_features_{"LASSO"}.csv")
    #--------------------- /Lasso ----------------------------

    config = {
        "regularization": 0.008,
        "rashomon_bound_multiplier": 0.05
    }

    forest = TREEFARMS(config)
    forest.fit(X_train_guessed, y_train)
    treesNum = forest.get_tree_count()

    fold_train_accs = []
    fold_test_accs = []
    real_features = X_train_guessed.columns.tolist()

    for i in range(treesNum):
        tree = forest[i]
        acc_train = tree.score(X_train_guessed, y_train)
        acc_test = tree.score(X_test_guessed, y_test)
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
    accuracies["TreeFARMS"].append(sum(fold_test_accs) / len(fold_test_accs))
    errors["TreeFARMS"].append(1 - accuracies["TreeFARMS"][-1])

    with open(output_dir / "rashomon_summary.csv", "a") as f:
        if fold_idx == 0:
            f.write("Fold,Train_Acc,Test_Acc,Num_Trees,Num_Features\n")
        f.write(f"{fold_idx+1},{sum(fold_train_accs)/len(fold_train_accs):.4f},{sum(fold_test_accs)/len(fold_test_accs):.4f},{forest.get_tree_count()},{X_train_guessed.shape[1]}\n")



# Summary stats
path_counts = Counter(all_paths)
feature_counts = Counter(all_features)

with open(output_dir / f"rashomon_path_overlap.csv", "w") as f:
    f.write("Path,Count\n")
    for path, count in path_counts.most_common():
        f.write(f"\"{path}\",{count}\n")

with open(output_dir / f"top10_features_TreeFARMS.csv", "w") as f:
    f.write("Feature,Count\n")
    for feat, count in feature_counts.most_common():
        f.write(f"{feat},{count}\n")

# with open(output_dir / f"rashomon_feature_stability.csv", "w") as f:
#     # 1️⃣ Header
#     f.write("Feature,Count\n")
    
#     # 2️⃣ Write feature_counts (existing code)
#     for feat, count in feature_counts.most_common():
#         f.write(f"{feat},{count}\n")
    
#     # 3️⃣ Separator line
#     f.write("\nModel,TopFeature,Importance\n")
    
    # print("I'm debugging feature imprtance here")
    # # 4️⃣ Append top-20 features from feature_importances
    # for model_name, feats in feature_importances.items():
    #     print(model_name)
    #     print(feats)
    #     # Convert to DataFrame and sort descending
    #     df = pd.DataFrame(list(feats.items()), columns=["Feature", "Importance"])
    #     top20 = df.sort_values(by="Importance", ascending=False).head(20)
        
    #     for _, row in top20.iterrows():
    #         f.write(f"{model_name},{row['Feature']},{row['Importance']:.5f}\n")




# =============================================================
# 8. Save Accuracy Summary
# =============================================================
summary = pd.DataFrame({
    "Model": list(models),
    "MeanAccuracy": [np.mean(accuracies[m]) for m in models],
    "StdDev": [np.std(accuracies[m]) for m in models],
})
summary.to_csv(output_dir / "model_accuracy_summary.csv", index=False)

# =============================================================
# 9. Optional Visualization
# =============================================================
plt.figure(figsize=(7,5))
plt.bar(summary["Model"], summary["MeanAccuracy"], yerr=summary["StdDev"], capsize=5 , color=['skyblue', 'lightgreen', 'salmon', 'orange'])
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)  
plt.title("Cross-validated Accuracy Comparison")
plt.errorbar([], [], yerr=[], capsize=5, label="Standard Deviation", color='black', fmt='none')
# plt.legend(loc="upper left", frameon=True)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig(output_dir / "accuracy_comparison.png", dpi=300)
plt.show()

print("\nAnalysis complete. Results saved in /results directory.")


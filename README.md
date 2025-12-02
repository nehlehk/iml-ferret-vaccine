# Interpretable Machine Learning Identifies Molecular Signatures of Vaccine Response in Ferret Multi-Omics Data 

🧬 Interpretable Machine Learning Identifies Molecular Signatures of Vaccine Response in Ferret Multi-Omics Data

This repository contains code and supplementary materials for the paper:
“Interpretable Machine Learning Reveals Molecular Signatures of Vaccine Response in Ferret Multi-Omics Data”
(Kargarfard et al., 2025, submitted to Briefings in Bioinformatics)

We apply the **TreeFARMS** framework (scalable optimal sparse decision trees with explicit Rashomon set enumeration) to a controlled ferret vaccination study, and compare it against common “black-box” baselines:

- Random Forest  
- L1-penalised Logistic Regression (LASSO)  
- Shallow Decision Tree  

The analysis demonstrates that:
1. **Interpretable models can match black-box performance** on a small-n, large-p multi-omics dataset.  
2. TreeFARMS + Rashomon analysis reveals **stable molecular signatures** of vaccine-induced immune response.  
3. The method is particularly suited to **few-samples / many-features** settings, typical in omics studies.


## Data

The pipeline assumes a processed CSV with the following structure:

  - Column 0: sample ID (e.g. INO_21_4311_0145)

  - Column 1: vaccination group label (status)

  - Columns 2+: omics features (transcriptomics, metabolomics, etc.)

In the original analysis, we used:
```
data/task6_vax2_processed_all_2filters.csv
```



This file is not included in the repository because it originates from an internal ferret vaccination experiment. If you have access to the dataset, place it under data/ with the same filename or update the path in ml_comparison_pipeline.py.

## Methods (summary)

### 1. Global thresholding for TreeFARMS

- Continuous features are discretised once using the TreeFARMS compute_thresholds function.

- We use parameters max_bins=40 and min_leaf=1 to guess informative cut points that balance granularity and stability.

- Each feature is converted into a small set of binary indicators, which are then used as input to TreeFARMS.

### 2. Cross-validation grouped by ferret

- We derive a helper column (Num) from the sample ID suffix to group samples belonging to the same ferret.

- 5-fold cross-validation is performed over ferrets, not individual samples, so all timepoints from a given animal are kept in the same fold.

### 3. TreeFARMS + Rashomon analysis

We fit TreeFARMS using configuration:
```
config = {
    "regularization": 0.008,
    "rashomon_bound_multiplier": 0.05,
}
```

For each training fold, TreeFARMS enumerates a Rashomon set of sparse decision trees whose loss is within a fixed bound of the globally optimal tree.

We record:

- mean train/test accuracy across all trees in the Rashomon set

- frequency of each feature across all trees (feature stability)

- frequency of each unique rule path (path stability)

- plain-text rule files for each tree (for inspection).

### 4. Baseline models

Baselines are trained on the raw continuous features (no binarisation):

- Random Forest (sklearn.ensemble.RandomForestClassifier)

- LASSO (sklearn.linear_model.LogisticRegression with L1 penalty)

- Decision Tree (sklearn.tree.DecisionTreeClassifier, shallow tree)

For each model we store:

- per-fold test accuracy

- per-fold feature importance (Gini / coefficient magnitude)

- top-k (e.g. top-10) most important features.

### 5. Feature comparison across models

- We average feature importance across folds for each baseline model.

- We then extract top-k features and construct:

  - a small core table (main text) with overlap between TreeFARMS and baselines for key biomarkers (e.g. AZGP1, CLEC4D, NLRP3, Ketoleucine, L-Alanine)

  - a larger supplementary table listing top-10 features for each method.


## How to run the pipeline

From the repository root:
```
# (Optional) create and activate a fresh virtualenv or conda env
pip install -r requirements.txt

# Run the main script
python scripts/IML_ferret_pipeline.py
```

The script will:

- load the processed dataset

- compute global thresholds for TreeFARMS

- perform 5-fold CV grouped by ferret

- train TreeFARMS + three baseline models

write all outputs to results/ (or whatever directory is configured in the script).

Key outputs include:

- results/model_accuracy_summary.csv – mean ± SD accuracy per model

- results/accuracy_comparison.png – bar plot of cross-validated accuracies

- results/treefarms_cv_summary.csv – per-fold Rashomon stats

- results/rashomon_feature_stability.csv – feature recurrence across near-optimal trees

- results/top10_features_*.csv – top‐10 features per model
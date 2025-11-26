# Interpretable Machine Learning for Vaccine Response in Ferrets

🧬 Interpretable Machine Learning Identifies Molecular Signatures of Vaccine Response in Ferret Multi-Omics Data

This repository contains code and supplementary materials for the paper:
“Interpretable Machine Learning Reveals Molecular Signatures of Vaccine Response in Ferret Multi-Omics Data”
(Kargarfard et al., 2025, submitted to Briefings in Bioinformatics)

📘 **Overview**

This study applies interpretable machine learning (IML) to multi-omics data from a controlled ferret vaccination experiment.
Using the TreeFARMS framework coupled with Rashomon set exploration, the workflow identifies compact, rule-based models that distinguish vaccinated from unvaccinated animals and reveal stable molecular signatures of vaccine response.

Unlike traditional black-box ensemble models, this approach provides transparent, biologically meaningful rules that can be directly examined and validated by researchers.

⚙️ **Workflow Summary**

The analysis proceeds in five main stages:

**Data preprocessing**

  - Import, QC filtering, imputation, normalization, and scaling of multi-omics features
  
  - Feature discretization using TreeFARMS threshold-guessing algorithm

**Cross-validation setup**

- 5-fold stratified CV by individual ferret to prevent data leakage

**Model training (TreeFARMS)**

- Global optimization via GOSDT solver (λ = 0.008)

- Generation of sparse, interpretable if–then rule sets

**Rashomon set exploration**

- Extraction of all near-optimal trees within 5 % of global optimum

- Quantification of rule and feature recurrence across models

**Interpretability & visualization**

- Aggregation of stable features and visualization of model diversity

- Comparison with Random Forest baseline

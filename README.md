# Predicting the Development of Inflammatory Bowel Disease from Routine Blood Tests 🩸
### A Weighted K-Nearest Neighbor Data Imputation Approach for Time Series Data

This repository describes a method for imputing missing data values in longitudinal tabular data.

#### Overview
1. Data Processing
2. Missing Data Imputation - A Weighted K-Nearest Neighbor Approach
3. Supervised Prediction of IBD Diagnosis
4. Unsupervised Learning from Blood Test Values

ML analysis of a longitudinal dataset of laboratory values from IBD patients and healthy controls five years prior to IBD diagnosis. <br>
### Table of Contents

**Data Processing**
- [IBD_long_dataset_expansion.ipynb](IBD_long_dataset_expansion.ipynb): Format data appropriately for imputation calculations.
- [IBD_long_dataset_prep.ipynb](IBD_long_dataset_prep.ipynb): Exploratory data analysis of the dataset to understand the level of missingness. This is highly dataset specific.<br>

**Missing Data Imputation - A Weighted K-Nearest Neighbor (wKNN) Approach**
- [wknn_imputation.py](wknn_imputation.py): All functions needed to implement the wKNN
- [weighted_knn_imputer.ipynb](weighted_knn_imputer.ipynb): Notebook implementation of the wKNN with a few more comments for clarity. Note that all relative paths need to be updated, this is an example of what was performed in one analysis.<br>

**Supervised Predition of IBD Diagnosis**
- [ml_models_ibd.ipynb](ml_models_ibd.ipynb): Machine learning models applied to the imputed dataset.<br>

**Unsupervised Learning from Blood Test Values**
- [ibd_clustering.ipynb](ibd_clustering.ipynb): UMAP analysis of blood test values<br>

**Miscellaneous**
- [years_1_2.ipynb](years_1_2.ipynb): Subgroup analysis of lab values from only years 1 and 2 of data collection (4 and 3 years prior to IBD diagnosis, respectively). 



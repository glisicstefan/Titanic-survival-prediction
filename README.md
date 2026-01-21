# Titanic Survival Prediction

## Project Overview
This project predicts the survival of passengers on the Titanic using Machine Learning. By analyzing passenger data (age, gender, socio-economic class, etc.), the goal is to build a predictive model that determines whether a passenger survived the disaster.

## Key Insights from EDA
- **Class Imbalance:** Only ~38% of passengers in the dataset survived. Because of this imbalance, I prioritized F1-Score and Precision-Recall over simple Accuracy to ensure the model effectively identifies survivors without being biased toward the majority class (non-survivors).
- **Gender:** Females had a significantly higher survival rate than males, following the "Women and children first" protocol.
- **Class:** Socio-economic status was a strong predictor; 1st class passengers had the highest survival probability.
- **Family Dynamics:** Travelling with a small family (2-4 members) increased survival chances compared to being alone or in a very large family.
- **Non-Linear Age Impact:** While age had a weak linear correlation with survival, children had a much higher chance of surviving, justifying the use of *Age Binning*.



## Feature Engineering
To improve model performance, I performed the following transformations:
- **Title Extraction:** Extracted titles (Mr, Miss, Mrs, Master) from names to more accurately impute missing Age values based on group medians.
- **Cabin Mapping:** Created a binary `Has_Cabin` feature, as having a recorded cabin was strongly correlated with higher social class and survival.
- **Age Binning:** Grouped ages into categories (Child, Teenager, Adult, etc.) to capture non-linear survival patterns.
- **Family Size:** Combined `SibSp` and `Parch` into a single feature to identify solo travelers vs. families.

## Model Performance & Evaluation
I compared two powerful ensemble algorithms using **GridSearchCV** for hyperparameter tuning and **5-fold Cross-Validation**.

| Model | CV Accuracy | Test Accuracy | F1-Score (Class 1) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest** | **83.1%** | **84.0%** | **0.8** | **Winner** |
| XGBoost | 84.3% | 82.0% | 0.76 | Overfitted |

**Why Random Forest?**
Although XGBoost showed a higher cross-validation accuracy during training, it failed to generalize as well on the unseen test set. **Random Forest** proved to be the more robust and stable model, maintaining high precision and recall (F1-score) across both training and testing phases.



## Project Structure
- `data/`: Contains `raw/` (original Kaggle data) and `processed/` (cleaned data) folders.
- `notebooks/`:
    - `01_eda.ipynb`: Data visualization and initial statistical analysis.
    - `02_preprocessing.ipynb`: Data cleaning, imputation, and feature engineering.
    - `03_modeling.ipynb`: Hyperparameter tuning, model comparison, and final evaluation.
- `requirements.txt`: List of Python libraries required to run the project.

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/titanic-survival-prediction.git](https://github.com/your-username/titanic-survival-prediction.git)
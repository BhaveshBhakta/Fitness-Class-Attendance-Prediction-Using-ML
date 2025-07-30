## Fitness Class Attendance Prediction

### Project Overview

This project aims to predict **whether a client will attend a fitness class** based on various factors related to their membership, booking behavior, and the class itself. By leveraging features such as months as a member, weight, days before booking, day of the week, time of class, and category, the goal is to develop a machine learning model that can forecast attendance, helping fitness centers optimize class scheduling and resource management.

-----

### Technical Highlights

  * **Dataset**: [Kaggle - DataCamp's Data Science Associate Certification](https://www.kaggle.com/datasets/ddosad/datacamps-data-science-associate-certification)
  * **Size**: 1500 entries, 8 columns
  * **Key Features**:
      * months\_as\_member, weight, days\_before, day\_of\_week, time, category.
  * **Approach**:
      * Data Cleaning: Filled missing values in the 'weight' column with its mean. Dropped 'booking\_id' as it's a unique identifier. No duplicates were found.
      * Exploratory Data Analysis: Histograms, Boxplots, and Heatmaps were used for visualization to understand data distributions and correlations.
      * Label Encoding: Applied to all categorical features and the target 'attended'.
      * Handling Class Imbalance with `SMOTE` (Synthetic Minority Over-sampling Technique) on the training data. This is important as the original dataset is imbalanced (1046 not attended vs 454 attended).
      * Binary Classification: The target variable 'attended' indicates class attendance (1) or non-attendance (0).
      * Models Used:
          * Logistic Regression, Ridge Classifier, SVC, Random Forest, XGBoost, AdaBoost, Gradient Boosting, Bagging, Decision Tree.
  * **Best Accuracy**:
      * 77.7% with Ridge Classifier.
      * 75.3% with Logistic Regression.
      * 72.0% with Gradient Boosting Classifier.

-----

### Purpose and Applications

  * Help fitness centers **predict class attendance**, optimizing class sizes and trainer allocation.
  * Reduce no-shows by identifying at-risk bookings for targeted reminders or interventions.
  * Improve member satisfaction by ensuring class availability and reducing overcrowding.
  * Support data-driven decision-making in fitness program development and scheduling.

-----

### Installation

Clone the repository:

```bash
git clone https://github.com/BhaveshBhakta/Fitness-Class-Attendance-Prediction-Using-ML.git
cd FFitness-Class-Attendance-Prediction-Using-ML
```

Install the necessary libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost
```

-----

### Collaboration

We welcome contributions to improve the project. You can help by:

  * Performing comprehensive hyperparameter tuning and cross-validation for all models to achieve optimal performance.
  * Exploring advanced feature engineering techniques, such as creating time-based features from 'days\_before' or 'time'.
  * Investigating alternative methods for handling categorical features (e.g., One-Hot Encoding) and comparing their impact.
  * Adding explainability (e.g., SHAP or LIME) to understand which factors most influence class attendance.

# Credit-Card-Fraud-Detection
## Problem Statement
Credit card fraud has surged due to the rise in digital transactions, posing severe financial risks. Traditional fraud detection methods struggle with evolving fraud patterns and high false-positive rates. This project leverages machine learning algorithms and SMOTE to enhance fraud detection accuracy. The goal is to develop a real-time, scalable system that efficiently identifies fraudulent transactions while minimizing false alarms.

## Data Dictionary
The dataset can be download using this link.

The credit card fraud detection dataset comprises 284,807 transactions from European cardholders in September 2013, with only 492 fraud cases (0.172%), making it highly imbalanced. It includes 28 PCA-transformed features and two untransformed ones: 'Time' (elapsed seconds) and 'Amount' (transaction value). The target variable, 'Class', denotes fraudulent (1) or legitimate (0) transactions. Due to the imbalance, evaluating model performance with AUPRC is recommended over traditional accuracy metrics.

## Project Pipeline
The project follows a structured pipeline to ensure an effective credit card fraud detection system:

__Data Understanding__: Load the dataset and analyze its structure, including feature distribution, class imbalance, and key characteristics such as PCA-transformed features, 'Time,' and 'Amount.'

__Exploratory Data Analysis (EDA)__: Perform univariate and bivariate analysis to identify patterns, detect anomalies, and explore relationships between variables. Investigate data skewness and apply necessary transformations for better model performance.

__Data Preprocessing__: Handle missing values, normalize numerical features if needed, and split the dataset into training and testing sets. Given the severe class imbalance, employ techniques such as SMOTE to balance the dataset.

__Model Building & Hyperparameter Tuning__: Train multiple machine learning models, including Logistic Regression, Decision Trees, Random Forest, XGBoost, and Neural Networks. Optimize hyperparameters using grid search or random search to improve model accuracy and generalization.

__Model Evaluation & Deployment__: Assess model performance using AUC-ROC, precision, recall, and F1-score to ensure accurate fraud detection. Deploy the best-performing model in a real-time fraud detection system for instant transaction verification and fraud prevention.


## Deployment

To deploy this project run

```bash
streamlit run your file path
```
![Image](https://github.com/user-attachments/assets/e6efcdc6-bb49-4ac0-8274-0f4a0ac9a99a)

![Image](https://github.com/user-attachments/assets/879419b9-ef1e-4666-906a-529cd14692e2)

## License

[MIT](https://choosealicense.com/licenses/mit/)

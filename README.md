# Credit Card Fraud Detection

## Project Overview
The aim of this project is to predict fraudulent credit card transactions using machine learning models. This is crucial from both the bank's and customers' perspectives, as banks cannot afford to lose their customers' money to fraudsters. Every fraud incident represents a loss to the bank, as they are responsible for fraudulent transactions.

## Table of Contents
- [Dataset](#Dataset)
- [Dataset Attributes](#Dataset Attributes)
- [Steps Involved] Steps Involved
- [Reading and Understanding the Data] Reading and Understanding the Data
- [Data Exploration and Visualization] Data Exploration and Visualization
- [Data Preprocessing] Data Preprocessing
- [Handling Imbalanced Data] Handling Imbalanced Data
- [Feature Scaling] Feature Scaling
- [Skewness Mitigation] Skewness Mitigation
- [Model Building] Model Building
- [Random Forest Model] Random Forest Model
- [XGBoost Model] XGBoost Model
- [Neural Network Model] Neural Network Model
- [Evaluation Metrics] Evaluation Metrics
- [Results and Visualizations] Results and Visualizations
- [Conclusion] Conclusion
- [Future Work] Future Work


## Dataset
The dataset contains transactions made over a period of two days in September 2013 by European credit cardholders. The dataset is highly unbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.

### Dataset Attributes
- **Time**: Time elapsed since the first transaction.
- **V1-V28**: PCA transformed features.
- **Amount**: Transaction amount.
- **Class**: Indicates whether a transaction is fraudulent (1) or not (0).

## Steps Involved
The project is organized into several key steps:

1. **Reading and Understanding the Data**
   - Load the dataset and perform initial data exploration.
   - Visualize class distributions and transaction amounts.

2. **Data Exploration and Visualization**
   - Conduct exploratory data analysis (EDA) to identify patterns in the data.

3. **Data Preprocessing**
   - Normalize features and handle missing values.
   - Address class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

4. **Feature Scaling**
   - Standardize features to improve model performance.

5. **Skewness Mitigation**
   - Check for skewness in features and apply PowerTransformer to normalize distributions.

6. **Model Building**
   - Train various machine learning models including Random Forest and XGBoost.
   - Develop a deep learning model using TensorFlow/Keras.

7. **Evaluation Metrics**
   - Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.

8. **Results and Visualizations**
   - Present visualizations that compare model performance metrics.

## Tools Used
- **Programming Language**: Python
- **Libraries & Framework**:
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - XGBoost
  - TensorFlow/Keras
  - Imbalanced-learn (for SMOTE)
- **Development Environment**: Jupyter Notebook

## Directory Structure:
credit-card-fraud-detection/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── data_preprocessing.ipynb
│   └── model_training.ipynb
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   ├── processed/
│   │   └── processed_data.csv
│   └── README.md
│
├── models/
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── neural_network_model.h5
│
├── visuals/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   └── model_performance.png
│
├── requirements.txt
└── README.md

## Installation and Setup
To run this project, ensure you have Python installed along with the required libraries. You can install the necessary libraries using pip:


## Running the Project
1. Clone or download this repository.
2. Navigate to the project folder.
3. Open Jupyter Notebook and run the notebooks in order:
   - `data_exploration.ipynb`: For exploratory data analysis.
   - `data_preprocessing.ipynb`: For data cleaning and preprocessing.
   - `model_training.ipynb`: For training models and evaluating performance.



## Usage
To run the project and train the model, execute the following command:
#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# The objective is to develop a robust classification model capable of identifying fraudulent credit card transactions accurately. By doing so, credit card companies can prevent unauthorized charges and protect their customers from financial losses.

# The dataset is highly unbalanced, fraud transactions account for 0.172% of all transactions. We need to take care of the data imbalance while building the model and come up with the best model by trying various algorithms.

# In[1]:


pip install imbalanced-learn


# In[2]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')




# In[3]:


#load dataset
data=pd.read_csv("D:/vijay Upgrad/Credit Card Project/creditcard.csv")

print(data.shape)
data.head()


# In[4]:


# Data Exploration and Analysis
# Check for missing values3322
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Data Preprocessing
data['Time'] = pd.to_datetime(data['Time'])  # Convert 'Time' column to datetime datatype

# Define features (X) and target variable (y)
X = data.drop(['Class'], axis=1)  # Features
y = data['Class']  # Target variable

# Remove constant features
data = data.loc[:, data.apply(pd.Series.nunique) != 1]


# In[5]:


data.describe()


# In[6]:


#distrinution for legit and fraundlent transaction
data['Class'].value_counts()


# In[7]:


non_fraudulent_percent = (data['Class'].value_counts()[0] / len(data)) * 100
fraudulent_percent = (data['Class'].value_counts()[1] / len(data)) * 100
print(non_fraudulent_percent)
print(fraudulent_percent)


# In[8]:


# Visualize the distribution of 'Class' (target variable)
plt.figure(figsize=(8, 6))
y.value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title('Distribution of Class (0: Non-fraudulent, 1: Fraudulent)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# Based on the graph, it is evident that the dataset is heavily imbalanced. The majority of the transactions are Non-Fraudulent (class 0), and only a small fraction of them are fraudulent (class 1). This indicates that the dataset has a class imbalance issue, which could potentially affect the performance of a model trained on this dataset. It may be necessary to use techniques like oversampling, undersampling, or class weighting to deal with the class imbalance problem when developing a model for fraud detection.
# 

# In[9]:


#Understanding patterns and relationships in the data
sns.set_style(style='white')
facet_grid = sns.FacetGrid(data=data, col='Class')
facet_grid.map(sns.scatterplot, 'Time', 'Amount', palette='Paired_r')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.show()


# In[10]:


# diffrentiating the fraud and legit data.
fraud = data[data['Class'] == 1]
legit = data[data['Class'] == 0]


# In[11]:


legit.Amount.describe()


# In[12]:


fraud.Amount.describe()


# In[13]:


# Lets check the fraudulent transactions occur more often during certain time frame

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(legit.Time, legit.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# In[14]:


# Feature Engineering
# Add new features
X['Transaction_hour'] = pd.to_datetime(X['Time'], unit='s').dt.hour
X['Normalized_amount'] = (X['Amount'] - X['Amount'].mean()) / X['Amount'].std()

#X = X.drop(['Time'], axis=1)


# SMOTE is used to balance the dataset as machine learning models perform better with balanced data.

# In[15]:


# Convert 'Time' feature to numerical format (e.g., seconds)
X['Time_seconds'] = (X['Time'] - X['Time'].min()).dt.total_seconds()

# Drop the original 'Time' feature
X = X.drop(['Time'], axis=1)

# Now, you can proceed with SMOTE oversampling

# Use SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)




# In[16]:


# Visualize the distribution of 'Class' (target variable) after SMOTE
plt.figure(figsize=(8, 6))
y.value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title('Distribution of Class after SMOTE (0: Non-fraudulent, 1: Fraudulent)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# In[17]:


# Feature Selection
# Select features
selected_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                    'Transaction_hour', 'Normalized_amount']



# In[18]:


# Perform PCA for dimensionality reduction
n_components = min(X.shape[0], X.shape[1])  # Number of components should be less than or equal to the minimum of samples or features
pca = PCA(n_components=n_components)  
X_pca = pca.fit_transform(X)

# Perform feature selection on the PCA-transformed data
k_best_selector = SelectKBest(score_func=f_classif, k=5)  # Adjust k as needed
X_k_best = k_best_selector.fit_transform(X_pca, y)

# Get the indices of selected features
selected_indices = k_best_selector.get_support(indices=True)

# Map selected PCA components back to original feature names
selected_features = [selected_features[i] for i in selected_indices]

print("Selected features using ANOVA F-test after PCA:")
print(selected_features)


# In[19]:


X=X[selected_features]
X.head()


# In[20]:


# Split the SMOTE-resampled data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[21]:


# Create a logistic regression model
logistic_model = LogisticRegression(random_state=42)

# Train the model on the training data
logistic_model.fit(X_train, y_train)

# Predict on the test data
y_pred_logistic = logistic_model.predict(X_test)



# In[22]:


# Evaluate the model
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)
class_report_logistic = classification_report(y_test, y_pred_logistic)

print("Logistic Regression Model Evaluation:")
print("Accuracy:", accuracy_logistic)
print("Classification Report:\n", class_report_logistic)


# In[23]:


LABELS = ['Legit', 'Fraud']
plt.figure(figsize=(4, 4))
sns.set(font_scale=1.1)
sns.heatmap(conf_matrix_logistic, cmap='Spectral', xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d')
plt.title('Confusion Matrix for Logistic Regression')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()


# We have got the accuracy of 0.92 which is good. Still we will observe the accuarcy for Random Forest Classifeer to detemine the accuarcy and see the model is not overfitting.

# In[24]:


# Model Selection and Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model on the training data
model.fit(X_train, y_train)


# In[25]:


# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validation Scores:", cv_scores)
print("Mean Cross-validation Score:", np.mean(cv_scores))



# In[26]:


# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Random Forest Classifier Model Evaluation:")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)


# In[27]:


LABELS = ['Legit', 'Fraud']
plt.figure(figsize=(4, 4))
sns.set(font_scale=1.1)
sns.heatmap(conf_matrix, cmap='Spectral', xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d')
plt.title('Confusion Matrix for Random Forest Classifier Model')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()


# As we can see the accuracy is more i.e. 0.99 we should check the ROC_AUC score to see the data is really clean giving us such a high accuracy or it is overfitting the model.

# In[28]:


# Assuming model is your trained classifier
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability estimates of the positive class

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC Score:", roc_auc)


# A ROC AUC score of 0.9999 indicates that our Random Forest classifier performs exceptionally well at distinguishing between the positive and negative classes in your dataset.

# In[29]:


# Calculate the false positive rate (FPR) and true positive rate (TPR)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Calculate the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[30]:


LABELS = ['Legit', 'Fraud']
plt.figure(figsize=(4, 4))
sns.set(font_scale=1.1)
sns.heatmap(conf_matrix, cmap='Spectral', xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d')
plt.title('Confusion Matrix for Random Forest Classifier')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()


# In[31]:


# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [1, 5, 10],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print("Best Hyperparameters:", best_params)



# In[32]:


# Model Evaluation
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)



# Based on the evaluation metrics for the Logistic Regression and Random Forest Classifier models, here are some conclusions for the credit card fraud detection project:
# 
# 1. **Logistic Regression Model**:
#    - **Accuracy**: The logistic regression model achieved an accuracy of approximately 92.17%. This means that it correctly classified about 92.17% of the transactions in the dataset.
#    - **Precision, Recall, F1-score**: The precision, recall, and F1-score for both classes (0: Non-Fraudulent, 1: Fraudulent) are quite balanced, with values around 0.92. This indicates that the model performs well in terms of both identifying non-fraudulent and fraudulent transactions.
#    - **Support**: The support values indicate the number of samples for each class in the dataset. Both classes have a similar number of samples, which is good for model evaluation.
#    - **Macro Avg and Weighted Avg**: The macro average and weighted average of precision, recall, and F1-score are also around 0.92, indicating a good overall performance of the model across classes.
# 
# 2. **Random Forest Classifier**:
#    - **Accuracy**: The random forest classifier achieved a significantly higher accuracy of approximately 99.76%. This indicates that it performed extremely well in classifying both non-fraudulent and fraudulent transactions.
#    - **Confusion Matrix**: The confusion matrix shows that the model made very few misclassifications, with only a small number of false positives (223) and false negatives (54). This suggests that the model has high precision and recall for both classes.
#    - **Support**: The support values are similar to those in the logistic regression model, indicating a balanced dataset.
#    
# **Overall Conclusion**:
# - Both models, logistic regression, and random forest classifier, show strong performance in detecting credit card fraud.
# - The random forest classifier outperforms the logistic regression model in terms of accuracy, with fewer misclassifications.
# - However, the logistic regression model also demonstrates good performance and may be preferred if interpretability and computational efficiency are important considerations.
# - Further analysis, such as feature importance assessment and model interpretation, could provide additional insights into the factors influencing credit card fraud detection.

# In[33]:


# Save the best model
joblib.dump(best_model, 'credit_card_fraud_detection_model.pkl')


# The line `joblib.dump(best_model, 'credit_card_fraud_detection_model.pkl')` saves the trained machine learning model `best_model` to a file named `'credit_card_fraud_detection_model.pkl'` using the `joblib` library. This file, in a pickle format, stores the model's state, including its parameters and learned patterns. Saving models in this way allows for easy retrieval and reuse for making predictions on new data without the need for retraining.
# 

# # Additional Visualizations

# In[34]:


# Additional Visualizations
# Visualize the distribution of 'Class' (target variable)
plt.figure(figsize=(8, 6))
data['Class'].value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title('Distribution of Class (0: Non-fraudulent, 1: Fraudulent)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig('class_distribution.jpg')  # Save the visualization as a JPEG image
plt.close()

# Creating a heatmap for correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.jpg')  # Save the visualization as a JPEG image
plt.close()

# Scatter plot to visualize the actual vs. predicted classes for test data
plt.figure(figsize=(15, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', marker='o', label='Actual')
plt.scatter(range(len(y_test)), y_pred, color='red', marker='x', label='Predicted')
plt.xlabel('Transaction Index')
plt.ylabel('Class (0: Non-fraudulent, 1: Fraudulent)')
plt.title('Actual vs. Predicted Classes for Test Data')
plt.legend()
plt.savefig('actual_vs_predicted.jpg')  # Save the visualization as a JPEG image
plt.close()

# Plot the transaction volume over time
plt.figure(figsize=(12, 6))
plt.plot(data['Time'], data['Amount'], color='blue', alpha=0.5)
plt.title('Transaction Volume Over Time')
plt.xlabel('Time')
plt.ylabel('Transaction Amount')
plt.grid(True)
plt.savefig('transaction_volume_over_time.jpg')  # Save the visualization as a JPEG image
plt.close()

# Provide links to download each image
print("Download Class Distribution visualization: class_distribution.jpg")
print("Download Correlation Matrix visualization: correlation_matrix.jpg")
print("Download Actual vs. Predicted visualization: actual_vs_predicted.jpg")
print("Download Transaction Volume Over Time visualization: transaction_volume_over_time.jpg")



## Results
The models developed in this project achieved high accuracy in detecting fraudulent transactions. The results are visualized using various plots that illustrate model performance metrics.

## Conclusion
This project demonstrates the application of machine learning techniques to solve real-world problems related to financial fraud detection. The insights gained from this analysis can help improve security measures for financial institutions.

## Future Work
Future enhancements could include:
- Implementing more advanced algorithms such as ensemble methods.
- Exploring additional feature engineering techniques.
- Conducting a more extensive hyperparameter tuning process.

## Contributing
Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

## Guidelines:
Ensure your code adheres to existing style guidelines.
Write clear commit messages.
Document any new features or changes made.
  
## Contact Information
For any inquiries or feedback regarding this project, please contact:

VIJAYAKUMAR R: vijay740125@gmail.com


# import
!pip piplite
!pip install seaborn
!pip install sweetviz
!pip install shap
!pip install catboost

from datetime import datetime
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pytz
import json
import seaborn as sns
import sweetviz as sv
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/credits.csv')

# EDA 
my_report = sv.analyze(df)
my_report.show_notebook()

pd.DataFrame(df) # print the dataframe

df.shape
df.info()
df.isnull().sum()

# Data Cleaning & Feature Engineer
# Removing Features: 'ph_country_code', 'fb_gender', 'fb_dob', 'ph_other_device_info'
df = df.drop('ph_country_code', axis=1)
df = df.drop('fb_dob', axis=1)
df = df.drop('fb_gender', axis=1)
df = df.drop('ph_other_device_info', axis=1)

# Standardizing data types: 'de_date_joined', 'fb_last_update_date'
df['de_date_joined'] = pd.to_datetime(df['de_date_joined'], format='%d/%m/%Y %H:%M')
df['days_since_joined'] = (datetime.now() - df['de_date_joined']).dt.days

utc = pytz.utc
df['fb_last_updated_date'] = pd.to_datetime(df['fb_last_updated_date'])
df['last_updated_utc'] = df['fb_last_updated_date'].dt.tz_convert(utc)
df['days_since_fb_update'] = (datetime.now(utc) - df['last_updated_utc']).dt.days
df = df.drop('fb_last_updated_date', axis=1)
df = df.drop('last_updated_utc', axis=1)
df = df.drop('days_since_fb_update', axis=1)

# 'ph_app_list': create a new column for the result of the length of the app list and drop 'ph_app_list'
def calculate_length(x):
    return len(eval(x))
df['app_list_length'] = df['ph_app_list'].apply(calculate_length)
df = df.drop('ph_app_list', axis=1)
print(df['app_list_length'])

# 'ph_call_log_stats': create new columns for each key-value pair in the dictionary to one-hot encoding
def extract_json_values(row, key):
    try:
        values = json.loads(row['ph_call_log_stats'])
        return values[key]
    except:
        return 0
keys = ['percent_incoming_nighttime', 'percent_outgoing_daytime', 'duration_incoming_daytime',
        'duration_outgoing_daytime', 'percent_incoming_daytime', 'percent_other_calls',
        'duration_outgoing_nighttime', 'percent_outgoing_nighttime', 'total_calls', 'duration_incoming_nighttime']
for key in keys:
    df[key] = df.apply(lambda row: extract_json_values(row, key), axis=1)

# Handle Missing Value: 'fb_relation', 'ph_call_log_stats', 'days_since_fb_update', 'last_updated_utc'
df['fb_relation']=df['fb_relation'].fillna("No_Relation")
df['ph_call_log_stats']=df['ph_call_log_stats'].fillna("No_Stats")

# 'fb_relation': recotegorise the variable types, create new columns for each key-value pair in the dictionary to one-hot encoding
df['fb_relation'] = df['fb_relation'].replace([
       "It's complicated", 'Widowed', 'Divorced',
       'In a civil union', "It's complicated (Pending)"], 'others')

df['fb_relation'] = df['fb_relation'].replace([
       'In an open relationship', 'Engaged',
       'In a domestic partnership', 'Separated',
       'Engaged (Pending)', 'Married (Pending)', 
       'In an open relationship (Pending)', 'In a relationship (Pending)',
       'In a domestic partnership (Pending)'], 'In a relationship')
counts = df["fb_relation"].value_counts()
plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
plt.title("fb_relation")
plt.show()

df['fb_relation'].unique()
def extract_json_values(row, key):
    try:
        values = json.loads(row['fb_relation'])
        return values[key]
    except:
        return 0

keys = ['Married', 'No_Relation', 'In a relationship', 'Single', 'others']
for key in keys:
    df[key] = df.apply(lambda row: extract_json_values(row, key), axis=1)

# Remove the outliers:
def remove_outliers(df, col_name, threshold):
    z_scores = stats.zscore(df[col_name])
    abs_z_scores = np.abs(z_scores)
    outliers = abs_z_scores > threshold
    df[col_name][outliers] = np.nan
    df.drop(df.index[outliers], inplace=True)
cols_to_process = ['de_monthly_salary', 'de_employment_duration', 'ph_total_contacts', 'days_since_joined', 'app_list_length', 'total_calls',
                   'duration_incoming_daytime', 'duration_incoming_nighttime', 'duration_outgoing_daytime', 'duration_outgoing_nighttime']
threshold = 3
for col in cols_to_process:
    remove_outliers(df, col, threshold)
    
# Model Training
## Logistic Regression
def logistic_regression(df, test_size=0.2):
    x = df.drop('flag_bad', axis=1) # features
    y = df['flag_bad'] # target variable
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    scaler = StandardScaler() # standardize
    X_train_scaled = scaler.fit_transform(X_train) # standardize
    X_test_scaled = scaler.transform(X_test) # standardize
    logistic_reg = LogisticRegression(random_state=42)
    logistic_reg.fit(X_train_scaled, y_train)
    y_pred = logistic_reg.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Confusion Matrix: ")
    print(confusion)
    print("ROC AUC Score: ", auc)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label="ROC Curve (area=%0.2f)" % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

logistic_regression(df, test_size=0.2)

# Random Forest Classifier
def random_forest_classification(df, test_size=0.2, random_state=42):

    x = df.drop('flag_bad', axis=1) # features
    y = df['flag_bad'] # target variable

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  

    rf_classifier = RandomForestClassifier(random_state=random_state)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:,1])

    print("Random Forest Classifier Results:")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print('ROC AUC:', roc_auc)

    fpr, tpr, thresholds = roc_curve(y_test, rf_classifier.predict_proba(X_test)[:,1])
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr)
    plt.plot(fpr, tpr, label="ROC AUC = {:.3f}".format(roc_auc))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.show()

    return rf_classifier, scaler

rf_classifier, scaler = random_forest_classification(df, test_size=0.2, random_state=42)

# Support Vector Machines (SVM)
def run_svm_classification(df, test_size=0.2, random_state=42):
    x = df.drop('flag_bad', axis=1) # features
    y = df['flag_bad'] # target variable
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=test_size, random_state=random_state)

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  

  
    svm_model = SVC(random_state=random_state)
    svm_model.fit(X_train, y_train)
    
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Support Vector Machines Results:")
    print("SVM Accuracy: ", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Precision: ", precision_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    print("F1 Score: ", f1_score(y_test, y_pred))

   
    y_prob = svm_model.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label="ROC AUC = {:.3f}".format(roc_auc))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()

run_svm_classification(df, test_size=0.2)

# XGBoost
def xgboost_model(df, test_size=0.2):
    
    x = df.drop('flag_bad', axis=1) # features
    y = df['flag_bad'] # target variable
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  
    
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train_scaled, y_train)

    y_pred = xgb_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Confusion Matrix: ")
    print(confusion)
    print("ROC AUC Score: ", auc)

     # Calculate and plot SHAP values
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_test_scaled)
    shap.summary_plot(shap_values, X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label="ROC Curve (area=%0.2f)" % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

xgboost_model(df, test_size=0.2) # can control train-test size here

# Gradient Boosting
def gradient_boosting(df, test_size=0.2):
    x = df.drop('flag_bad', axis=1) # features
    y = df['flag_bad'] # target variable
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  

    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train_scaled, y_train)

    y_pred = gb.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Confusion Matrix: ")
    print(confusion)
    print("ROC AUC Score: ", auc)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label="ROC Curve (area=%0.2f)" % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

gradient_boosting(df, test_size=0.2)

# CatBoost
from catboost import CatBoostClassifier
def catboost_classifier(df, test_size=0.2):
    x = df.drop('flag_bad', axis=1) # features
    y = df['flag_bad'] # target variable
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    catboost_clf = CatBoostClassifier(random_seed=42, iterations=500, learning_rate=0.1, loss_function='Logloss')
    catboost_clf.fit(X_train, y_train, verbose=10)
    y_pred = catboost_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Confusion Matrix: ")
    print(confusion)
    print("ROC AUC Score: ", auc)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label="ROC Curve (area=%0.2f)" % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    feature_importances = catboost_clf.get_feature_importance()
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    importance_df = importance_df.sort_values('importance', ascending=False)
    plt.bar(x=importance_df['feature'], height=importance_df['importance'])
    plt.xticks(rotation=90)
    plt.title('Feature Importances')
    plt.show()

catboost_classifier(df, test_size=0.2)

# Model Improvenment
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import pandas as pd
X = df.drop('flag_bad', axis=1)
y = df['flag_bad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
selector = SelectKBest(f_classif, k=5)
X_train_new = selector.fit_transform(X_train, y_train)
X_test_new = selector.transform(X_test)
print(selector.get_support(indices=True))
selected_features = X_train.columns[selector.get_support()]
print(selected_features)

# The feature importance values for the selected features in XGBoost
import xgboost as xgb
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train_new, y_train)
feature_importance = xgb_model.feature_importances_
selected_features = X_train.columns[selector.get_support()]
for feature_name, importance_value in zip(selected_features, feature_importance):
    print("{}: {}".format(feature_name, importance_value))
    
# Use Top 5 Feature Selection to train the model in XGBoost
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

def xgboost_model(df, test_size=0.2):
    x = df[['de_employment_type', 'de_education', 'app_list_length', 'percent_incoming_daytime', 'percent_other_calls']] # top five selected features
    y = df['flag_bad'] # target variable
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    y_pred = xgb_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Confusion Matrix: ")
    print(confusion)
    print("ROC AUC Score: ", auc)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label="ROC Curve (area=%0.2f)" % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
xgboost_model(df, test_size=0.2)

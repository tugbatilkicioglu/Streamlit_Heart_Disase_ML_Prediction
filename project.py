import os.path
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import warnings


# Uyarıları kapat
warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
warnings.filterwarnings("ignore")


# Veri setini yükleme (örnek olarak CSV dosyası kullanılıyor)
df = pd.read_csv("heart.csv")
# Özellik mühendisliği
df['age_max_heart_rate_ratio'] = df['age'] / df['thalach']
df['age_range'] = pd.cut(df['age'], bins=[29, 39, 49, 59, 69, 79], labels=['30-39', '40-49', '50-59', '60-69', '70-79'])
df['cholesterol_hdl_ratio'] = df['chol'] / df['thalach']
df['heart_rate_reserve'] = df['thalach'] - df['trestbps']


df.dropna(inplace=True)
duplicates = df.duplicated()
df = df[~duplicates]

main_dir = os.path.join(os.getcwd())
read_file = 'heartfeature.csv'
path = os.path.join(main_dir,read_file)


df.to_csv(path, index=False)

df = pd.read_csv(path)


# Özellik ve hedef değişkeni tanımlama
X = df.drop(['target',"age_range"], axis=1)
y = df['target']

# Veriyi eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Modelleri ve hiperparametre aralıklarını tanımla
models = [
    ('Logistic Regression', LogisticRegression(), {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}),
    ('K-Nearest Neighbors', KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}),
    ('Decision Tree', DecisionTreeClassifier(), {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}),
    ('Random Forest', RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30],
                                                 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    ('Gradient Boosting', GradientBoostingClassifier(),
     {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2],
      'max_depth': [3, 5, 7]}),
    ('XGBoost', XGBClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2],
                                  'max_depth': [3, 5, 7], 'subsample': [0.8, 0.9, 1.0]})
]

# En iyi modeli ve en iyi skoru saklamak için değişkenleri tanımla
best_model = None
best_score = 0.0

# Her model için
for name, model, param_grid in models:
    # Grid Search ile en iyi hiperparametreleri bul
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='recall')
    grid_search.fit(X_train, y_train)

    # En iyi modeli seç
    best_model_candidate = grid_search.best_estimator_

    # Modeli eğit
    best_model_candidate.fit(X_train, y_train)

    # Test seti üzerinde modelin performansını değerlendir
    y_pred = best_model_candidate.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, best_model_candidate.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Tüm metrikleri göster
    print(f"{name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}\n")

    # En iyi modeli güncelle
    if recall > best_score:
        best_score = recall
        best_model = best_model_candidate

    # ROC Curve çizimi
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Modeli dosyaya kaydet
    joblib.dump(best_model_candidate, f'{name.lower().replace(" ", "_")}_best_model.joblib')

# En iyi modeli dosyaya kaydet
joblib.dump(best_model, 'best_of_the_best_model.joblib')

# Tüm modelleri eğit ve kaydet
for name, model, _ in models:
    # Modeli eğit
    model.fit(X_train, y_train)

    # Modeli dosyaya kaydet
    joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.joblib')

print("Tüm modeller başarıyla eğitildi ve kaydedildi.")

# Eğitimde kullanılan özellik isimlerini belirle
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'age_max_heart_rate_ratio', 'cholesterol_hdl_ratio', 'heart_rate_reserve']


# Model dosyalarını yükle
logistic_regression_model = joblib.load(os.path.join(main_dir, 'logistic_regression_best_model.joblib'))
knn_model = joblib.load(os.path.join(main_dir, 'k-nearest_neighbors_best_model.joblib'))
decision_tree_model = joblib.load(os.path.join(main_dir, 'decision_tree_best_model.joblib'))
random_forest_model = joblib.load(os.path.join(main_dir, 'random_forest_best_model.joblib'))
gradient_boosting_model = joblib.load(os.path.join(main_dir, 'gradient_boosting_best_model.joblib'))
xgboost_model = joblib.load(os.path.join(main_dir, 'xgboost_best_model.joblib'))




# Tahminler yap
logistic_regression_predictions = logistic_regression_model.predict(X_test)
knn_predictions = knn_model.predict(X_test)
decision_tree_predictions = decision_tree_model.predict(X_test)
random_forest_predictions = random_forest_model.predict(X_test)
gradient_boosting_predictions = gradient_boosting_model.predict(X_test)
xgboost_predictions = xgboost_model.predict(X_test)

# Tahmin sonuçlarını göster
print("Logistic Regression Predictions:", logistic_regression_predictions)
print("K-Nearest Neighbors Predictions:", knn_predictions)
print("Decision Tree Predictions:", decision_tree_predictions)
print("Random Forest Predictions:", random_forest_predictions)
print("Gradient Boosting Predictions:", gradient_boosting_predictions)
print("XGBoost Predictions:", xgboost_predictions)


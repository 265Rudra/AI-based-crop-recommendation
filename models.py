import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # Interactive backend for plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('crop_data.csv')

print("Dataset shape:", df.shape)
print("\\nDataset info:")
print(df.info())
print("\\nFirst 5 rows:")
print(df.head())

# Data visualizations
fig = plt.figure(figsize=(20, 15))

# 1. Crop distribution
plt.subplot(3, 3, 1)
df['label'].value_counts().plot(kind='bar')
plt.title('Crop Distribution')
plt.xticks(rotation=45)

# 2. N-P-K scatter
plt.subplot(3, 3, 2)
plt.scatter(df['N'], df['P'], c=df['label'].astype('category').cat.codes, cmap='tab20')
plt.xlabel('Nitrogen')
plt.ylabel('Phosphorus')
plt.title('N vs P colored by Crop')

# 3. Temperature vs Humidity
plt.subplot(3, 3, 3)
plt.scatter(df['temperature'], df['humidity'], c=df['label'].astype('category').cat.codes, cmap='tab20')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Temp vs Humidity by Crop')

# 4. Boxplot N by crop
plt.subplot(3, 3, 4)
df.boxplot(column='N', by='label', ax=plt.gca(), fontsize=8)
plt.title('Nitrogen by Crop')
plt.xticks(rotation=45)

# 5. Correlation heatmap
plt.subplot(3, 3, 5)
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')

# 6. Rainfall distribution
plt.subplot(3, 3, 6)
df['rainfall'].hist(bins=30, alpha=0.7)
plt.title('Rainfall Distribution')

# 7. pH distribution
plt.subplot(3, 3, 7)
df['ph'].hist(bins=30, alpha=0.7)
plt.title('pH Distribution')

# 8. Pairplot sample
plt.subplot(3, 3, 8)
sns.pairplot(df[['N', 'P', 'K', 'label']], hue='label')
plt.suptitle('N,P,K Pairplot', y=1.02)

plt.tight_layout()


print("\\nData visualizations saved and displayed!")

# Prepare features and label
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_str = df['label']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_str)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f'Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}')

# Train SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print(f'SVM Accuracy: {accuracy_score(y_test, svm_pred):.4f}')

# Train XGBoost
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print(f'XGBoost Accuracy: {accuracy_score(y_test, xgb_pred):.4f}')

# Fixed predict_crop function
def predict_crop(features):
    rf_p = rf.predict([features])[0]
    svm_p = svm.predict([features])[0]
    xgb_p = xgb_model.predict([features])[0]
    # Majority vote
    predictions = [rf_p, svm_p, xgb_p]
    final_p = np.bincount(predictions).argmax()
    
    crop_name = le.inverse_transform([final_p])[0]
    rf_name = le.inverse_transform([rf_p])[0]
    svm_name = le.inverse_transform([svm_p])[0]
    xgb_name = le.inverse_transform([xgb_p])[0]
    
    return {
        'predicted_crop': crop_name,
        'rf': rf_name,
        'svm': svm_name,
        'xgb': xgb_name
    }

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrices and visualizations for each model
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Random Forest
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Random Forest Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# SVM
cm_svm = confusion_matrix(y_test, svm_pred)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('SVM Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

# XGBoost
cm_xgb = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=axes[2])
axes[2].set_title('XGBoost Confusion Matrix')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

print('Classification Reports:')
print('Random Forest:\\n', classification_report(y_test, rf_pred, target_names=le.classes_))
print('\\nSVM:\\n', classification_report(y_test, svm_pred, target_names=le.classes_))
print('\\nXGBoost:\\n', classification_report(y_test, xgb_pred, target_names=le.classes_))

# Save models and label encoder
joblib.dump({'rf': rf, 'svm': svm, 'xgb': xgb_model, 'le': le, 'predict_crop': predict_crop}, 'models.joblib')
# Popup with results
root = tk.Tk()
root.withdraw()
rf_acc = accuracy_score(y_test, rf_pred) * 100
svm_acc = accuracy_score(y_test, svm_pred) * 100
xgb_acc = accuracy_score(y_test, xgb_pred) * 100
message = f"Models trained successfully!\\n\\nRandom Forest: {rf_acc:.2f}%\\nSVM: {svm_acc:.2f}%\\nXGBoost: {xgb_acc:.2f}%"
messagebox.showinfo("Training Complete", message)
print('Models and predict function saved to models.joblib')


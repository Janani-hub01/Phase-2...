import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Create synthetic dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Speed_limit': np.random.choice([30, 40, 50, 60, 70], size=n_samples),
    'Weather_conditions': np.random.choice(['Clear', 'Rainy', 'Snowy', 'Foggy'], size=n_samples),
    'Light_conditions': np.random.choice(['Daylight', 'Darkness - lights lit', 'Darkness - no lighting'], size=n_samples),
    'Road_type': np.random.choice(['Single carriageway', 'Dual carriageway', 'Roundabout'], size=n_samples),
    'Accident_severity': np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.25, 0.05])  # 0=Slight, 1=Serious, 2=Fatal
})

# Step 3: Encode categorical variables
from sklearn.preprocessing import LabelEncoder
label_cols = ['Weather_conditions', 'Light_conditions', 'Road_type']
le_dict = {}

for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le  # Save encoders if needed for prediction later

# Step 4: Prepare features and target
X = data.drop('Accident_severity', axis=1)
y = data['Accident_severity']

# Step 5: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 7: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Predict
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
sns.countplot(x='Accident_severity', data=data)
plt.title("Distribution of Accident Severity in Synthetic Data")
plt.xlabel("Severity (0=Slight, 1=Serious, 2=Fatal)")
plt.show()

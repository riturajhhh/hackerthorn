import os
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from utils.feature_engineering import engineer
from utils.reassign import reassign
from utils.genai import generate_insight
from utils.alerts import send_slack

# Load data
df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

df_train['breach'] = df_train['breach'].astype(int)
df_test['breach'] = df_test['breach'].astype(int)

# Feature engineering
df_train = engineer(df_train)
df_test = engineer(df_test)

features = [
    'priority_score','estimated_time','time_remaining',
    'completion_percentage','time_ratio','urgency_score',
    'risk_score','tight_deadline','very_tight','assigned_to'
]

X_train = df_train[features].copy()
y_train = df_train['breach']

X_test = df_test[features].copy()

# Encode
le = LabelEncoder()
X_train['assigned_to'] = le.fit_transform(X_train['assigned_to'].astype(str))
X_test['assigned_to'] = X_test['assigned_to'].astype(str).apply(lambda x: 0)

# Scale
imp = SimpleImputer()
scaler = StandardScaler()

X_train = scaler.fit_transform(imp.fit_transform(X_train))
X_test = scaler.transform(imp.transform(X_test))

# Train
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

df_test['predicted_breach'] = y_pred
df_test['probability'] = y_prob

# SLA status
df_test['SLA_Status'] = df_test['predicted_breach'].map({
    0: 'SAFE', 1: 'UNSAFE'
})

# Reassign
df_test['reassigned_to'] = df_test.apply(reassign, axis=1)

# GenAI (set SKIP_GENAI=1 for a fast run without API calls)
if os.environ.get("SKIP_GENAI"):
    df_test["AI_Insight"] = ""
else:
    df_test["AI_Insight"] = df_test.apply(
        lambda r: generate_insight(r) if r["predicted_breach"] == 1 else "",
        axis=1,
    )

# Alerts (set SKIP_SLACK=1 to skip webhook calls during local runs)
if not os.environ.get("SKIP_SLACK"):
    for _, row in df_test.iterrows():
        if row["predicted_breach"] == 1 and row["probability"] > 0.7:
            send_slack(
                f"⚠️ SLA Risk: {row['assigned_to']} → {row['reassigned_to']}"
            )

# Save
df_test.to_csv("outputs/results.csv", index=False)
pickle.dump(model, open("models/model.pkl","wb"))

print("Pipeline complete!")
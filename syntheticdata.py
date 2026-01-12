import pandas as pd
import numpy as np
import random
import uuid
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Generate synthetic data
# -----------------------------
n_samples = 150000
countries = ["AM","RU","US","DE","FR","IN","CN"]
statuses = ["active","suspended","closed"]
verif_levels = ["none","basic","full"]
currencies = ["USD","EUR","AMD","RUB"]
tr_types = ["purchase","withdrawal","deposit","refund"]
status_3d_choices = ["passed","failed","not_enrolled"]

data = []
for i in range(n_samples):
    user_id = str(uuid.uuid4())
    user_country = random.choice(countries)
    user_city = f"City_{random.randint(1,500)}"
    user_account_status = random.choice(statuses)
    user_balance = round(random.uniform(0,10000),2)
    user_verification_level = random.choice(verif_levels)
    transaction_id = str(uuid.uuid4())
    transaction_amount = round(random.uniform(1,5000),2)
    transaction_currency = random.choice(currencies)
    transaction_type = random.choice(tr_types)
    card_bin = str(random.randint(400000,499999))
    card_last = str(random.randint(1000,9999))
    status_3d = random.choice(status_3d_choices)
    cvv_result = random.choice([True,False])
    ip = f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}"
    device_id = str(uuid.uuid4())
    shipping_country = random.choice(countries)
    billing_country = random.choice(countries)

    risk_score = 0
    if shipping_country != billing_country: risk_score += 1
    if user_account_status=="suspended": risk_score += 1
    if user_verification_level=="none": risk_score += 1
    if transaction_amount>3000 and not cvv_result: risk_score += 1
    if status_3d=="failed": risk_score += 1

    is_fraud = 1 if (risk_score>=2 and random.random()<0.1) else 0
    data.append([
        user_id,user_country,user_city,user_account_status,user_balance,user_verification_level,
        transaction_id,transaction_amount,transaction_currency,transaction_type,
        card_bin,card_last,status_3d,cvv_result,ip,device_id,shipping_country,billing_country,
        is_fraud,risk_score
    ])

columns = ["user_id","user_country","user_city","user_account_status","user_balance","user_verification_level",
           "transaction_id","transaction_amount","transaction_currency","transaction_type",
           "card_bin","card_last","status_3d","cvv_result","ip","device_id","shipping_country","billing_country",
           "is_fraud","risk_score"]

df = pd.DataFrame(data, columns=columns)

# -----------------------------
# Data preprocessing
# -----------------------------
df_model = df.drop(columns=['user_id','transaction_id','ip','device_id','card_bin','card_last'])

cat_cols = df_model.select_dtypes(include=['object']).columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

X = df_model.drop('is_fraud', axis=1)
y = df_model['is_fraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# SMOTE to balance classes
# -----------------------------
sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(X_scaled, y)

# -----------------------------
# Train CatBoost model
# -----------------------------
cat_model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, random_state=42)
cat_model.fit(X_bal, y_bal)

# -----------------------------
# Dynamic threshold calculation by F1
# -----------------------------
y_proba_train = cat_model.predict_proba(X_bal)[:,1]

precisions, recalls, thresholds = precision_recall_curve(y_bal, y_proba_train)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Best F1 threshold: {best_threshold:.3f}, "
      f"Precision={precisions[best_idx]:.3f}, Recall={recalls[best_idx]:.3f}")

threshold = best_threshold

# -----------------------------
# Save model, scaler, encoders, feature columns, threshold
# -----------------------------
cat_model.save_model('catboost_fraud_model.cbm')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
feature_cols = X.columns.tolist()
joblib.dump(feature_cols, 'feature_cols.pkl')
joblib.dump(threshold, 'threshold.pkl')

print("Training complete. Model, scaler, label encoders, feature columns, and threshold saved.")

# -----------------------------
# Plot Precision, Recall, F1 vs Threshold
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.plot(thresholds, f1_scores[:-1], label="F1 Score")
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f"Best threshold={best_threshold:.3f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall and F1 vs Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("threshold_plot.png")  # Save plot to file
plt.close()
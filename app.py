from flask import Flask, request, render_template
from flask_mail import Mail, Message
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import base64
from cryptography.fernet import Fernet
import io
import hashlib

app = Flask(__name__, template_folder="templates")
app.secret_key = "supersecret"


cat_model = CatBoostClassifier()
cat_model.load_model('catboost_fraud_model.cbm')

scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_cols = joblib.load('feature_cols.pkl')
threshold = joblib.load('threshold.pkl')


app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'tamara.khachatryan.2001@gmail.com'
app.config['MAIL_PASSWORD'] = 'yngy clnw ebhu bexa'  
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/orenq')
def indexorenq():
    return render_template('indexorenq.html')

@app.route('/about')
def about():
    return render_template('indexabout.html')

@app.route('/contact')
def kap_mez_het():
    return render_template('kap_mez_het.html')

@app.route('/send-email', methods=['POST'])
def send_email():
    name = request.form['name']
    email = request.form['email']
    phone = request.form['phone']
    message_body = request.form['message']

 
    msg = Message(
        subject=f"Նոր հաղորդագրություն {name}-ից",
        sender=app.config['MAIL_USERNAME'],
        recipients=[app.config['MAIL_USERNAME']],
        body=f"Name: {name}\nEmail: {email}\nPhone: {phone}\nMessage:\n{message_body}"
    )
    mail.send(msg)

    reply = Message(
        subject="Շնորհակալություն հաղորդագրության համար",
        sender=app.config['MAIL_USERNAME'],
        recipients=[email],
        body=f"Բարև Ձեզ {name},\n\nՇնորհակալություն, որ դիմեցիք մեզ։ Մեր մասնագետները շուտով կապ կհաստատեն Ձեզ հետ՝ ներկայացնելու խարդախությունների հայտնաբերման լուծումները Ձեր բիզնեսի ու գործարքների համար։"
    )
    mail.send(reply)

    return render_template("thanks.html")



@app.route('/fraud')
def indexfraud():
    return render_template('indexfraud.html')

@app.route('/predict', methods=['POST'])
def predict_fraud():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    file_bytes = file.read()
    encrypted_file = cipher_suite.encrypt(file_bytes)
    decrypted_bytes = cipher_suite.decrypt(encrypted_file)

    df_input = pd.read_csv(io.BytesIO(decrypted_bytes))

    import hashlib
    for col in ['user_id', 'ip', 'device_id']:
        if col in df_input.columns:
            df_input[col + '_hashed'] = df_input[col].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
    df_input.drop(columns=['user_id','ip','device_id'], inplace=True, errors='ignore')

    drop_cols = ['transaction_id','card_bin','card_last'] 
    df_model_input = df_input.drop(columns=drop_cols, errors='ignore')


    for col in df_model_input.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            le = label_encoders[col]
            df_model_input[col] = df_model_input[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        else:
            df_model_input[col] = 0

   
    for col in feature_cols:
        if col not in df_model_input.columns:
            df_model_input[col] = 0

   
    df_model_input = df_model_input[feature_cols]


    df_scaled = scaler.transform(df_model_input)

    y_proba = cat_model.predict_proba(df_scaled)[:,1]
    y_pred = (y_proba >= threshold).astype(int)

    fraud_count = sum(y_pred)
    fraud_indices = df_input.index[y_pred==1].tolist()
    result_text = f"{fraud_count} Fraud transactions detected!"

    plt.figure(figsize=(8,4))
    plt.plot(y_proba, label='Fraud Probability', marker='o', linestyle='-')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold:.3f}')
    plt.xlabel('Transaction Index')
    plt.ylabel('Fraud Probability')
    plt.title('Fraud Probability per Transaction')
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('indexfraud.html',
                           result=result_text,
                           fraud_rows=fraud_indices,
                           plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)

    
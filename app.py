import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import accuracy_score

# Load and clean data
df = pd.read_csv('loan predictor.csv')
df.columns = df.columns.str.strip()  # remove any leading/trailing spaces

df.rename(columns={'Credit_History': 'Loan_Approved'}, inplace=True)
df.drop('Loan_ID', axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
    df[col] = le.fit_transform(df[col])

# One-hot encode Dependents
df = pd.get_dummies(df, columns=['Dependents'], drop_first=True)

# Remove missing values
df.dropna(inplace=True)

# Feature & target split
X = df.drop('Loan_Approved', axis=1)
y = df['Loan_Approved']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initial model training
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Hyperparameter tuning
param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 7],
    'min_samples_split': [2, 3, 4, 5, 6, 7],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7]
}
grid = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
best_params = grid.best_params_

# Cross-validation score
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_score = cross_val_score(model, X, y, cv=kf)

# -----------------------
# Streamlit App UI
# -----------------------
st.title("🏦 Loan Approval Predictor")
st.write("Enter applicant details to predict if the loan will be approved.")

# Input fields
gender = st.selectbox("Gender", ['Male', 'Female'])
married = st.selectbox("Married", ['Yes', 'No'])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
applicant_income = st.number_input("Applicant Income", value=5000)
coapplicant_income = st.number_input("Coapplicant Income", value=0)
loan_amount = st.number_input("Loan Amount)", value=100)
loan_term = st.number_input("Loan Amount Term (in days)", value=360)
property_area = st.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'])
dependents = st.selectbox("Number of Dependents", ['0', '1', '2', '3+'])

# Encode input
input_dict = {
    'Gender': 1 if gender == 'Male' else 0,
    'Married': 1 if married == 'Yes' else 0,
    'Education': 1 if education == 'Graduate' else 0,
    'Self_Employed': 1 if self_employed == 'Yes' else 0,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Property_Area': {'Urban': 2, 'Rural': 1, 'Semiurban': 0}[property_area],
    'Dependents_1': 1 if dependents == '1' else 0,
    'Dependents_2': 1 if dependents == '2' else 0,
    'Dependents_3+': 1 if dependents == '3+' else 0
}

input_df = pd.DataFrame([input_dict])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100
    if prediction == 1:
        st.success(f"✅ Loan Approved with {probability:.2f}% confidence")
    else:
        st.error(f"❌ Loan Not Approved (Confidence: {100 - probability:.2f}%)")

# Evaluation metrics
st.markdown("---")
st.subheader("📊 Model Evaluation")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")


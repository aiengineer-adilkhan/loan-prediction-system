```markdown
🏦 LoanIQ – Credit Risk Prediction System

LoanIQ is a Machine Learning application that predicts whether a loan applicant represents good credit risk or bad credit risk based on financial and demographic attributes.

The system uses multiple supervised learning models trained on the German Credit Dataset and provides predictions through an interactive Streamlit web application.

This project demonstrates an end-to-end ML workflow, including data preprocessing, model training, model serialization, and deployment through a user-friendly interface.

📊 Dataset

The model is trained on the German Credit Dataset, which contains 1000 loan applicant records used to evaluate borrower credit risk.

Features Used

Age → Age of the applicant

Sex → Gender of the applicant

Job → Employment skill level (0–3)

Housing → Housing status (own / rent / free)

Saving accounts → Category of savings balance

Checking account → Category of checking account balance

Credit amount → Loan amount requested by the applicant

Duration → Loan repayment period (months)

Purpose → Reason for taking the loan

Target Variable

Risk = good → Low credit risk (loan likely to be approved)

Risk = bad → High credit risk (loan likely to be rejected)
For modeling purposes:

0 → Good Credit Risk
1 → Bad Credit Risk
🧠 Machine Learning Models

The following algorithms were trained and evaluated:

Random Forest

Logistic Regression

Decision Tree

The models are saved using Joblib and loaded dynamically in the Streamlit application.

⚙️ Machine Learning Pipeline

The project follows a standard ML workflow:

Data preprocessing

Handling categorical features using encoders

Feature selection

Model training and evaluation

Saving trained models

Integrating models with a Streamlit application

Saved artifacts:

models/
│
├── loan_rf.pkl
├── loan_lr.pkl
├── loan_dt.pkl
├── encoders.pkl
└── feature_columns.pkl
🖥️ Streamlit Application

The project includes a professional Streamlit interface that allows users to:

Enter applicant information

Select a machine learning model

Predict loan approval probability

View prediction confidence and visualizations

The application loads trained models and performs real-time predictions.

Run the app:

streamlit run app.py
📂 Project Structure
loan_predictor/
│
├── data/
│   └── loan_data.csv
│
├── models/
│   ├── loan_rf.pkl
│   ├── loan_lr.pkl
│   ├── loan_dt.pkl
│   ├── encoders.pkl
│   └── feature_columns.pkl
│
├── loan_prediction.ipynb
├── app.py
└── README.md
🚀 How to Run the Project

Clone the repository:

git clone https://github.com/yourusername/loan-prediction-system.git

Install dependencies:

pip install -r requirements.txt

Train the models:

Run loan_prediction.ipynb

Start the application:

streamlit run app.py
🧑‍💻 Author

Adil Khan

Machine Learning & Data Science Enthusiast

This project demonstrates practical experience with:

Machine Learning

Data preprocessing

Model evaluation

Streamlit application development

End-to-end ML project workflow




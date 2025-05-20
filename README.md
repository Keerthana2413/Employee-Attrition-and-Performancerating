# Employee-Attrition-and-Performancerating
Employee Attrition & Performance Rating Predictor
This is a Streamlit-based web application that allows users to predict:

Whether an employee is likely to leave the company (Attrition)

The employee’s Performance Rating

Both predictions are powered by trained machine learning models with appropriate preprocessing like feature scaling and handling class imbalance.

🔧 Features
Interactive user interface via Streamlit

Sidebar navigation to switch between:

Attrition Prediction

Performance Rating Prediction

Input validation and scaling

Uses:

Random Forest Classifier for attrition

Random Forest Classifier (or SMOTE-enhanced) for performance prediction

Custom models and scalers loaded via pickle

🚀 Installation
Clone this repository or download the files.

Ensure you have Python 3.7+

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
requirements.txt (create this file if needed):

txt
Copy
Edit
streamlit
pandas
numpy
scikit-learn
imblearn
📁 Project Structure
bash
Copy
Edit
.
├── app.py                      # Main Streamlit application
├── README.md                   # This file
├── scaler1.pkl                 # Scaler for attrition features
├── Randomforestfinal.pkl       # Trained model for attrition
├── sc.pkl                      # Scaler for performance rating
├── RM.pkl                      # Trained model for performance rating
└── Employee-Attrition.csv      # Reference dataset (not used for prediction)
▶️ How to Run
bash
Copy
Edit
streamlit run app.py
This will open the app in your web browser. Use the sidebar to switch between "Attrition" and "Performance Rating" prediction modes.

📌 Notes
Ensure model and scaler .pkl files are available in the same directory as the app or update the paths.

The app does not write any data or store input—everything runs locally.


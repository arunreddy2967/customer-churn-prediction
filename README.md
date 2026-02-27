This project predicts whether a customer will churn (leave) or stay using Machine Learning.

⸻

Description

The system uses a trained Random Forest model and a Flask web application to take user input and predict customer churn.

⸻

Files
	•	customer_churn_model.pkl → Trained model
	•	encoders.pkl → Data encoders
	•	Telco-Customer-Churn.csv → Dataset
	•	Customer_Churn_Prediction.py → Model training
	•	Predictor_App.py → Flask web app
	•	README.md → Documentation

 Technologies
	•	Python
	•	Flask
	•	Scikit-learn
	•	Pandas

How to Run:
	
    1.	Install libraries:
    pip install pandas numpy scikit-learn flask
    
    2.	Run the app:
    python Predictor_App.py
   
    3.	Open browser
    http://127.0.0.1:5000/


How it Works:
	•	Model is trained using customer data
	•	User enters input values
	•	Model predicts churn or stay
	•	Result is shown on screen


 Output:

	•	❌ Customer will Churn
    
	•	✅ Customer will Stay



 Author:

 Sudireddy Arun Kumar Reddy

# customer-churn-prediction

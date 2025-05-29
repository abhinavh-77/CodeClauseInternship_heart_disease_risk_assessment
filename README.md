❤️ Heart Disease Risk Assessment

🎯 Aim:

To develop a machine learning model that can predict whether a patient is at high risk of heart disease based on clinical attributes. The model helps identify high-risk individuals for early intervention.


📁 Dataset:

File: [heart.csv](https://github.com/user-attachments/files/20499294/heart.csv)


Description: The dataset includes patient health metrics such as age, sex, chest pain type, cholesterol, blood pressure, heart rate, and more.


Target Column: target


1 indicates presence of heart disease (high risk)


0 indicates absence of heart disease (low risk)



🔧 Technologies Used:

Python


Pandas / NumPy


Matplotlib / Seaborn


Scikit-learn


Google Colab


Streamlit (for deployment)



📊 Data Exploration:

Checked for missing values


Split into features (X) and target (y)


Visualized data using matplotlib and seaborn (you can expand this for EDA)



print("Null values in dataset:", df.isnull().sum().sum())


🧠 Model Training:

Model: Random Forest Classifier


Train/Test Split: 80% training, 20% testing


Hyperparameters: n_estimators=100, random_state=42



model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


📈 Evaluation:

Accuracy Score


Classification Report (Precision, Recall, F1-score)



print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


🔮 Risk Prediction

Example input (manual):

input_data = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])
prediction = model.predict(input_data)
print("Prediction:", "High Risk" if prediction[0] == 1 else "Low Risk")


🚀 Deployment:

The model is set up for deployment using Streamlit:

!streamlit run app.py


✅ Improvements & Future Work:

Hyperparameter Tuning: Improve performance with GridSearchCV.


Feature Importance: Plot which features contribute most to prediction.


Cross-validation: Improve robustness with K-Fold validation.


UI Enhancement: Build a user-friendly web app with Streamlit.


Model Comparison: Try Logistic Regression, SVM, or Gradient Boosting models.


Explainability: Integrate SHAP/LIME for model transparency.


📜 License:

This project is open-source and available under the MIT License.

🙋‍♂️ Author:

Haridasula Abhinav

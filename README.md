🏥 Patient Stay Estimator – Regression Model
📘 Project Overview
This project predicts how long a patient is likely to stay in the hospital, measured in days, using historical medical and demographic data.
By estimating the Length of Stay (LOS) before or during admission, hospitals can better manage beds, staff allocation, and resources — improving both efficiency and patient care.
________________________________________
🎯 Objective
Develop a Machine Learning Regression Model that accurately predicts the hospital stay duration (length_of_stay) based on various patient-level and clinical features such as demographics, comorbidities, and lab test results.
________________________________________
📊 Dataset Details
Dataset: LengthOfStay.csv
Type: Healthcare (encounter-level) dataset
Target variable: length_of_stay
Main Feature Categories
Category	Example Columns	Description
Identifiers / Dates	episode_id, visit_date, discharge_date, facility_id	Episode (hospital stay) details
Demographics	gender, readmission_count	Basic patient info
Comorbidities	asthma, depression, pneumonia, malnutrition, substance_dependence, etc.	0/1 flags for chronic or acute conditions
Lab Tests & Vitals	hemoglobin, hematocrit, sodium, glucose, blood_urea_nitrogen, creatinine, bmi, pulse, respiration	Clinical measurements that may affect recovery time
________________________________________
🧹 Data Preprocessing
All preprocessing steps were performed programmatically in the notebook.
1.	Column Renaming
Shortened and standardized column names for clarity (e.g., eid → episode_id, facid → facility_id, lengthofstay → length_of_stay, etc.).
2.	Datetime Conversion
Converted visit_date and discharge_date into datetime format.
Extracted year, month, day, and weekday from both, then dropped the original columns.
3.	Data Cleaning
o	Replaced '5+' in readmission_count with numeric 5.
o	Converted readmission_count to integer type.
4.	Encoding
o	gender and facility_id encoded using LabelEncoder.
5.	Scaling
o	Applied StandardScaler to all numeric features.
6.	Feature Selection
o	Used Mutual Information and Correlation Analysis to understand which variables influence length_of_stay.
o	Dropped low-importance and redundant columns listed in X_drop_cols.
7.	Multicollinearity Check
o	Used Variance Inflation Factor (VIF) to ensure no strong linear dependencies among predictors.
8.	Train-Test Split
o	80% training and 20% testing (random_state=42).
________________________________________
🧠 Model Development
All models were built using a Pipeline that included the preprocessing transformer and the regression model.
Models Trained
Model	Description
K-Nearest Neighbors (KNN)	Simple, non-parametric baseline model
Decision Tree Regressor	Interpretable tree-based model
Random Forest Regressor	Ensemble of trees reducing variance
AdaBoost Regressor	Sequential ensemble improving weak learners
Gradient Boosting Regressor	Sequential ensemble minimizing bias
XGBoost Regressor	Optimized gradient boosting (fast, accurate)
Linear Regression	Simple baseline for linear relationships
SVR (Support Vector Regressor)	Tried for non-linear patterns (not finalized in output)
________________________________________
📈 Model Evaluation
Models were evaluated using R² Score (explained variance) and RMSE (average prediction error).
Model	R²	RMSE	Remarks
KNN	0.7638	1.1386	Strong local baseline
Decision Tree	0.6904	1.3033	Slight overfitting
Random Forest	0.7816	1.0946	Stable, good performance
AdaBoost	0.1384	2.1743	Underperformed on this dataset
Gradient Boost	0.8072	1.0286	Excellent predictive strength
XGBoost	0.8109	1.0187	✅ Best model overall
Linear Regression	0.7577	1.1530	Solid baseline, interpretable
🧩 Interpretation:
RMSE measures the average error magnitude in the same unit as the target.
Since the target is length of stay (days), an RMSE ≈ 1.02 means the model’s predictions are, on average, about one day off from the actual stay duration.
________________________________________
🏆 Final Model
•	Chosen Model: XGBoost Regressor
•	Saved Pipeline: regression_model_rf.pkl (contains XGBoost model despite filename)
•	Evaluation: R² = 0.8109, RMSE = 1.0187
________________________________________
📦 Tools & Libraries
•	Programming Language: Python
•	Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Statsmodels, XGBoost
•	Deployment: Streamlit + Hugging Face Spaces
________________________________________
📊 Visualizations in Notebook
•	Correlation heatmap between features and target
•	Mutual Information ranking plot
•	VIF table for multicollinearity check
•	Model performance comparison (R² & RMSE)
________________________________________
💡 Insights & Business Impact
•	Features like readmission_count, certain lab results, and comorbidities influence stay length.
•	Ensemble models (especially XGBoost) handle these complex relationships best.
•	The project demonstrates how ML can support hospital resource optimization, capacity forecasting, and better patient flow management.
________________________________________
🚀 Deployment
A simple Streamlit web app was created and deployed on Hugging Face Spaces.
It allows users to input patient data and predict the estimated hospital stay duration.
🔗 Live Demo: Patient Stay Estimator – Hugging Face
________________________________________
📁 Repository Structure
│
├── ML_Regression_Model.ipynb      # Main Jupyter notebook (EDA + Modeling)
├── regression_model_rf.pkl        # Saved XGBoost model pipeline
├── app.py                         # Streamlit web app (for Hugging Face)
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
________________________________________
🧾 Conclusion
•	Gradient Boosting and XGBoost provided the most reliable predictions.
•	Achieved an R² ≈ 0.81, indicating strong explanatory power for real-world hospital data.
•	RMSE ≈ 1.02 shows high prediction precision.
•	Demonstrates strong data cleaning, feature engineering, and model comparison workflow.
________________________________________
✨ Author
Divya
📫 LinkedIn Profile
💻 Data Science & Machine Learning Enthusiast
 
💼 LinkedIn Post (Corrected)
🚑 Project: Patient Stay Estimator — Regression Model
Hospitals need accurate estimates of how long patients will stay for better bed management and resource planning. I built a regression model that predicts Length of Stay (LOS, in days) using encounter-level hospital data and clinical variables.
What I did
•	Cleaned and renamed columns, extracted date features, encoded categorical variables, and scaled numeric values
•	Selected features using mutual information and correlation analysis, and checked multicollinearity (VIF)
•	Trained and compared multiple regressors: KNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, and Linear Regression
•	Evaluated models using R² and RMSE
Result
•	XGBoost achieved R² = 0.81 and RMSE = 1.02,
meaning the model’s predictions differ from actual stay length by roughly one day on average
Impact
Helps hospitals forecast patient stays more accurately, optimize bed allocation, and improve discharge planning.
Tech: Python, pandas, scikit-learn, XGBoost, seaborn, matplotlib, Streamlit (deployed on Hugging Face Spaces)
👉 Try the demo: https://huggingface.co/spaces/divya55/patient_stay_estimator-Regression_Model
#MachineLearning #HealthcareAI #Regression #XGBoost #DataScience
 

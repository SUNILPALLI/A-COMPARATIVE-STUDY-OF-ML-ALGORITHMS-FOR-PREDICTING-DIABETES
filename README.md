🩺 A Comparative Study of Machine Learning Algorithms for Predicting Diabetes
📌 Project Overview

Diabetes is a rapidly increasing health concern worldwide. Early detection can significantly reduce complications and improve patient outcomes.
This project presents a comparative analysis of multiple supervised Machine Learning algorithms to predict whether a patient is diabetic or not using clinical and demographic data.

The main goal is to evaluate and compare different ML models based on performance metrics such as Accuracy, Precision, Recall, F1-Score, and Mean Squared Error (MSE), and identify the most effective algorithm for diabetes prediction.

🎯 Objectives

To build predictive models for diabetes classification

To implement and compare multiple machine learning algorithms

To evaluate model performance using standard classification metrics

To identify the best-performing algorithm for diabetes prediction

🧠 Algorithms Implemented

The following six supervised learning algorithms are implemented and evaluated:

Logistic Regression

K-Nearest Neighbors (KNN)

Naïve Bayes

Support Vector Machine (SVM)

Decision Tree

Random Forest

📊 Dataset Description

Source: Kaggle – Diabetes Dataset

Total Records: 2000

Features: 8 input features + 1 target variable

Features:
Feature	Description
Pregnancies	Number of times pregnant
Glucose	Plasma glucose concentration
Blood Pressure	Diastolic blood pressure
Skin Thickness	Triceps skinfold thickness
Insulin	2-hour serum insulin
BMI	Body Mass Index
Diabetes Pedigree Function	Genetic influence
Age	Patient age
Outcome	Target (1 = Diabetic, 0 = Non-Diabetic)
🛠️ Methodology

Data Collection – Dataset obtained from Kaggle

Data Preprocessing

Handling missing values

Feature scaling using StandardScaler

Data normalization

Train-Test Split

80% Training

20% Testing

Model Training

Train six ML classifiers

Model Evaluation

Compare models using performance metrics

Model Selection

Identify best algorithm based on results

📈 Performance Metrics

The models are evaluated using:

Accuracy

Precision

Recall

F1-Score

Mean Squared Error (MSE)

These metrics help assess both prediction correctness and model reliability.

💻 Technologies Used
Programming Language

Python 3.10+

Libraries & Tools

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

ML Modules Used

StandardScaler

train_test_split

LogisticRegression

KNeighborsClassifier

GaussianNB

DecisionTreeClassifier

RandomForestClassifier

SVC

accuracy_score

precision_score

recall_score

f1_score

mean_squared_error

🖥️ System Requirements
Hardware

Processor: 500 MHz or above

RAM: 4 GB

Storage: 4 GB

Software

OS: Windows 10 / 11

IDE: Visual Studio Code

Python: 3.10+

📌 Results & Conclusion

All six models successfully predicted diabetes with varying accuracy.

Random Forest and SVM demonstrated superior performance compared to other algorithms.

Logistic Regression provided good interpretability but slightly lower accuracy.

Ensemble models showed better robustness and generalization.

🚀 Future Enhancements

Use larger and real-time healthcare datasets

Apply deep learning models

Perform hyperparameter tuning

Deploy the model as a web application

Integrate real-time prediction dashboards

👨‍💻 Author

Palli Sunil
B.Tech (Information Technology)
Machine Learning | Data Science | DevOps Enthusiast

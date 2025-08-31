# Heart-Disease-Prediction
Heart Disease Prediction using Machine Learning
This project aims to predict whether a person has heart disease based on various medical attributes. A Logistic Regression model is trained on a dataset of patient information to classify them into two categories: having heart disease or not.

📋 Table of Contents
Project Overview

Dataset

Workflow

Technologies Used

Model Performance

How to Use

📖 Project Overview
The primary goal of this project is to build a machine learning model that can provide an early prediction of heart disease. By analyzing key medical features, the model can assist medical professionals in identifying patients at risk. We use a classic classification algorithm, Logistic Regression, for this purpose.

📊 Dataset
The dataset used for this project is heart.csv, which contains 1025 records of patient data with 14 medical attributes.

Features:

age: Age of the patient

sex: Sex of the patient (1 = male; 0 = female)

cp: Chest pain type

trestbps: Resting blood pressure

chol: Serum cholesterol in mg/dl

fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)

restecg: Resting electrocardiographic results

thalach: Maximum heart rate achieved

exang: Exercise induced angina (1 = yes; 0 = no)

oldpeak: ST depression induced by exercise relative to rest

slope: The slope of the peak exercise ST segment

ca: Number of major vessels (0-3) colored by fluoroscopy

thal: Thalassemia (a blood disorder)

Target Variable:

target: 1 (diseased heart) or 0 (healthy heart)

⚙️ Workflow
The project follows a standard machine learning pipeline:

Data Collection & Processing:

Load the dataset using Pandas.

Perform an initial exploratory data analysis (EDA) to understand the data's structure, check for missing values, and view statistical summaries.

Feature and Target Splitting:

Separate the dataset into features (X) and the target variable (Y).

Data Splitting:

Divide the data into training (80%) and testing (20%) sets using train_test_split from Scikit-learn.

Model Training:

Train a Logistic Regression model on the training data (X_train, Y_train).

Model Evaluation:

Evaluate the model's performance by calculating the accuracy score on both the training and testing data to check for overfitting.

Predictive System:

Build a simple system to take new input data and predict whether the patient has heart disease.

💻 Technologies Used
Python 3

NumPy: For numerical operations and creating NumPy arrays.

Pandas: For data manipulation and loading the CSV file.

Scikit-learn: For model building, data splitting, and performance evaluation.

Jupyter Notebook: For interactive development and documentation.

📈 Model Performance
The trained Logistic Regression model achieved the following accuracy:

Accuracy on Training Data: 85.24%

Accuracy on Testing Data: 80.49%

🚀 How to Use
To run this project on your local machine, follow these steps:

Clone the repository:

git clone [https://github.com/YourUsername/Heart-Disease-Prediction.git](https://github.com/YourUsername/Heart-Disease-Prediction.git)
cd Heart-Disease-Prediction

Install the required libraries:

pip install numpy pandas scikit-learn

Download the dataset:

Ensure the heart.csv file is in the same directory as the notebook.

Run the Jupyter Notebook:

Open and run the Heart_Disease_Prediction.ipynb file in a Jupyter environment.

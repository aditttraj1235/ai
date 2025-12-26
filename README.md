## Student Performance Prediction
Decision Tree vs Naive Bayes (Machine Learning Demo)

This project demonstrates how machine learning classification algorithms can be used to predict student performance levels based on academic, demographic, and social factors. Instead of predicting exact marks, students are classified into Low, Medium, or High performance categories.

📌 Project Overview

Educational institutions can benefit from early identification of students who may need academic support. This project applies two popular classification algorithms:

Decision Tree Classifier

Gaussian Naive Bayes

Both models are trained and evaluated on the Student Performance Dataset, and their results are compared using accuracy, classification reports, and confusion matrices.

📂 Dataset

Source: Student Performance Dataset (Math)

Target Variable: performance

Performance Levels:

Low: 0–9

Medium: 10–14

High: 15–20

🔢 Features Used

Numeric:
age, absences, G1, G2, studytime, failures, traveltime

Categorical:
School, gender, family background, study support, activities, internet access, etc.

⚙️ Data Preprocessing

Converted final grade (G3) into categorical performance levels

Scaled numeric features using StandardScaler

Encoded categorical features using One-Hot Encoding

Used ColumnTransformer for clean preprocessing

Performed stratified train–test split (75% / 25%)

🧠 Machine Learning Models
🌳 Decision Tree Classifier

Non-parametric supervised learning algorithm

Splits data based on information gain

Easy to interpret and visualize

Handles feature interactions effectively

📊 Gaussian Naive Bayes

Probabilistic classifier based on Bayes’ theorem

Assumes feature independence

Very fast and efficient

Works well with large datasets

📈 Model Evaluation

The models are evaluated using:

Accuracy score

Precision, Recall, and F1-score

Confusion Matrix

5-Fold Cross Validation

🔍 Results Summary

Decision Tree provides better interpretability

Naive Bayes performs faster with simple assumptions

Previous grades (G1, G2) strongly influence predictions

📊 Visualizations

Confusion matrices for both classifiers

Decision Tree visualization (top levels)

Clear distinction between correct and incorrect predictions

🔮 Live Prediction Example

A sample student profile is created to demonstrate real-time prediction.
Both models output:

Predicted performance level

Class probability distribution

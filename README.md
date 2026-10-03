# COVID-19 Adverse Effects Prediction

## Introduction

This project focuses on building a practical and reliable **risk prediction system using Machine Learning** to analyse adverse effects associated with COVID-19.

Throughout the project, we explored the challenges involved in working with real-world datasets. The experience demonstrated that building an effective machine learning system is not simply about achieving high accuracy. It also requires understanding **data imbalance, model behaviour, consistency, and generalisation**.

Two machine learning approaches, **Random Forest** and **XGBoost**, were evaluated during the project. Although both models demonstrated strong performance, their behaviour highlighted an important distinction between high predictive performance and reliable generalisation.

Random Forest produced more realistic and consistent predictions, while the near-perfect performance observed with XGBoost suggested the possibility of **overfitting**. This reinforced the importance of evaluating models beyond a single performance metric.

Handling **class imbalance** was another major challenge. Several approaches were explored to address the imbalance before selecting a more stable solution.

Overall, the project demonstrates the practical application of machine learning to a real-world prediction problem. It also highlights the importance of making informed decisions based on **data quality, distribution, model behaviour, and generalisation**, rather than relying solely on accuracy.

---

## Project Workflow

The overall workflow followed during the project is illustrated below:

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/9c671e12-f607-46e0-b6a6-575d0c41308f"
    width="298"
    alt="COVID-19 adverse effects prediction project workflow"
  />
</p>

---

## Tools & Technologies

- **Python**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Random Forest**
- **XGBoost**

---

## Machine Learning Models

### Random Forest

Random Forest was used as one of the primary prediction models. Its ensemble-based approach provided relatively consistent and realistic predictions across the dataset.

### XGBoost

XGBoost was also evaluated because of its strong performance on structured/tabular datasets.

Although it achieved very high performance during evaluation, its near-perfect results raised concerns about possible **overfitting** and highlighted why model generalisation should be considered alongside performance metrics.

---

## Key Challenges

### 1. Data Imbalance

The dataset contained an imbalance between different outcome classes. Different techniques were explored to reduce the impact of this imbalance on model training and prediction.

### 2. Model Behaviour

The project demonstrated that two models can achieve strong evaluation results while behaving differently when making predictions.

### 3. Overfitting

The near-perfect performance of XGBoost indicated the possibility that the model was learning patterns too specifically from the training data.

### 4. Generalisation

The project emphasised the importance of developing models that perform consistently on unseen data rather than simply maximising training or evaluation scores.

---

## Key Learning Outcomes

Through this project, we gained practical experience in:

- Data preprocessing
- Exploratory data analysis
- Handling imbalanced datasets
- Feature preparation
- Machine learning model development
- Random Forest
- XGBoost
- Model evaluation
- Identifying potential overfitting
- Comparing model behaviour
- Understanding the importance of generalisation

---

## Group Project

This project was developed as a **group project** by:

- **Subhadip Jana**
- **Shrijit Sengupta**
- **Ayan Biswas**
- **Souvik Patra**

---

## Under the Guidance of

**Dr. Tahamina Yesmin**  
Assistant Professor  
Department of Computer Science and Engineering  
**Adamas University**

---

## Project Demonstration

A video demonstration of the project is available below:

[▶️ Watch the Project Demonstration on YouTube](https://youtu.be/X_I3c1Eje70)

---

## Project Summary

The project provided practical experience in developing a machine learning-based prediction system while demonstrating an important lesson in applied ML:

> **A model that performs well on paper is not automatically a model that generalises well in the real world.**

The comparison between Random Forest and XGBoost, along with the challenges of class imbalance and potential overfitting, helped us understand the importance of evaluating machine learning systems from multiple perspectives rather than focusing on a single metric.

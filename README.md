# 🧪 Exploratory Data Analysis on Superstore Dataset using Python in Jupyter Notebook

This project involves performing **Exploratory Data Analysis (EDA)** on the popular **Sample Superstore** dataset to uncover actionable business insights and trends. Additionally, a **Streamlit-based web app** is built for predicting whether an order leads to **profit or loss** using a trained **Random Forest model**.

---

## 📌 Project Objectives

- Clean and explore the dataset using Python.
- Visualize trends in sales, profit, category, and region.
- Build and deploy a machine learning model to classify order outcomes.
- Create a Streamlit app for real-time predictions.

---

## 🧰 Tools & Technologies Used

- **Languages**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit, Joblib
- **Platform**: Jupyter Notebook, GitHub, Streamlit
- **ML Algorithm**: Random Forest Classifier

---

## 📁 Repository Structure

├── Sample Superstore EDA.ipynb # Jupyter notebook for data analysis
├── Sample Superstore Steamlit/
│ ├── app.py # Streamlit application script
│ ├── SampleSuperstore.csv # Dataset used for prediction
│ └── random_forest_model.pkl # Trained ML model
└── README.md # Project documentation


---

## 📊 Key EDA Insights

- **Top Category**: Technology contributes the highest profit.
- **Loss Areas**: Tables and Bookcases often lead to losses.
- **Region-wise Trends**: West region is the most profitable.
- **Discount Impact**: Higher discounts can lead to losses beyond a threshold.

---

## 🚀 Streamlit Web App

A web interface for real-time prediction of **Profit/Loss** based on order input features using a trained Random Forest model.

### 🔮 Features

- Upload new order data and get profit/loss predictions.
- Simple and interactive UI.
- Can be extended for business dashboard integration.

### 🏃 How to Run the App

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Exploratory-Data-Analysis-on-Superstore-Dataset-using-Python-in-Jupyter-nootbook.git
2. Navigate to the app folder:
   cd "Exploratory-Data-Analysis-on-Superstore-Dataset-using-Python-in-Jupyter-nootbook/Sample Superstore Steamlit"
3. Install dependencies:
   pip install -r requirements.txt
4. Run the app:
   streamlit run app.py

### 📦 Requirements
To run this project, install the following libraries:
streamlit
pandas
scikit-learn
joblib
matplotlib
seaborn

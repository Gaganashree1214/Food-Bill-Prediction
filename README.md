# 🍜 BillAI — End-to-End Food Delivery Bill Prediction

## 📌 Overview

**BillAI** is an end-to-end **Machine Learning project** that predicts the final food delivery bill using real-world order parameters.

The project covers the complete pipeline:
➡️ Data Cleaning
➡️ Feature Engineering
➡️ Model Building
➡️ Deployment using Streamlit

It simulates how companies like Swiggy/Zomato estimate pricing dynamically.

---

## 🚀 Key Features

* 🔮 ML-based bill prediction (Linear Regression)
* 🧹 Data cleaning & preprocessing pipeline
* 🏗️ Feature engineering (derived variables like total price, discount amount)
* 🎯 One-hot encoding for categorical features
* 📊 Interactive dashboard using Streamlit
* 🎨 Custom modern UI (CSS-based)

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Cleaning

* Handled missing values
* Removed inconsistencies in categorical data
* Converted data types (e.g., price columns)
* Cleaned special characters (₹, commas, etc.)

### 2️⃣ Feature Engineering

* Created:

  * `total_items_price`
  * `discount_amount`
* Encoded:

  * Restaurant name
  * Cuisine type
  * Meal time
* Converted:

  * Gender → Binary
  * Weekend → Binary

### 3️⃣ Model Training

* Algorithm: **Linear Regression**
* Train-test split applied
* Model evaluated using:

  * R² Score
  * Mean Absolute Error (MAE)

### 4️⃣ Model Saving

```python
joblib.dump(model, "model.pkl")
```

---

## 🏗️ Project Structure

```
BillAI/
│
├── app.py                  # Streamlit app (UI + prediction)
├── model.pkl              # Trained ML model
├── notebook.ipynb         # Data cleaning + training
├── requirements.txt       # Dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repo

```bash
git clone https://github.com/your-username/BillAI.git
cd BillAI
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run App

```bash
streamlit run app.py
```

---

## 📊 Model Features Used

* Number of items
* Average item price
* Discount %
* Delivery distance
* Delivery rating
* Customer age & gender
* Weekend order
* Previous orders
* Restaurant (encoded)
* Cuisine type (encoded)
* Meal time (encoded)

---

## 🖥️ Application Preview

* Enter order details
* Click **"Calculate Estimated Bill"**
* Get:

  * 💰 Final predicted bill
  * 📉 Discount savings
  * 📋 Order breakdown

---
## 🌐 Try live 
https://food-bill-prediction-lycbyh6fmlz7oqkzvjrh5h.streamlit.app/

---

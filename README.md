# House-prediction-using-Neural-Networks-
# 🏡 House Price Prediction using Neural Network

This project implements a **House Price Prediction Model** using a **Feedforward Neural Network** built with **TensorFlow/Keras**. The dataset used is the popular **King County House Prices** dataset.

---

## 📂 Dataset

- Source: [King County House Price Dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
- Contains details of **house sales in King County, USA**, including prices, house size, number of bedrooms/bathrooms, lot size, and more.

---

## ⚙️ Preprocessing Steps

The raw dataset undergoes these transformations:

1. **Date Processing:** Extract the year of sale (`reg_year`).
2. **House Age Calculation:**  
   - If the house was **never renovated**, house age = sale year - year built.
   - If the house was **renovated**, house age = sale year - year renovated.
3. **Column Removal:** Drop unnecessary columns (date, coordinates, zipcode).
4. **Invalid Data Handling:** Remove houses with invalid age (-1).

5. **Feature Scaling:**  
   All input features are scaled using **StandardScaler** from `sklearn` to ensure even contribution to training.

---

## 🚀 Model Architecture

The neural network consists of:

- Input Layer: 14 features
- Hidden Layers:
    - Dense (64 units) - ReLU activation
    - Dense (32 units) - ReLU activation
    - Dense (16 units) - ReLU activation
- Output Layer: 1 neuron (predicting price)

---

## 🧰 Tools & Libraries

| Tool/Library | Purpose |
|---|---|
| **pandas** | Data loading & preprocessing |
| **numpy** | Numerical operations |
| **matplotlib & seaborn** | Visualization |
| **scikit-learn** | Train-Test Split & Scaling |
| **TensorFlow / Keras** | Neural Network |

---

## 📊 Training Details

- Loss Function: **Mean Squared Error (MSE)**
- Optimizer: **Adam**
- Epochs: **100**
- Batch Size: **32**
- Validation Split: **33%** (manual split using `train_test_split`)

---

## 📈 Results - Loss Curve

The training process produced is available as a picture.


---

## 🔮 Prediction Example

Once trained, the model can predict house prices for new data. Example prediction flow:

```python
Xnew = np.array([[2, 3, 1280, 5550, 1, 0, 0, 4, 7, 2280, 0, 1440, 5750, 60]])
Xnew_scaled = scaler.transform(Xnew)
Ynew = model.predict(Xnew_scaled)
print("Predicted house price:", Ynew[0][0])
📬 Author
Speranza Deejoe

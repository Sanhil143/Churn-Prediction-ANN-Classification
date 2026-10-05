# Customer Churn Prediction (ANN)

An Artificial Neural Network that predicts whether a bank customer will leave the bank (churn), with a Streamlit web app for interactive predictions.

## Dataset

[`Churn_Modelling.csv`](Churn_Modelling.csv) has 10,000 bank customers. The target column is `Exited` (1 = left the bank, 0 = stayed).

Input features used by the model:

| Feature | Description | Preprocessing |
|---|---|---|
| CreditScore | Credit score | Scaled |
| Geography | France / Germany / Spain | One-hot encoded |
| Gender | Male / Female | Label encoded |
| Age | Age in years | Scaled |
| Tenure | Years with the bank | Scaled |
| Balance | Account balance | Scaled |
| NumOfProducts | Number of bank products used | Scaled |
| HasCrCard | Has a credit card (0/1) | Scaled |
| IsActiveMember | Active member (0/1) | Scaled |
| EstimatedSalary | Estimated salary | Scaled |

`RowNumber`, `CustomerId` and `Surname` are dropped because they don't help prediction.

## Model

```
Input (12 features)
  → Dense(64, ReLU)
  → Dense(32, ReLU)
  → Dense(1, Sigmoid)  → churn probability
```

- **Loss:** Binary cross-entropy
- **Optimizer:** Adam (learning rate 0.001)
- **Train/test split:** 80/20 (`random_state=42`)
- **Callbacks:** EarlyStopping (`val_loss`, patience 10, restores best weights) and TensorBoard
- **Validation accuracy:** ~86%

A customer is predicted to churn when the probability is above 0.5.

## Project structure

| File | Purpose |
|---|---|
| [`experiments.ipynb`](experiments.ipynb) | Data preprocessing, model training, saving the model and encoders |
| [`prediction.ipynb`](prediction.ipynb) | Step-by-step prediction on a single example customer |
| [`app.py`](app.py) | Streamlit web app |
| `ann_model.h5` | Trained Keras model |
| `scaler.pkl` | Fitted `StandardScaler` |
| `label_encoder_gender.pkl` | Fitted `LabelEncoder` for Gender |
| `onehot_encoder_geography.pkl` | Fitted `OneHotEncoder` for Geography |
| [`requirement.txt`](requirement.txt) | Python dependencies |

## Setup

Requires Python 3.11.

```bash
git clone https://github.com/<username>/ANN-Implementation.git
cd ANN-Implementation

# Create and activate an environment (conda)
conda create -p ./venv python=3.11 -y
conda activate ./venv

# Install dependencies
pip install -r requirement.txt
```

Using plain `venv` instead of conda:

```bash
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# source venv/bin/activate     # macOS / Linux
pip install -r requirement.txt
```

## Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501, enter the customer's details, and the app shows the churn probability and whether the customer is likely to churn.

## Retrain the model

Run all cells in [`experiments.ipynb`](experiments.ipynb). This regenerates `ann_model.h5` and the three `.pkl` files.

To view training curves in TensorBoard:

```bash
tensorboard --logdir logs/fit
```

## Prediction pipeline

The app applies the same preprocessing that was used during training:

1. Label-encode `Gender` with the saved `LabelEncoder`
2. One-hot encode `Geography` with the saved `OneHotEncoder`
3. Scale all features with the saved `StandardScaler` (`transform` only, never `fit`)
4. `model.predict()` returns the churn probability

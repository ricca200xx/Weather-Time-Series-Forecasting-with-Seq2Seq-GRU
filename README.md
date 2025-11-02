# ☁️ Weather Time-Series Forecasting with Seq2Seq GRU

## 🎯 Project Overview

This project presents the solution developed for the **Deep Learning Course Challenge** on Kaggle, which focused on multivariate time-series forecasting of weather data. The primary goal was to demonstrate the understanding and application of deep learning architectures for sequential data analysis.

The core solution implements a **Sequence-to-Sequence (Seq2Seq)** model based on **Gated Recurrent Units (GRU)** to predict future weather conditions across multiple stations.

### Forecasting Objective

The task was a many-to-many sequence prediction:
* **Input:** Utilize a historical sequence of **90 time steps**.
* **Output:** Predict **76 weather variables** for the subsequent **30 time steps**.

### Evaluation Metric

The model was optimized to minimize the **Mean Absolute Scaled Error (MASE)**, the official metric of the competition, which provides a scaled measure of forecasting accuracy.

---

## 🧠 Model Architecture: Seq2Seq GRU

The **Seq2Seq** architecture was chosen for its capability to map a fixed-length input sequence ($T_{in}=90$) to a fixed-length output sequence ($T_{out}=30$).

### 1. GRU Encoder
The Encoder processes the input sequence and summarizes the salient information into a **context vector** (the final hidden state).

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Layer Type** | Single-layer GRU | Efficient for capturing temporal dependencies. |
| **Input Size** | 76 | The number of weather variables. |
| **Hidden Size** | 128 | The dimension of the hidden state, which acts as the context vector. |
| **Dropout** | 0.4 | Regularization applied to mitigate overfitting. |

### 2. GRU Decoder
The Decoder receives the context vector from the Encoder as its initial state and generates the output sequence, one time step at a time.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Layer Type** | Single-layer GRU | Generates the output sequence. |
| **Output Size** | 76 | The 76 variables to predict for each step. |
| **Output Length** | 30 | The length of the forecasted sequence. |

---

## 📈 Training Strategy and Regularization

Several best practices were implemented to ensure stable training, rapid convergence, and strong generalization:

* **Teacher Forcing:** Used to stabilize training. A **Teacher Forcing Ratio** (starting at 0.5 and linearly decaying) balances feeding the ground truth vs. the model's own previous prediction into the next decoding step.
* **Data Augmentation:** A small amount of Gaussian noise ($\text{Noise} = 0.015$) was injected into the input sequences during training to improve model robustness.
* **Optimizer:** **AdamW** was used with **Weight Decay** ($\text{1e-2}$) for optimization and L2 regularization.
* **Scheduler:** The **ReduceLROnPlateau** scheduler dynamically reduces the learning rate when the validation loss plateaus.
* **Early Stopping:** Implemented with a patience of 10 epochs to stop training once overfitting on the validation set began.
* **Best Result:** The best Mean Absolute Error (MAE) loss achieved on the validation set was **0.5566** (Epoch 2).

---

## 💾 Dataset and Preprocessing

The underlying data is a complex **multivariate time-series dataset** provided in the file `train_dataset.csv`.

### Dataset Characteristics

* **Number of Stations (`N_STATIONS`):** 422
* **Number of Variables (`N_VARS`):** 76
* **Sequencing:** The dataset consists of 422 inherently correlated time series.

### Preprocessing Pipeline

1.  **Scaling:** All 76 variables were **standardized** using `StandardScaler` from scikit-learn. This is crucial for stabilizing RNN training.
2.  **Train/Validation Split:** The split was performed **chronologically**. The validation set consists of the very last non-overlapping sequence (90 input + 30 output) available for each of the 422 stations.
    * Training Samples: 192,432 sequences
    * Validation Samples: 422 sequences

---

## ⚙️ Setup and Usage

### 1. Requirements

The project dependencies are managed via `requirements.txt`:

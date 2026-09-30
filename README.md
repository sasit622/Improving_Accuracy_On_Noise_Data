# 🧠 Data-Centric Neural Optimizer

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

**An empirical study on noisy dataset refinement and multi-model benchmarking.**

The **Data-Centric Neural Optimizer** is an interactive dashboard that demonstrates the profound impact of data quality on machine learning performance. Instead of focusing solely on model architecture, this tool focuses on **refining the data** (removing outliers, handling missing values) and empirically proving how much better models perform on clean data versus raw, noisy data.

---

## 📸 Screenshots

| Dashboard Overview | Dataset Overview |
| :---: | :---: |
| ![Dashboard](image/1) | ![Dataset Overview](image/2) |

| Feature Summary | Outlier Detection & Feature Selection |
| :---: | :---: |
| ![Feature Summary](image/3) | ![Feature Selection](image/4) |

| Refinement Analysis | Model Comparison (Random Forest) |
| :---: | :---: |
| ![Refinement Analysis](image/5) | ![Random Forest](image/6) |

| Model Comparison (Logistic Regression) | Model Comparison (ANN) |
| :---: | :---: |
| ![Logistic Regression](image/7) | ![ANN](image/8) |

| Summary Leaderboard & Export |
| :---: |
| ![Summary Leaderboard](image/9) |

---

## ✨ Key Features

*   **📁 Universal Data Upload:** Supports CSV, XLSX, and XLS files up to 200MB.
*   **🔍 Automated EDA:** Instant dataset overview, feature type detection (Numeric, Categorical), missing value analysis, and statistical summaries.
*   **🧹 Data Refinement Pipeline:**
    *   Automatic handling of missing rows.
    *   **KMeans Outlier Detection:** Users can select specific numeric features and set the number of clusters to identify and remove outliers intelligently.
*   **🤖 Multi-Model Benchmarking:** Train and evaluate up to 14 different models simultaneously, including:
    *   *Classical ML:* Random Forest, Logistic Regression, SVM, KNN, Decision Tree, Naive Bayes, Gradient Boosting, AdaBoost, Ridge, Lasso.
    *   *Deep Learning:* ANN (TensorFlow/Keras), RNN/LSTM (TensorFlow/Keras).
*   **📊 "Before vs. After" Comparison:** The core feature. The app trains models on the **Raw Dataset** and the **Refined Dataset** and displays side-by-side metrics.
*   **📈 Refinement Impact Metrics:** Clearly shows the **Accuracy Δ** and **F1 Score Δ** to quantify exactly how much the data cleaning helped (or hurt) the model.
*   **⚙️ Hyperparameter Tuning:** Adjust ANN Epochs, RNN/LSTM Epochs, Batch Size, and KMeans clusters directly from the sidebar.
*   **📥 Export Results:** Download the refined dataset as a CSV for use in external pipelines.

---

## 🛠️ Tech Stack

*   **Frontend/UI:** [Streamlit](https://streamlit.io/)
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn
*   **Deep Learning:** TensorFlow / Keras
*   **Visualization:** Streamlit native components & custom CSS

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/data-centric-neural-optimizer.git
    cd data-centric-neural-optimizer
    

├── app.py                  # Main Streamlit application entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── image/                  # Screenshots for documentation (1.png, 2.png, etc.)
├── src/                    # Source code (if applicable)
│   ├── data_loader.py      # Functions for reading CSV/Excel
│   ├── preprocessing.py    # Missing value handling, encoding
│   ├── outlier_detection.py# KMeans clustering logic
│   ├── models.py           # Model definitions (Sklearn & TensorFlow)
│   └── pipeline.py         # Orchestration of training and evaluation
└── data/                   # Sample datasets (optional)

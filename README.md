# Sentiment Analysis with Recurrent Neural Network (RNN)

## Objective

The objective of this project is to implement a Recurrent Neural Network (RNN) using TensorFlow to perform sentiment analysis on the IMDB movie review dataset. The RNN is trained to classify movie reviews as either positive or negative, and its performance is evaluated using standard metrics.

## Project Overview

This project involves:

1. **Data Preprocessing:** Preparing the IMDB dataset by tokenizing text, padding sequences, and splitting into training and testing sets.
2. **Model Implementation:** Building an RNN using TensorFlow and Keras, incorporating layers such as embedding, LSTM/GRU, and dense layers.
3. **Training:** Training the model on the preprocessed dataset and optimizing its performance.
4. **Evaluation:** Analyzing the model’s accuracy, loss, precision, recall, and F1-score using validation and test data.
5. **Hyperparameter Tuning:** Improving model performance through tuning of learning rate, batch size, number of units, and other hyperparameters.

## Key Features

- **Dataset:** The IMDB movie review dataset, containing 50,000 labeled movie reviews.
- **Model Architecture:** A Recurrent Neural Network with embedding and LSTM/GRU layers.
- **Performance Metrics:**
  - Training Accuracy
  - Validation Accuracy
  - Test Accuracy
  - Precision, Recall, and F1-score from confusion matrix analysis

## Results

### Model Performance

#### Model 1 (Baseline RNN):

- **Training Accuracy:** 91.98%
- **Validation Accuracy:** 82.96%
- **Test Accuracy:** 82.47%
- **F1-Score:** 0.8277

#### Model 2 (After Hyperparameter Tuning):

- **Training Accuracy:** 98.97%
- **Validation Accuracy:** 85.96%
- **Test Accuracy:** 87.47%
- **F1-Score:** 0.8368

#### Model 3 (Feedforward Neural Network):

- **Training Accuracy:** 96.44%
- **Validation Accuracy:** 82.82%
- **Test Accuracy:** 82.6%
- **F1-Score:** 0.824

### Confusion Matrix Analysis

- True Positives, True Negatives, False Positives, and False Negatives were analyzed for each model to compute metrics such as precision, recall, and F1-score.

## Installation and Usage

### Prerequisites

- Python 3.7 or later
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib

### Steps to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rnn-sentiment-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd rnn-sentiment-analysis
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python train_rnn.py
   ```

### File Structure

- `train_rnn.py`: Main script for training and evaluating the RNN.
- `data/`: Directory containing the IMDB dataset.
- `models/`: Directory for saving trained models.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and experimentation.
- `results/`: Directory for storing results and visualizations.

## Key Insights

- RNNs with proper hyperparameter tuning can achieve high performance for sentiment analysis tasks.
- Overfitting can be mitigated using techniques such as dropout, regularization, and early stopping.
- Hyperparameter tuning significantly impacts model performance metrics.

## Future Work

- Explore advanced architectures such as Bidirectional LSTMs or Transformer models.
- Implement attention mechanisms to enhance performance.
- Extend the project to multi-class sentiment analysis.

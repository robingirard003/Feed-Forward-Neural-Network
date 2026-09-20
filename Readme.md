# Feed Forward Neural Network (NumPy Implementation)
This repository contains a fully manual implementation of a Feed Forward Neural Network built primarily using NumPy.
The project focuses on transparency, mathematical clarity, and hands‑on understanding of neural network training without relying on high‑level machine learning frameworks.

The model is trained using the Brier Score, and evaluated on both the Brier Score Loss and the Asymmetric Economic Loss, reflecting the dual objectives of statistical accuracy and economic relevance in credit risk modeling.

# Project Description
The goal of this project is to implement a neural network from scratch, including:

forward propagation

backpropagation

gradient computation

weight updates

mini batch

training loop

evaluation metrics

The implementation avoids libraries such as TensorFlow or PyTorch, relying instead on NumPy to highlight the underlying mechanics of neural networks.

The model is trained to predict default probabilities, and its performance is assessed using two complementary metrics:

Brier Score
A proper scoring rule measuring the accuracy of probabilistic predictions.

Asymmetric Economic Loss
A cost‑sensitive metric that penalizes misclassifications differently depending on their economic impact.

# Key Features
Pure NumPy neural network implementation

Custom training loop optimized for Brier Score minimization

Evaluation on asymmetric loss for realistic credit‑risk modeling

Modular code structure for readability and experimentation

CSV‑based dataset loading for easy reproducibility

# Repository Structure
Code
project/
│
├── X_trn.csv              # Training features
├── Y_trn.csv              # Training labels
├── X_test.csv             # Test features
│
└── Code_ML.py             # Neural network implementation

# Usage
Code_ML.py
Make sure the CSV files are placed in the project directory.

# Metrics
Metric	Description
Brier Score	Measures probabilistic accuracy. Lower is better.
Asymmetric Economic Loss	Penalizes errors based on economic cost asymmetry.


# Academic Context
This project was developed as part of the Machine Learning course at the Erasmus School of Economics, focusing on hands‑on implementation and evaluation of predictive models in credit risk.

# License
This repository is intended for academic and educational use. It uses MIT License.
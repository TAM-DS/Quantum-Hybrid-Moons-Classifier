# Hybrid Quantum-Classical Classifier on make_moons

A minimal, reproducible proof-of-concept hybrid quantum neural network (HQNN) for binary classification on the non-linear `make_moons` dataset.

Combines classical pre-processing with a variational quantum circuit (RealAmplitudes ansatz via Qiskit EstimatorQNN) wrapped as a PyTorch layer using TorchConnector.

Achieves **100.00% test accuracy** after 50 epochs — demonstrating effective learning of the decision boundary in this toy setting.

## Why This Project?
- Demonstrates seamless integration of Qiskit quantum circuits into PyTorch training loops
- Shows how small variational quantum layers can contribute to classification tasks
- Serves as a baseline for exploring quantum advantages in optimization-heavy domains (e.g., FinOps anomaly detection, portfolio modeling, multicloud resource allocation)

## Tech Stack
- **Quantum**: Qiskit 1.x (RealAmplitudes ansatz, StatevectorEstimator, EstimatorQNN, TorchConnector)
- **Classical/ML**: PyTorch (nn.Module, Adam optimizer), scikit-learn (make_moons, StandardScaler, train_test_split)
- **Environment**: Python 3.10+, Jupyter/Colab

## Results
- Dataset: make_moons (n=100, noise=0.1)
- Train/test split: 80/20
- Model: Classical linear + tanh → 1-qubit RealAmplitudes (reps=3) → linear output
- Optimizer: Adam (lr=0.01)
- Epochs: 50
- Final test accuracy: **100.00%** (perfect separation on this small, clean dataset)

Decision boundary (visualized in notebook):
![Decision Boundary](decision_boundary.png)  
*(Add screenshot of plot_decision_boundary output here)*

Final circuit:
![Quantum Circuit](circuit.png)  
*(Add qc.draw('mpl') screenshot here)*

## How to Run
1. Open in Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TAM-DS/Your-Repo-Name/blob/main/Hybrid_QNN_Moons.ipynb)

2. Install dependencies (first cell):
   ```bash
   !pip install qiskit qiskit-machine-learning qiskit-aer torch scikit-learn matplotlib --quiet

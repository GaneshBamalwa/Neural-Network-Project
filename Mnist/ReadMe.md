# Neural Network from Scratch (Python + NumPy)

Build + train a feedforward **neural network from first principles** — no TensorFlow/PyTorch abstractions.  
I implemented forward propagation, backpropagation, and gradient descent manually to deeply understand the math and mechanics.

> 🎯 **Why this repo?**  
> It demonstrates low-level mastery of neural nets (derivatives, chain rule, vectorization) and clean engineering (modular code, plots, reproducibility).


## ✨ Features
- Pure **Python + NumPy** implementation (no high-level DL frameworks)
- Configurable architecture: layers, activations, learning rate, batch size, epochs
- Vectorized **forward & backward** passes
- Loss/accuracy tracking and **training curves**
- Clean project layout for quick reading


## 🔎 Project Structure

project/
│── README.md<br>
│── requirements.txt<br>
│── .gitignore<br>
│── data/ # (optional: sample or download notes)<br>
│── notebooks/ # (optional: exploration/training logs)<br>
│── src/<br>
│ ├── model.py # NeuralNetwork class (init, forward, backward, update)<br>
│ ├── train.py # Training loop, evaluation, CLI args<br>
│ ├── utils.py # Metrics, plotting, data loaders, seeds<br>
│── results/<br>
│ ├── demo.mp4 # Demo video of predictions<br>


## 🚀 Quick Start

```bash
pip install -r requirements.txt
python src/train.py
```

## 📊 Results

**Training Performance (batch size = 32):**
```bash
Epoch 1 — Loss: 0.1916, Accuracy: 0.9409
Epoch 2 — Loss: 0.0776, Accuracy: 0.9769
Epoch 3 — Loss: 0.0594, Accuracy: 0.9829
Epoch 4 — Loss: 0.0503, Accuracy: 0.9859
Epoch 5 — Loss: 0.0445, Accuracy: 0.9877
Epoch 6 — Loss: 0.0407, Accuracy: 0.9892
Epoch 7 — Loss: 0.0378, Accuracy: 0.9904
Epoch 8 — Loss: 0.0355, Accuracy: 0.9910
Epoch 9 — Loss: 0.0336, Accuracy: 0.9916
Epoch 10 — Loss: 0.0320, Accuracy: 0.9922
```



## 🎥 Demo

[▶️ Watch the demo video](results/demo.mp4)




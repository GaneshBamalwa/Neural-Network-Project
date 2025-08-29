# My Neural Network Journey

This repository is not just code — it’s the story of how I built my first neural network **completely from scratch**.  
I avoided high-level libraries like TensorFlow or PyTorch, relying only on **Python + NumPy**. Every equation, derivative, and matrix multiplication was implemented by hand.

---

##  The Steps I Took

### 1. Understanding the Basics
- Started with the **mathematical foundation**:
  - Forward propagation equations for linear layers.  
  - Backpropagation using the **chain rule**.  
  - Gradient descent update rule: `weights -= learning_rate * gradient`.  
- Wrote everything out on paper before coding — so I knew exactly *why* each line of code existed.  

---

### 2. Implementing Forward Propagation
- Created a function to:
  - Multiply inputs by weights.  
  - Add biases.  
  - Apply activation functions (Sigmoid → later added ReLU and Softmax).  
- Debugged countless shape mismatches (`ValueError: shapes not aligned`).  
- The first successful forward pass gave me confidence: the outputs finally made sense.

---

### 3. Coding Backpropagation
- This was the hardest part:
  - Derived the gradients manually for each layer.  
  - Translated derivatives into NumPy operations.  
  - Implemented gradient descent updates.  
- Faced the classic **exploding/vanishing gradient problem**.  
- Realized how important proper weight initialization was (fixed with small random values).  

---

### 4. Training Loop
- Wrote my own loop to:
  - Pass inputs forward.  
  - Compute loss (cross-entropy for classification).  
  - Run backward pass.  
  - Update weights.  
- At first, the **loss didn’t change at all** → discovered a bug where I was averaging incorrectly.  
- After fixing, I finally saw the **loss decrease epoch by epoch**.  

---

### 5. Activation Functions
- Started only with **Sigmoid**, but it caused saturation issues.  
- Added **ReLU** → training sped up significantly.  
- Implemented **Softmax** for the output layer to handle multi-class classification.  

---

### 6. Debugging Milestones 
Some of the toughest bugs I hit:
- **NaN losses** → traced to division by zero in Softmax. Fixed by adding an epsilon.  
- **Gradients exploding** → solved by tweaking initialization.  
- **Wrong accuracy calculation** → was comparing floats instead of argmax labels.  

Every fix felt like a small win.

---

### 7. First Success 
- After many attempts, I got the model to:
  - Learn properly.  
  - Reduce loss smoothly.  
  - Cross **95%+ accuracy** on digit recognition.  
- That was the true “aha” moment — I wasn’t just running code, I had built a learning system from scratch.

---

##  Current Limitations
- Works fine for small digit classification tasks.  
- But when I tried **7-digit recognition** (multiple digits in sequence), performance dropped heavily.  
- A simple fully connected NN doesn’t capture spatial or sequential features well.  

---

##  Next Step: Moving Towards CNNs
- To overcome these limits, I’m now exploring **Convolutional Neural Networks (CNNs)**.  
- CNNs are designed to handle images better by capturing **local patterns** (edges, shapes, spatial hierarchies).  
- My scratch NN gave me the intuition I need to understand CNNs at a deeper level.

---

##  Learning Resources
Big thanks to the tutorials that guided me:  
- [Vizuara Neural Network Playlist](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSj6tNyn_UadmUeU3Q3oR-hu)  
- [Sentdex Deep Learning Series](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)  

They helped me connect theory to implementation.

---

This repo marks my **first real step into deep learning**.  
I built something real, broke it multiple times, fixed it, and learned more than I expected.  
The journey continues with CNNs 🚀

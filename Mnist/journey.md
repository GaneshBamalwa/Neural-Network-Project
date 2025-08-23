# My Neural Network Journey 🚀

This project is the result of my deep dive into **neural networks from scratch**.  
Instead of using TensorFlow or PyTorch, I wanted to build everything using only **Python + NumPy** so I could truly understand the math behind the magic.

---

## 🛠️ The Process

1. **Starting Point**  
   - Began with only the raw equations of forward propagation, backpropagation, and gradient descent.  
   - Learned how to translate derivatives and the chain rule into **vectorized NumPy code**.  

2. **Building the Core**  
   - Implemented a fully connected feedforward architecture.  
   - Added activation functions (Sigmoid, ReLU, Softmax).  
   - Manually coded the forward & backward passes.  

3. **Debugging Hell 🔥**  
   - Initial versions either diverged or gave **NaN losses**.  
   - Tackled problems like:
     - wrong weight initialization,  
     - exploding gradients,  
     - shape mismatches in matrix multiplications.  
   - Fixed them step by step — each bug taught me something new.  

4. **First Success 🎉**  
   - Finally saw the **loss curve decreasing smoothly** and accuracy crossing 95%+.  
   - That was the “aha” moment where the network *came alive*.  

---

## 🚧 Current Limitations

- While the model works great on small-scale digit recognition, it **struggles with more complex cases**, especially **7-digit recognition** tasks.  
- The fully connected layers quickly become inefficient and don’t capture spatial features well.  

---

## 🔮 Next Step: CNNs

To overcome the above challenges, I’m now moving towards **Convolutional Neural Networks (CNNs)**.  
CNNs handle spatial hierarchies much better and are the natural progression from this project.  
This NN-from-scratch work gave me the **intuition and foundation** I’ll carry into CNNs and beyond.

---

## 🙏 Acknowledgments

Learning resources that guided me:  
- [Sentdex Deep Learning Series](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)  
- [Vizuara Neural Network Playlist](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSj6tNyn_UadmUeU3Q3oR-hu)  

Without these, I couldn’t have bridged theory to implementation.

---

✍️ Written as a reflection of my journey building **neural networks from scratch**.  
This repo is more than just code — it’s a learning milestone.  

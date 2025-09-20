# Notes on Recurrent Neural Networks (RNNs)

## Resource  
- Based on StatQuest RNN lesson.  

## Motivation  
- **Goal**: Predict stock prices.  
- Stock prices are highly variable, but the longer a company has been on the market, the more historical data is available.  
- We require a neural network that can handle **different sequence lengths** (i.e., varying amounts of input data).  
- RNNs are designed for sequence data and can process inputs of arbitrary length by maintaining a hidden state.  

## Structure of RNNs  
- RNNs are similar to standard feedforward neural networks but with a **feedback loop** that allows information to persist.  
- At each time step:  
  \[
  h_t = f(W_h h_{t-1} + W_x x_t + b)
  \]  
  \[
  y_t = g(W_y h_t + c)
  \]  
- When unrolled over time, an RNN appears as multiple copies of the same network, one per time step.  
- **Key difference from feedforward NNs**: the weights and biases are **shared across all time steps**. This weight sharing allows the model to generalize across sequences of varying lengths.  

## Advantages  
- Can handle sequential data of varying lengths.  
- Maintains a form of “memory” through hidden states, allowing the network to use information from earlier inputs to influence later outputs.  

## Training Challenges  
- **Vanishing and Exploding Gradient Problem**:  
  - Occurs during backpropagation through time (BPTT).  
  - When gradients are repeatedly multiplied across many time steps, they can either:  
    - **Explode**: If values > 1, they grow exponentially (e.g., \(2^{50}\)), leading to unstable updates.  
    - **Vanish**: If values < 1, they shrink exponentially (e.g., \(0.5^{50}\)), leading to negligible updates.  

- **Exploding gradients**: cause extremely large updates, making the training process unstable.  
- **Vanishing gradients**: cause extremely small updates, preventing the network from learning long-term dependencies.  

## Mitigation Techniques  
- **Gradient clipping**: Prevents exploding gradients by capping gradient values at a threshold.  
- **Careful weight initialization**: Helps reduce the risk of gradient explosion or vanishing at the start of training.  
- **Use of specialized architectures**: LSTMs (Long Short-Term Memory) and GRUs (Gated Recurrent Units) address vanishing gradients by introducing gating mechanisms that regulate the flow of information.  
- **Layer normalization / batch normalization**: Stabilizes training.  

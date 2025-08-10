### CODE : 
```python
import numpy as np 
from mnist import MNIST
from img_process import preprocess_image

class denseLayer: 
    def __init__(self , input_size , numberOfNeurons):
        self.weights =  .1 * np.random.randn(input_size , numberOfNeurons)
        self.biases = np.zeros((1,numberOfNeurons))
        self.velocity_weights = np.zeros_like(self.weights)
        self.velocity_biases = np.zeros_like(self.biases)


    def forward(self , input):
        self.output = np.dot(input , self.weights) + self.biases 
    def backward(self , dvalues , input):
        self.dweights = np.dot(input.T , dvalues)
        self.dbiases = np.sum(dvalues , axis=0 , keepdims=True  )
        self.dinputs = np.dot(dvalues , self.weights.T)

class reluActivation: 
    def forward(self , x ):
        self.inputs= x 
        self.output = np.maximum(0,x) 


    def backward(self , dvalues): 
        self.dinputs =  dvalues.copy()
        self.dinputs[self.inputs<=0] = 0 

class Accuracy:
    def calculate(self ,y_pred , y_true): 
        self.y_pred_class = np.argmax(y_pred , axis = 1)
        

        if(len(y_true.shape)==2):
            self.y_true_class = np.argmax(y_true , axis = 1)
        else:
            self.y_true_class = y_true


        self.correct_pred =(self.y_pred_class == self.y_true_class)
        return np.mean(self.correct_pred)  
            



class LOSS: 
    def forward(self , y_pred , y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)


        if len(y_true.shape)==1:
            correct_confidences  = y_pred_clipped[range(len(y_pred)) , y_true]
        elif len(y_true.shape)==2:
            correct_confidences = np.sum(y_pred_clipped * y_true  , axis=1)


        neg_log = -np.log(correct_confidences)
        return np.mean(neg_log)



class softmaxActivation: 
    def forward(self , x):
        exp_values = np.exp(x - np.max(x, axis = 1 , keepdims=True))
        self.output = exp_values / np.sum(exp_values , axis = 1 , keepdims=True)

class softmaxCrossEntropy: 
    def __init__(self):
        self.activation = softmaxActivation()
        self.loss = LOSS()
    def forward(self , inputs , y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.forward(self.output , y_true)

    def backward(self   , y_true):
        self.dinputs = self.output.copy() 
        if len(y_true.shape) == 1:
            self.dinputs[range(len(self.dinputs)), y_true] -= 1
        elif len(y_true.shape) == 2:
            self.dinputs -= y_true 

        self.dinputs = self.dinputs / len(self.dinputs)



class Optimizer:
    def __init__(self , learning_rate=1. , decay=0. , iterations = 0 , momentum = 0. ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = iterations
        self.momentum = momentum 


    def update(self, layer):

        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + (self.decay * self.iterations)) #change learning rate before updation
        else: 
            self.current_learning_rate = self.learning_rate
        
        
        layer.velocity_weights = self.momentum * layer.velocity_weights - self.current_learning_rate * layer.dweights
        layer.velocity_biases = self.momentum * layer.velocity_biases - self.current_learning_rate * layer.dbiases

        layer.weights += layer.velocity_weights
        layer.biases += layer.velocity_biases

        
        self.iterations+=1 


import pickle

def save_model(filename, layers):
    # layers is a list of your dense layer objects
    params = []
    for layer in layers:
        params.append({
            'weights': layer.weights,
            'biases': layer.biases
        })
    with open(filename, 'wb') as f:
        pickle.dump(params, f)

def load_model(filename, layers):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    for layer, param in zip(layers, params):
        layer.weights = param['weights']
        layer.biases = param['biases']




'''              TRAINING                    '''


#prepping of data

mndata = MNIST('mnist_data')

X_train , Y_train = mndata.load_training()
X_test, Y_test = mndata.load_testing()


X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.int32)
Y_test = np.array(Y_test, dtype=np.int32)



#Normalization of the data


X_train = X_train / 255.0
X_test = X_test/255.



##shuffling data


indices = np.arange(len(X_train))
np.random.shuffle(indices)

X_train = X_train[indices]
Y_train = Y_train[indices]


#### BATCHES GENERATION 

def get_batches(X, Y, batch_size):
    for start_idx in range(0, len(X), batch_size):
        end_idx = start_idx + batch_size
        yield X[start_idx:end_idx], Y[start_idx:end_idx]


layer1 = denseLayer(784, 256)
activation1 = reluActivation()

layer2 = denseLayer(256, 128)
activation2 = reluActivation()

layer3 = denseLayer(128, 10)

softmax_loss = softmaxCrossEntropy()
accuracy = Accuracy()
optimizer = Optimizer(learning_rate = 0.05 , decay = 1e-3, momentum=0.9)


layers = [layer1, layer2, layer3]

# Load saved weights
load_model('trained_model.pkl', layers)
print("Model loaded!")

custom_img = preprocess_image('4_digit.png')


layer1.forward(custom_img)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layer3.forward(activation2.output)
softmax_loss.activation.forward(layer3.output) 

print("Output probabilities:", softmax_loss.activation.output)
predicted_digit = np.argmax(softmax_loss.activation.output)
print(f"Predicted digit: {predicted_digit}")















'''
test_loss = 0
test_acc = 0
test_batches = 0
batch_size = 32

for X_batch, Y_batch in get_batches(X_test, Y_test, batch_size):
    layer1.forward(X_batch)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    layer3.forward(activation2.output)

    loss = softmax_loss.forward(layer3.output, Y_batch)
    acc = accuracy.calculate(softmax_loss.output, Y_batch)

    test_loss += loss
    test_acc += acc
    test_batches += 1

test_loss /= test_batches
test_acc /= test_batches

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")




'''


'''


epochs = 10 
batch_size = 32

layers = [layer1, layer2, layer3]

#### ACTUAL TESTING 
for epoch in range(epochs):
    
    
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]


    epoch_acc = 0 
    epoch_loss = 0 
    batch_count = 0 


    for X_batch, Y_batch in get_batches(X_train, Y_train, batch_size):
        batch_count += 1

        layer1.forward(X_batch)
        activation1.forward(layer1.output)

        layer2.forward(activation1.output)
        activation2.forward(layer2.output)

        layer3.forward(activation2.output)

        loss_value = softmax_loss.forward(layer3.output, Y_batch)
        softmax_loss.backward(Y_batch)  # dinputs: (batch_size, 10)

        layer3.backward(softmax_loss.dinputs, activation2.output)  # dinputs: (batch_size, 128)

        activation2.backward(layer3.dinputs)  # uses layer3.dinputs (128 neurons)

        layer2.backward(activation2.dinputs, activation1.output)  # dinputs: (batch_size, 256)

        activation1.backward(layer2.dinputs)  # uses layer2.dinputs (256 neurons)

        layer1.backward(activation1.dinputs, X_batch)  
        optimizer.update(layer1)
        optimizer.update(layer2)
        optimizer.update(layer3)

        # 4. Accumulate loss and accuracy
        epoch_loss += loss_value
        epoch_acc += accuracy.calculate(softmax_loss.output, Y_batch)
    epoch_loss /= batch_count
    epoch_acc /= batch_count

    print(f"Epoch {epoch+1} — Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")




save_model('trained_model.pkl', layers)
print("Model saved to 'trained_model.pkl'")
'''



save_model('trained_model.pkl', layers)
print("Model saved to 'trained_model.pkl'")


```

import numpy as np
from mnist import MNIST
from img_process import preprocess_image
from model import DenseLayer, ReLU, SoftmaxCrossEntropy, Accuracy, Optimizer
from utils import get_batches, save_model, load_model


mndata = MNIST('mnist_data')
X_train, Y_train = mndata.load_training()
X_test, Y_test = mndata.load_testing()

X_train = np.array(X_train, dtype=np.float32) / 255.0
X_test = np.array(X_test, dtype=np.float32) / 255.0
Y_train = np.array(Y_train, dtype=np.int32)
Y_test = np.array(Y_test, dtype=np.int32)


indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train, Y_train = X_train[indices], Y_train[indices]


layer1 = DenseLayer(784, 256)
activation1 = ReLU()
layer2 = DenseLayer(256, 128)
activation2 = ReLU()
layer3 = DenseLayer(128, 10)
softmax_loss = SoftmaxCrossEntropy()
accuracy = Accuracy()
optimizer = Optimizer(learning_rate=0.05, decay=1e-3, momentum=0.9)

layers = [layer1, layer2, layer3]


# load_model('trained_model.pkl', layers) (loading saved weights.)

epochs = 10
batch_size = 32
for epoch in range(epochs):
    epoch_loss, epoch_acc, batch_count = 0, 0, 0
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train, Y_train = X_train[indices], Y_train[indices]

    for X_batch, Y_batch in get_batches(X_train, Y_train, batch_size):
       
        layer1.forward(X_batch)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        layer3.forward(activation2.output)

        loss_value = softmax_loss.forward(layer3.output, Y_batch)

       
        softmax_loss.backward(Y_batch)
        layer3.backward(softmax_loss.dinputs, activation2.output)
        activation2.backward(layer3.dinputs)
        layer2.backward(activation2.dinputs, activation1.output)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs, X_batch)

       
        optimizer.update(layer1)
        optimizer.update(layer2)
        optimizer.update(layer3)


        epoch_loss += loss_value
        epoch_acc += accuracy.calculate(softmax_loss.output, Y_batch)
        batch_count += 1

    print(f"Epoch {epoch+1}: Loss {epoch_loss/batch_count:.4f}, Acc {epoch_acc/batch_count:.4f}")


save_model('trained_model.pkl', layers)
print("Model saved!")


custom_img = preprocess_image('4_digit.png') #4_digit.png is the custom handwritten digit I used.
layer1.forward(custom_img)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layer3.forward(activation2.output)
softmax_loss.activation.forward(layer3.output)
print("Predicted:", np.argmax(softmax_loss.activation.output))

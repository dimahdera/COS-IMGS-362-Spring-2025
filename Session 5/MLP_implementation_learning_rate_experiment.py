
import tensorflow as tf
from d2l import tensorflow as d2l
from matplotlib import pyplot as plt


#Experiment using different learning rates to find out how quickly the loss function value drops. Can
#you reduce the error by increasing the number of training epochs? Check Section 3.4 in the textbook.


#In order to show how the learning rate affects the speed of the neural network convergence (or learning),
#we train a multi-layer perception (MLP) neural network with two layers and a rectifier linear unit (ReLU)
#activation function on the Fashion MNIST dataset that presents a multi-classification problem with 10
#classes. We try a different learning rate every time and check training, validation losses, and validation
#accuracy. We also increased the number of epochs from 25 to 100 to check how that affects the training and
#validation losses and accuracy.

class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(
            tf.random.normal((num_inputs, num_hiddens)) * sigma)
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(
            tf.random.normal((num_hiddens, num_outputs)) * sigma)
        self.b2 = tf.Variable(tf.zeros(num_outputs))
        
def relu(X):
    return tf.math.maximum(X, 0)
    
    
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = tf.reshape(X, (-1, self.num_inputs))
    H = relu(tf.matmul(X, self.W1) + self.b1)
    return tf.matmul(H, self.W2) + self.b2
    
#Training with 25 epochs and lr=0.01
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.01)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=25)
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/loss_lr0p01_ep25.png")

#Training with 100 epochs and lr=0.01
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.01)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=100)
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/loss_lr0p01_ep100.png")

#Training with 100 epochs and lr=0.1
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=100)
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/loss_lr0p1_ep100.png")

#Training with 100 epochs and lr=0.2
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.2)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=100)
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/loss_lr0p2_ep100.png")

#Training with 100 epochs and lr=1
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=100)
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/loss_lr1_ep100.png")

#Concise Implementation
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_outputs)])
            
#Training with 100 epochs and lr=0.01
model = MLP(num_outputs=10, num_hiddens=256, lr=0.01)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=100)
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/Concise_Implementation_loss_lr0p01_ep100.png")


#Training with 100 epochs and lr=0.1
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer = d2l.Trainer(max_epochs=100)
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/Concise_Implementation_loss_lr0p1_ep100.png")


#Training with 100 epochs and lr=0.2
model = MLP(num_outputs=10, num_hiddens=256, lr=0.2)
trainer = d2l.Trainer(max_epochs=100)
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/Concise_Implementation_loss_lr0p2_ep100.png")


#Training with 100 epochs and lr=1
model = MLP(num_outputs=10, num_hiddens=256, lr=1)
trainer = d2l.Trainer(max_epochs=100)
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/Concise_Implementation_loss_lr1_ep100.png")


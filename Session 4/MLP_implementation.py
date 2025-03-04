
import tensorflow as tf
from d2l import tensorflow as d2l
from matplotlib import pyplot as plt


##What would happen if we initialized the parameters/weights to zero? Would the algorithm still work? What
#if we initialized the parameters/weights with a normal distribution with a variance of 1000 rather than 0.01? Please show
#examples that support your answers and plot the graphs of
#the training and validation losses versus epoch for every initialization. Check Sections 3.4, 5.2 and 5.4 in the textbook.
         
#we train a multi-layer perception (MLP) neural network
#with two layers and a rectifier linear unit (ReLU) activation function on the Fashion MNIST dataset that
#presents a multi-classification problem with 10 classes. We try a different initialization scheme every time
#and check training, validation losses, and validation accuracy

             
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


class MLPScratch_constant_initialization(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(initial_value=tf.constant(1., dtype=tf.float32, shape=(num_inputs, num_hiddens)), trainable=True)
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(initial_value=tf.constant(1., dtype=tf.float32, shape=(num_hiddens, num_outputs)), trainable=True)
        self.b2 = tf.Variable(tf.zeros(num_outputs))


class MLPScratch_zero_initialization(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(tf.zeros(shape=(num_inputs, num_hiddens)))
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(tf.zeros(shape=(num_hiddens, num_outputs)))
        self.b2 = tf.Variable(tf.zeros(num_outputs))


def relu(X):
    return tf.math.maximum(X, 0)



@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = tf.reshape(X, (-1, self.num_inputs))
    H = relu(tf.matmul(X, self.W1) + self.b1)
    return tf.matmul(H, self.W2) + self.b2

@d2l.add_to_class(MLPScratch_constant_initialization)
def forward(self, X):
    X = tf.reshape(X, (-1, self.num_inputs))
    H = relu(tf.matmul(X, self.W1) + self.b1)
    return tf.matmul(H, self.W2) + self.b2

@d2l.add_to_class(MLPScratch_zero_initialization)
def forward(self, X):
    X = tf.reshape(X, (-1, self.num_inputs))
    H = relu(tf.matmul(X, self.W1) + self.b1)
    return tf.matmul(H, self.W2) + self.b2



model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)#d2l.
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/loss.png")


#Initializing the parameters/weights with a normal distribution with a variance of 1000

class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=1000):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(
            tf.random.normal((num_inputs, num_hiddens)) * sigma)
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(
            tf.random.normal((num_hiddens, num_outputs)) * sigma)
        self.b2 = tf.Variable(tf.zeros(num_outputs))

@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = tf.reshape(X, (-1, self.num_inputs))
    H = relu(tf.matmul(X, self.W1) + self.b1)
    return tf.matmul(H, self.W2) + self.b2

model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/loss_1000_initialization.png")

#MLP with Zero Initialization for the Fashion MNIST dataset
model = MLPScratch_zero_initialization(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.001)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/loss_zero_initialization.png")


#MLP with constant Initialization for the Fashion MNIST dataset
model = MLPScratch_constant_initialization(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.001)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/loss_constant_initialization.png")


#Concise Implementation

class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_outputs)])
            
#Training
#[The training loop] is exactly the same as when we implemented softmax regression. This modularity enables us to separate matters concerning the model architecture from orthogonal considerations.      


model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)     
plt.savefig("/home/dxdcis/IMGS-389/Concise_Implement_loss.png")

#MLP Concise Implementation with Zero Initialization
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, kernel_initializer='zeros',
    bias_initializer='zeros', activation='relu'),
            tf.keras.layers.Dense(num_outputs, kernel_initializer='zeros',
    bias_initializer='zeros')])
    
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/Concise_Implement_loss_Zero_Initialization.png")   

#MLP Concise Implementation with constant Initialization


class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, kernel_initializer=tf.keras.initializers.Constant(1.),
    bias_initializer='zeros', activation='relu'),
            tf.keras.layers.Dense(num_outputs, kernel_initializer=tf.keras.initializers.Constant(1.),
    bias_initializer='zeros')])
    
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
plt.savefig("/home/dxdcis/IMGS-389/Concise_Implement_loss_constant_Initialization.png") 
    


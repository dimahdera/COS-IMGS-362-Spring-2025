
import tensorflow as tf
from d2l import tensorflow as d2l
from matplotlib import pyplot as plt

#Experiment with the value of the weight decay λ in the estimation problem in Section 3.7. Plot
#training and validation loss per epoch for different values of the weight decay λ. You can experiment with
#three situations when λ is low, high and in-between. What do you observe?


#To show how the weight decay λ affects the estimation problem in Section 3.7, we change the value of
#the weight decay and plot the training and validation losses for each case. The synthetic data are generated
#where the label is given by an underlying linear function of the inputs, corrupted by Gaussian noise with
#zero mean and standard deviation of 0.01. To make the effects of overfitting pronounced, we intentionally
#increase the dimensionality of the problem to 200 and train a linear regression model with a small training
#set with only 20 examples. 

#igh-Dimensional Linear Regression
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = tf.random.normal((n, num_inputs))
        noise = tf.random.normal((n, 1)) * 0.01
        w, b = tf.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = tf.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)


def l2_penalty(w):
    return tf.reduce_sum(w**2) / 2

#Defining the Model    
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()

    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))

data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    plt.savefig("/home/dxdcis/IMGS-389/loss_weight_decay_{}.png".format(lambd))
    print('L2 norm of w:', float(l2_penalty(model.w)))
    print('\n The weight decay Lambda is :', float(lambd))
    
    
    

train_scratch(0)

train_scratch(0.01)

train_scratch(0.1)

train_scratch(3)


train_scratch(10)

train_scratch(20)

#Concise Implementation

class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(wd),
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)
        )

    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.net.losses
        

def train_concise(lambd):
    model = WeightDecay(wd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    plt.savefig("/home/dxdcis/IMGS-389/concise_loss_weight_decay_{}.png".format(lambd))
    print('L2 norm of w:', float(l2_penalty(model.wd)))
    print('\n The weight decay Lambda is :', float(lambd))
    
    
    
train_concise(0)
train_concise(0.01)
train_concise(0.1)
train_concise(10)
train_concise(20)


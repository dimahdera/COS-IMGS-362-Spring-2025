

!pip install tensorflow==2.12.0 tensorflow-probability==0.20.0

!pip install d2l==1.0.3



import tensorflow as tf
from d2l import tensorflow as d2l


class GRUScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        init_weight = lambda *shape: tf.Variable(tf.random.normal(shape) * sigma)
        triple = lambda: (init_weight(num_inputs, num_hiddens),
                          init_weight(num_hiddens, num_hiddens),
                          tf.Variable(tf.zeros(num_hiddens)))

        self.W_xz, self.W_hz, self.b_z = triple()  # Update gate
        self.W_xr, self.W_hr, self.b_r = triple()  # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple()  # Candidate hidden state

"""### Defining the Model

Now we are ready to [**define the GRU forward computation**].
Its structure is the same as that of the basic RNN cell,
except that the update equations are more complex.

"""

@d2l.add_to_class(GRUScratch)
def forward(self, inputs, H=None):
    if H is None:
        # Initial state with shape: (batch_size, num_hiddens)
        H = tf.zeros((inputs.shape[1], self.num_hiddens))
    outputs = []
    for X in inputs:
        Z = tf.sigmoid(tf.matmul(X, self.W_xz) +
                        tf.matmul(H, self.W_hz) + self.b_z)
        R = tf.sigmoid(tf.matmul(X, self.W_xr) +
                        tf.matmul(H, self.W_hr) + self.b_r)
        H_tilde = tf.tanh(tf.matmul(X, self.W_xh) +
                           tf.matmul(R * H, self.W_hh) + self.b_h)
        H = Z * H + (1 - Z) * H_tilde
        outputs.append(H)
    return outputs, H

"""### Training

[**Training**] a language model on *The Time Machine* dataset
works in exactly the same manner as in :numref:`sec_rnn-scratch`.

"""

data = d2l.TimeMachine(batch_size=1024, num_steps=32)
with d2l.try_gpu():
    gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)

"""## [**Concise Implementation**]

In high-level APIs, we can directly instantiate a GRU model.
This encapsulates all the configuration detail that we made explicit above.

"""

class GRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = tf.keras.layers.GRU(num_hiddens, return_sequences=True,
                                       return_state=True)

"""The code is significantly faster in training as it uses compiled operators
rather than Python.

"""

gru = GRU(num_inputs=len(data.vocab), num_hiddens=32)
with d2l.try_gpu():
    model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)

"""After training, we print out the perplexity on the training set
and the predicted sequence following the provided prefix.

Perplexity (PPL) is the loss function to measures the language model quality

"""

model.predict('it has', 20, data.vocab)
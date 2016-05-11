import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

# Parameters
learning_rate = 0.0001
training_epochs = 10000
display_step = 100

# Network Parameters
# the size of the hidden state for the lstm (notice the lstm uses 2x of this amount so actually lstm will have state of size 2)
size = 200
# 2 different sequences total
batch_size = 1
# the maximum steps for both sequences is 200
n_steps = 20
# each element of the sequence has dimension of 13
seq_width = 5

initializer = tf.random_uniform_initializer(-1, 1)

# the sequences, has n steps of maximum size
seq_input = tf.placeholder(tf.float32, [n_steps, batch_size, seq_width])
# what timesteps we want to stop at, notice it's different for each batch hence dimension of [batch]
# early_stop = tf.placeholder(tf.int32, [batch_size])

# inputs for rnn needs to be a list, each item being a timestep.
# we need to split our input into each timestep, and reshape it because split keeps dims by default
# encoder_inputs = [tf.reshape(i, (batch_size, seq_width)) for i in tf.split(0, n_steps, seq_input)]
encoder_inputs = [tf.reshape(seq_input, [-1, seq_width])]
decoder_inputs = ([tf.zeros_like(encoder_inputs[0], name="GO")] + encoder_inputs[:-1])
targets = encoder_inputs
weights = [tf.ones_like(targets_t, dtype=tf.float32) for targets_t in targets]
# targets = [inputs[i + 1] for i in xrange(len(inputs) - 1)]

cell = tf.nn.rnn_cell.BasicLSTMCell(size)

_, enc_state = tf.nn.rnn(cell, encoder_inputs, dtype=tf.float32)
cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, seq_width)
dec_outputs, dec_state = tf.nn.seq2seq.rnn_decoder(decoder_inputs, enc_state, cell)

# Construct model
# e_stop = np.array([1, 1])

# y_pred = [tf.reshape(dec_output, [-1]) for dec_output in dec_outputs]
y_true = [tf.reshape(encoder_input, [-1]) for encoder_input in encoder_inputs]
y_pred = [tf.reshape(dec_output, [-1]) for dec_output in dec_outputs]
# y_pred = dec_outputs
# y_true = encoder_inputs
print y_pred
print y_true
# Define loss and optimizer, minimize the squared error
# loss = tf.nn.seq2seq.sequence_loss(y_pred, y_true, weights)
# optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
loss = 0
for i in range(len(y_true)):
    loss += tf.reduce_sum(tf.square(tf.sub(y_pred[i], y_true[i])))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        rand = np.random.rand(n_steps, batch_size, seq_width).astype('float32')
        x = np.arange(n_steps * batch_size * seq_width)
        x = x.reshape((n_steps, batch_size, seq_width))
        feed = {seq_input: x}
        # Fit training using batch data
        _, cost_value = sess.run([optimizer, loss], feed_dict=feed)
         # Display logs per epoch step
        if epoch % display_step == 0:       
            # print sess.run(decoder_inputs, feed_dict=feed)
            print "logits"
            a = sess.run(y_pred, feed_dict=feed)
            print a
            print "labels"
            b = sess.run(y_true, feed_dict=feed)
            print b
            c = sess.run(loss, feed_dict=feed)
            print "loss: ", c

            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(cost_value))

    print("Optimization Finished!")
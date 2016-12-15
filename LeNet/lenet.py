"""
LeNet Architecture

HINTS for layers:

    Convolutional layers:

    tf.nn.conv2d
    tf.nn.max_pool

    For preparing the convolutional layer output for the
    fully connected layers.

    tf.contrib.flatten
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten


EPOCHS = 100
BATCH_SIZE = 50
n_classes = 10  # MNIST total classes (0-9 digits)

# LeNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
#
# Don't worry about anything else in the file too much, all you have to do is
# create the LeNet and return the result of the last fully connected layer.
def conv2d(x, W, b, strides=1, k=2):
    """
    Conv2D wrapper, with bias, relu activation, and max pooling
    """
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    return tf.nn.max_pool(x, [1, k, k, 1], [1, k, k,1], padding='SAME')

def LeNet(x):
    """
    LeNet implementation
    """
    # Reshape from 2D to 4D. This prepares the data for
    # convolutional and pooling layers.
    x = tf.reshape(x, (-1, 28, 28, 1))
    # Pad 0s to 32x32. Centers the digit further.
    # Add 2 rows/columns on each side for height and width dimensions.
    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")
    # Define depth of each hidden layer
    layer_depth = {
      'layer_1': 6,
      'layer_2': 16,
      'fully_connected': 120
    }
    # Initialized weights to small random numbers, 0 mean, 0.01 stddev
    weights = {
      'layer_1': tf.Variable(tf.truncated_normal(
          [5, 5, 1, layer_depth['layer_1']], stddev=0.01)),
      'layer_2': tf.Variable(tf.truncated_normal(
          [5, 5, layer_depth['layer_1'], layer_depth['layer_2']], stddev=0.01)),
      'fully_connected': tf.Variable(tf.truncated_normal(
          [5*5*16, layer_depth['fully_connected']], stddev=0.01)),
      'out': tf.Variable(tf.truncated_normal(
          [layer_depth['fully_connected'], n_classes], stddev=0.01))
    }
    # Initialize biases to zero
    biases = {
      'layer_1': tf.Variable(tf.zeros(layer_depth['layer_1'])),
      'layer_2': tf.Variable(tf.zeros(layer_depth['layer_2'])),
      'fully_connected': tf.Variable(tf.zeros(layer_depth['fully_connected'])),
      'out': tf.Variable(tf.zeros(n_classes))
    }
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'])
    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])
    flat = tf.contrib.layers.flatten(conv2)
    fc1 = tf.add(tf.matmul(flat, weights['fully_connected']), biases['fully_connected'])
    fc1 = tf.nn.tanh(fc1) # tanh gave better result than relu, for fc1
    # Return the result of the last fully connected layer.
    return tf.add(tf.matmul(fc1, weights['out']), biases['out'])


# MNIST consists of 28x28x1, grayscale images
x = tf.placeholder(tf.float32, (None, 784))
# Classify over 10 digits 0-9
y = tf.placeholder(tf.float32, (None, 10))
fc2 = LeNet(x)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def eval_data(dataset):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    # If dataset.num_examples is not divisible by BATCH_SIZE
    # the remainder will be discarded.
    # Ex: If BATCH_SIZE is 64 and training set has 55000 examples
    # steps_per_epoch = 55000 // 64 = 859
    # num_examples = 859 * 64 = 54976
    #
    # So in that case we go over 54976 examples instead of 55000.
    steps_per_epoch = dataset.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    for step in range(steps_per_epoch):
        batch_x, batch_y = dataset.next_batch(BATCH_SIZE)
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss/num_examples, total_acc/num_examples


if __name__ == '__main__':
    # Load data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps_per_epoch = mnist.train.num_examples // BATCH_SIZE
        num_examples = steps_per_epoch * BATCH_SIZE

        # Train model
        for i in range(EPOCHS):
            for step in range(steps_per_epoch):
                batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

            val_loss, val_acc = eval_data(mnist.validation)
            print("EPOCH {} ...".format(i+1))
            print("Validation loss = {:.3f}".format(val_loss))
            print("Validation accuracy = {:.3f}".format(val_acc))
            print()

        # Evaluate on the test data
        test_loss, test_acc = eval_data(mnist.test)
        print("Test loss = {:.3f}".format(test_loss))
        print("Test accuracy = {:.3f}".format(test_acc))



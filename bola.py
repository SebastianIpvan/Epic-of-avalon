import tensorflow as tf
import h5py
import numpy as np

#load data 
full_data = None
full_label = None

with h5py.File('/mnt/disk1/bole.h5', 'r') as hf:
    full_data = np.array(hf.get('data'))
    full_label = np.array(hf.get('label'))

#data
data = full_data[:40]
label = full_label[:40]
test_data = full_data[40:]
test_label = full_label[40:]

#Hyperparameters
learning_rate = 0.0001  #if loss is nan try to lower this shit
step_size = 20000
batch_size = 10
display_progress_step = 1

#network parameters
input_layer = 4
hidden_layer_1 = 12
hidden_layer_2 = 6
output_layer = 3

def next_batch(num, data, labels):
    '''
    Return a total SUPER RESOLTUIOof `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# A TensorFlow graph consists of the following parts which will be detailed below:

# Placeholder variables used to feed input into the graph.
# Model variables that are going to be optimized so as to make the model perform better.
# The model which is essentially just a mathematical function that calculates some output given the input in the placeholder variables and the model variables.
# A cost measure that can be used to guide the optimization of the variables.
# An optimization method which updates the variables of the model.

#tensorflow input and output placeholder (dtype args = [Num of data, Dimension of layer])
#none mean the tensor can hold any number of image

X = tf.placeholder("float", [None, input_layer]) 
Y = tf.placeholder("float", [None, output_layer])

#Define Weight and bias in tensorflow variable by setting the initial value

#weight shape are in matrix fromat ([x,y])
weights = {
    1: tf.Variable(tf.random_normal([input_layer, hidden_layer_1])),
    2: tf.Variable(tf.random_normal([hidden_layer_1, hidden_layer_2])),
    3: tf.Variable(tf.random_normal([hidden_layer_2, output_layer]))
}

#biases shape are in vector format ([1,x] or [1])
biases = {
    1 : tf.Variable(tf.random_normal([hidden_layer_1])),
    2 : tf.Variable(tf.random_normal([hidden_layer_2])),
    3 : tf.Variable(tf.random_normal([output_layer]))
}

#model blueprint in forward propagation way (x.w + b)
def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights[1]), biases[1])
    layer_2 = tf.add(tf.matmul(layer_1, weights[2]), biases[2])
    return tf.matmul(layer_2, weights[3]) + biases[3]

#build model 
logits = neural_net(X)
prediction = tf.reduce_mean(tf.nn.softmax(logits), axis=0)

#define loss and training optimizer
#cross entropy loss is used form classification problem where 
#the loss become 0 when the predicted output is the same as the true output
#since the output of cross entory are bunch of tensors , we need to average them to get a scalar value
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
training_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=loss_function)

#evaluate the model (correct prediction and accuracy)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1)) #using argmax with axis 1 becuase the output is bunch of one hot ecoded rows in the matrix
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # tf.cast convert bunch of [True , False, True, ....] into [1., 0., 1.,] AKA boolean to float element wise
#intitialize the value of tf.Variable
init = tf.global_variables_initializer()

#start training
# what we do in this step
# 1. run a session of initializer to give initial value to variable
# 2. for each step run a session of training optimizer (doing backpropagation)
# 3. Also run a session to display the loss and accuracy for each display step
# 4. Calculate the test accuracy by running the model on the test dataset

"""
step is the number of times we update the gradient which mean 
the number of weight and bias update
"""
with tf.Session() as session:

    session.run(init) #execute the initializer

    #start training
    for step in range(1, step_size+1):
        x_batch, y_batch =  next_batch(40, data, label) #take the input and true output data from the train batch
        train_feed_dict = {X: x_batch, Y:y_batch} #map the placeholder values to the batch values
        session.run(training_optimizer, feed_dict=train_feed_dict) #backpropagation

        if step % display_progress_step == 0 or step == 1: #display progress if step equal 1 or step is the multiply of step_size
            train_loss, train_acc = session.run([loss_function, accuracy], feed_dict=train_feed_dict) #caulculate loss and accuracy of training
            print("Step " + str(step) + ", Minibatch Loss = {:.4f}, ".format(train_loss) + "Training accuracy = {:.3f}".format(train_acc)) # print result

    #start testing
    test_feed_dict = {X: test_data, Y: test_label}
    result = session.run(prediction, feed_dict=test_feed_dict) * 100
    print("P(liverpool): {}, P(Man United): {}, P(Draw): {}".format(result[0], result[1], result[2]) )







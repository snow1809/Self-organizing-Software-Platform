import tensorflow as tf
##################################################
### CONFIGURATIONS
##################################################

##################################################
### CONSTANTS
##################################################
n_inputs  = 28*28
n_hidden1 = 300			# neurons in hidden layer
n_hidden2 = 100
n_outputs = 10

##################################################
### DEFINE GRAPH
##################################################
# 1. Train & target placeholder
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
training = tf.placeholder_with_default(False, shape=(), name="training")
y = tf.placeholder(tf.int64, shape=(None), name="y")

# 2. Create Layers
# 2.1. Custom way to make layers
# def neuron_layer(X, n_neurons, name, activation=None):
# 	with tf.name_scope(name):
# 		n_inputs = int(X.get_shape()[1])
# 		stddev = 2/np.sqrt(n_inputs+n_neurons)
# 		init = tf.truncated_normal((n_inputs, n_neurons), stddev=dtddev)
# 		W = tf.Variable(init, name="kernel")
# 		b = tf.Variable(tf.zeros([n_neurons]), name=bias)
# 		Z = tf.matmiul(X, W)+b
# 		if activation is not None:
# 			return activation(Z)
# 		else:
# 			return Z;

# with tf.name_scope("dnn"):
# 	hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf.nn.relu);
# 	hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu);
# 	logits  = neuron_layer(hidden2, n_outputs, name="outputs");

# 2.2. Pre-defined & standard way to make layers - also, fully connected
with tf.name_scope("dnn"):
	# 2.2.1. in case, standard distribution
	hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu);
	logits  = tf.layers.dense(hidden2, n_outputs, name="outputs");
	# 2.2.2. in case, normal distribution

	# standard distrbiution
	hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu);
	# standard distrbiution
	tf.contrib.layers.variance_scaling_initializer()
	hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu,
			kernel_initializer=he_init, name="hidden1");


# 3. Cost function
with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
	loss = tf.reduce_mean(xentropy, name="loss")

# 4. Train to minimize the cost function
learning_rate = 0.01
with tf.name_scope("train"):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	training_op = optimizer.minimize(loss)

# 5. Evaluate using accuracy
with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 6. Generally, initializer & trained model saver are used to save in the disk
init = tf.global_variables_initializer()
saver = tf.train.Saver()

##################################################
### LEARNING
##################################################
# train data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

# running size
n_epochs = 10
batch_size = 50

# Train started
with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		for iteration in range(mnist.train.num_examples // batch_size):
			X_batch, y_batch = mnist.train.next_batch(batch_size)
			sess.run(training_op, feed_dict={X: X_batch, y:y_batch})
		acc_train = accuracy.eval(feed_dict={X: X_batch, y:y_batch})
		acc_val = accuracy.eval(feed_dict={X: mnist.validation.images, 
				y: mnist.validation.labels})
		print(epoch, "Train accuracy:", acc_train, "Validation accuracy:", acc_val)

	for op in (X, y, accuracy, training_op, logits):
		tf.add_to_collection("my_important_ops", op)
	save_path = saver.save(sess, "./my_model_final.ckpt");


# for op in tf.get_default_graph().get_operations():
# 	print(op.name)

# X, y, accuracy, training_op, logits = tf.get_collection("my_important_ops")
# print(accuracy)

# using neural networks
# with tf.Session() as sess:
# 	saver.restore(sess, "./my_model_final.ckpt")
# 	X_new_scaled = [...]
# 	Z = logits.eval(feed_dict={X: X_new_scaled})
# 	y_pred = np.argmax(Z, axis=1)




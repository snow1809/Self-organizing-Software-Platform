import numpy as np
import tensorflow as tf

##################################################
### DEFINE GRAPH
##################################################
# [LOAD existing model if exists]
saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")

# 1. Train & target placeholder
X = tf.get_default_graph().get_tensor_by_name("X:0");
y = tf.get_default_graph().get_tensor_by_name("y:0");
accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0");
training_op = tf.get_default_graph().get_tensor_by_name("GradientDescent");

# can check every names
for op in tf.get_default_graph().get_operations():
	print(op.name)

# How to save for others
for op in (X, y, accuracy, training_op):
	tf.add_to_collection("my_important_ops", op)

# How to load for others
X, y, accuracy, training_op, logits = tf.get_collection("my_important_ops")
print(X);
print(y);
print(accuracy);
print(training_op);
print(logits);

train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden[34]|outputs")
training_op = optimizeer.minimize(loss, var_list=train_vars)


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

# running size
# n_epochs = 40
# batch_size = 50

# with tf.Session() as sess:
# 	init.run()
# 	saver.restore(sess, "./my_model_final.ckpt")
	# train
	# for epoch in range(n_epochs):
	# 	for iteration in range(mnist.train.num_examples // batch_size):
	# 		X_batch, y_batch = mnist.train.next_batch(batch_size)
	# 		sess.run(training_op, feed_dict={X: X_batch, y:y_batch})
	# 	acc_train = accuracy.eval(feed_dict={X: X_batch, y:y_batch})
	# 	acc_val = accuracy.eval(feed_dict={X: mnist.validation.images, 
	# 			y: mnist.validation.labels})
	# 	print(epoch, "Train accuracy:", acc_train, "Validation accuracy:", acc_val)
	# save_path = saver.save(sess, "./my_new_model_final.ckpt")

# using neural networks
with tf.Session() as sess:
	saver.restore(sess, "./my_model_final.ckpt")
	X_new_scaled = 1/255
	Z = logits.eval(feed_dict={X: mnist.validation.images})
	y_pred = np.argmax(Z, axis=1)
	print(y_pred)


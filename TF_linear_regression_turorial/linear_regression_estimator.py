import numpy as np
import tensorflow as tf

# declare list of features (one real-valued feature here)
def model_fn(features, labels, mode):
    # build linear model and predict values
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W*features['x'] + b

    # loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))

    # training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    # EstimatorSpec connects our subgraphs to appropriate functionality.
    return tf.estimator.EstimatorSpec(mode=mode, predictions=y, loss=loss, train_op=train)

# PASS OUR CUSTOM MODEL IN
estimator = tf.estimator.Estimator(model_fn=model_fn)
# below is the same as just estimator:
# define data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# invoke 1000 training steps by invoking the method and passing the training data set.
estimator.train(input_fn=input_fn, steps=1000)

# evaluate the model
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)

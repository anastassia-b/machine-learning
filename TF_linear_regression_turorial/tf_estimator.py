# tf.estimator does: running training loops, running evaluation loops, and managing data sets.

import numpy as np
import tensorflow as tf

# declare list of features - here just one numeric feature
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# an estimator is the front end to invoke training (fitting) and evaluation (inference).
# this estimator does linear regression
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# we have two data sets, one for training and one for evaluation.
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

# our eval data has a higher loss, but its close to zero, so we are learning properly.

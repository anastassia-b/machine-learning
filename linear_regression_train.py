# The simplest optimizer is gradient descent. It modifies each variable
# according to the magnitude of the derivative of loss with respect to that variable.
import tensorflow as tf

# model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss : sum of the squares
loss = tf.reduce_sum(tf.square(linear_model - y))

# optimizer: gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init) #reset values
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

#evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

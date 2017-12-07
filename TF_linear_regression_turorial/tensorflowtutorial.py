# from __future__ import print_function
import tensorflow as tf

#building the computational graph
#running the computational graph

#A computational graph is a series of TensorFlow operations arranged into a graph of nodes.
#Each node takes zero or more tensors as inputs and produces a tensor as an output.
#One type of node is a constant (takes no inputs, outputs a value stored internally).

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

# => Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
# To actually evaluate the nodes, we need a Session.

sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# A graph can be parameteriezd to accept external inputs, known as placeholders.
# A placeholder is a promise to provide a value later.

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b
# + is a shortcut for tf.add(a, b)

#can feed concrete values into the placeholders:
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# 7.5
# [3. 7.]

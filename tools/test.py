import tensorflow as tf
import numpy as np

r = tf.placeholder(tf.float32, [2, 2, 3])
# r = tf.reshape(r, [-1])
# r = tf.tile(r, [])
# r = tf.reshape(r, [121, -1])
# r = tf.transpose(r)
# r = tf.reshape(r, [-1, 11, 11, 1])

# new code
# s = tf.expand_dims(tf.expand_dims(r, axis=1), axis=2)
# y = tf.tile(s, [1, 2, 2])

s = tf.reshape(r, [-1])
t = tf.reshape(s, [-1, 3, 1])
u = tf.reshape(t, [-1, 2, 3])


with tf.Session() as sess:
	x = np.array([[[[143], [231], [422]], [[543], [764], [908]]], [[[224], [657], [754]], [[432], [897], [549]]]]).squeeze()
	y = sess.run(u, feed_dict={r: x})
	print(x)
	print(y)
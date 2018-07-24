from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def compute_accuracy(v_x, v_y):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x: v_x, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x: v_x, y: v_y, keep_prob: 1})
    return result


def weight_variable(shape):
    # tf.truncated_normal(shape, mean,stddev): shape表示生成张量的维度，mean是均值，stddev是标准差。
    # 这个函数产生正太分布，均值和标准差自己设定。这是一个截断的产生正太分布的函数，就是说产生正态分布
    # 的值如果与均值的差值大于两倍的标准差，那就重新生成。和一般的正太分布的产生随机数据比起来，这个函
    # 数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
    # 除去name参数用以指定该操作的name，与方法有关的一共五个参数：
    # 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]
    #     这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个
    #     4维的Tensor，要求类型为float32和float64其中之一
    # 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]
    #     这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，
    #       有一个地方需要注意，第三维in_channels，就是参数input的第四维
    # 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
    # 第四个参数padding：string类型的量，只能是 "SAME", "VALID" 其中之一，这个值决定了不同的卷积方式
    # 第五个参数：use_cudnn_on_gpu: bool类型，是否使用cudnn加速，默认为true，结果返回一个Tensor，
    # 这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。

    # strides=[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')		# 这里的conv2d()与keras的conv2D()函数不太一致，
																		# 如 keras的Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(inputs_dim)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# load mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
# reshape(data you want to reshape, [-1, reshape_height, reshape_weight, imagine layers]) image layers=1 when the imagine is in white and black, =3 when the imagine is RGB
x_image = tf.reshape(x, [-1, 28, 28, 1])
# 上面reshape里面等于-1的话，那么Numpy会根据其余的维度计算出这个维度值。这里需要根据实际的None来确定。
# 例如x为100张图片，[None, 784]=[100,74],  那么100*784 /(28*28*1)=100。即[100,28,28,1]

# ********************** conv1 *********************************
# transfer a 5*5*1 imagine into 32 sequence
W_conv1 = weight_variable([5, 5, 1, 32])    # [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
b_conv1 = bias_variable([32])
# input a imagine and make a 5*5*1 to 32 with stride=1*1, and activate with relu
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28*28*32
#   x_image维度为[None,28,28,1]，W_conv1维度为[5, 5, 1, 32]，因为padding='SAME'，
#   所以conv2d输出size为28*28*32，应该是None*28*28*32吧，这个None是输入图片的数量，对于某一张图片的输出是28*28*32。

h_pool1 = max_pool_2x2(h_conv1)  # output size 14*14*32   最大值池化操作，图片长宽尺寸减半

# ********************** conv2 *********************************
# transfer a 5*5*32 imagine into 64 sequence
W_conv2 = weight_variable([5, 5, 32, 64])   # [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
b_conv2 = bias_variable([64])
# input a imagine and make a 5*5*32 to 64 with stride=1*1, and activate with relu
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7*7*64

# ********************* func1 layer *********************************
W_fc1 = weight_variable([7 * 7 * 64, 1024])  #为什么是这样的维度？把64个通道的图片拉伸为一条并变成1024个通道吗？
# 上面的w_fc1是权重，维度为[7 * 7 * 64, 1024]

b_fc1 = bias_variable([1024])

# reshape the image from 7,7,64 into a flat (7*7*64)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# h_fc1的维度为 [-1, 7 * 7 * 64]*[7 * 7 * 64, 1024] = [-1,1024]， 即 [图片张数，1024]

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# ********************* func2 layer *********************************
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# [图片张数，1024]*[1024, 10]= [图片张数，10]

# calculate the loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
# use Gradientdescentoptimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# init session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))



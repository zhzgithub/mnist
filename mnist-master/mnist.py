from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def compute_accuracy(v_x, v_y):
    global prediction
    # input v_x to nn and get the result with y_pre
    y_pre = sess.run(prediction, feed_dict={x: v_x})
    # find how many right
    #    tf.argmax()用法：tf.argmax(input, dimension, name=None)
    #           dimension = 0 按列找，找出每列的最大值，返回最大数值的下标
    #           dimension = 1 按行找，找出每行的最大值，返回最大数值的下标
    #           tf.argmax()  返回最大数值的下标
    #           通常和tf.equal()一起使用，计算模型准确度
    #   tf.equal(A, B)用法：对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反之返
    #           回False， 返回的值的矩阵维度和A是一样的，如[1,0,0,1,1]返回[ True False False  True  True]
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_y, 1))
    # 分析后面程序可知y_pre是n行10列，按行找y_pre的每行(即每个样本)的10个类别的最大值（因为是
    # softmax,所以找最大值的索引，表示预测的类别）的索引，然后和标签v_y对比最大值索引是否一致。

    # calculate average
    # tf.cast(x, dtype, name=None) 将x的数据格式转化成dtype.例如，原来x的数据格式是bool，
    # 那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以
    # 由于把TRUE和FALSE都变为了0和1，所以用tf.reduce_mean()函数来求取平均值就正好是正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # get input content
    result = sess.run(accuracy, feed_dict={x: v_x, y: v_y})
    return result


def add_layer(inputs, in_size, out_size, activation_function=None, ):
    # init w: a matric in x*y
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # init b: a matric in 1*y
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    # calculate the result
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # add the active hanshu
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )	# 1行10列
    return outputs


# load mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# define placeholder for input
x = tf.placeholder(tf.float32, [None, 784])  # 每行是一个样本，每个样本有784个元素，
                                             # None表示任意行，对应后面设置的100，即batch_size,
y = tf.placeholder(tf.float32, [None, 10])
# add layer
prediction = add_layer(x, 784, 10, activation_function=tf.nn.softmax)   # x是输入样本，784*10是权重矩阵的维度
# calculate the loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
# 先对-y * tf.log(prediction)每行的元素求和，n行10列*n行10列,结果还是n行10列，然后再对这n行求平均值，但是y是oneho形式，
# 如y * tf.log(prediction)=[0000100000]*log([0.1 0.3 0.04,...])，这样怎么求解？
# 我测试过，矩阵元素乘法就是*星号，是矩阵里面对应元素相乘。即hadamard乘积
# 如np.array([1,2,3])*np.array([4,5,6])=array([4,10,18])
#       reduction_indices:表示在哪一维上求解，reduction_indices=[1]表示对每一行的元素求和
#       reduction_indices=[1]不是tf.reduce_mean()的参数，而是tf.reduce_sum()的参数，表示对
#       y * tf.log(prediction)求每行的元素的和，因为y和prediction是n行10列，所以求每行的和之后变成了n行1列，
#       此即每个样本的损失函数。然后tf.reduce_mean再对这n行1列求平均值，即n个样本的损失函数的平均值
#   由于标签y每行只有一个元素为1，因此，y * tf.log(prediction)的结果每行只有一个元素非0，其余为0，sum求
#   每行的和，即变为每行只有一个元素了，即n行1列。



# use Gradientdescentoptimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# init session
sess = tf.Session()
# init all variables
sess.run(tf.global_variables_initializer())

# start training
# 对for循环的解释：每次循环都对batch_x, batch_y 取一组100个新样本，然后对这100个样本进行训练，更新权重偏置
# 照这么说的话，那就是循环1000次就有1000*100=100000个即十万个样本，明显不对。
# 应该这么解释，next_batch循环取，取完了60000张训练样本再继续从头取一遍，对，就是这样。
for i in range(1000):       # 这里的for循环不是全部样本循环多少次，而是循环多少个新batch，
                            # 新batch没了再继续重头取batch，直到for循环结束。
    # get batch to learn easily
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
    # 上面这一步是进行训练操作，每次把100个训练样本代入进去训练，进行优化，内部得到每次更新的权重矩阵。
    # 当进行下一次for循环时候，就对新的100个样本更新权重矩阵。
    # 我的问题：最后怎样获取全部训练完成的得到权重矩阵呢？因为权重矩阵定义在add_layer中，那该怎样获取出来呢？
    # 怎样获取出来以供下次使用？就是读取出权重，这是模型持久化的概念了吧应该！！

    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
        # 注意这里统计正确率是用的测试集，不是训练集。
        # 并且这里统计正确率是使用的全部10000个测试样本，并没有分批次。即每训练50个batch=50*100个样本后，
        # 利用更新的权重来统计正确率。

#         在统计正确率的时候会用到权重矩阵，权重矩阵是在prediction里面，而prediction是通过add_layer()函数
#         得到， add_layer()函数定义了权重矩阵

# 但是在add_layer()函数里面，只是定义了权重weights,初始化为随机数，并没有更新。难道更新是在train_step
#  = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)这一步的时候，把权重重新赋值更新了吗？
# 好像也只有这一种解释，因为优化器优化的就是权重和偏置，但是权重和偏置都没设置成全局变量，只是add_layer()
# 函数的内部参数。
# 我还得再看看模型持久化的问题，看看怎么提取权重。看看更新权重的具体步骤。

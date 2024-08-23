print('以下是tensorflow的安装')
# 导入TensorFlow
import tensorflow as tf
# 打印TensorFlow版本，确保它已正确安装
print(f"TensorFlow version: {tf.__version__}")
# 列出可用的GPU设备
print("Available GPU devices:")
print(tf.config.list_physical_devices('GPU'))

import timeit

# 指定在cpu上运行
def cpu_run():
    with tf.device('/cpu:0'):
        cpu_a = tf.random.normal([10000, 1000])
        cpu_b = tf.random.normal([1000, 2000])
        c = tf.matmul(cpu_a, cpu_b)
    return c


# 指定在gpu上运行
def gpu_run():
    with tf.device('/gpu:0'):
        gpu_a = tf.random.normal([10000, 1000])
        gpu_b = tf.random.normal([1000, 2000])
        c = tf.matmul(gpu_a, gpu_b)
    return c


cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print("cpu:", cpu_time, "  gpu:", gpu_time)

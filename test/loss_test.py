import tensorflow as tf

def custom_mse(y_true, y_pred):
    # 自定义计算损失的逻辑
    squared_difference = tf.square(y_true - y_pred)
    mean_squared_error = tf.reduce_mean(squared_difference)
    return mean_squared_error

import tensorflow as tf


def with_column(x, i, column):
    """
    Given a tensor `x`, change its i-th column with `column`
    x :: (N, T)
    return :: (N, T)
    """

    N, T = tf.shape(x)[0], tf.shape(x)[1]

    left = x[:, :i]
    right = x[:, i+1:]
        
    return tf.concat([left, column, right], axis=1)

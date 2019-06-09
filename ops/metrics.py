import numpy as np
import tensorflow as tf

from rouge import Rouge


rouge = Rouge()


def calculate_rouge(y, y_hat):
    """
    Calculate ROUGE scores between the target 'y' and
    the model prediction 'y_hat'
    """
    
    def f(a, b):
        rouges = rouge.get_scores(' '.join(a), ' '.join(b))[0]
        r1_val, r2_val, rl_val = rouges['rouge-1']["f"], rouges['rouge-2']["f"], rouges['rouge-l']["f"]
        r_avg = np.mean([r1_val, r2_val, rl_val], dtype=np.float32)
        return r1_val, r2_val, rl_val, r_vag
    
    return tf.py_func(f, [y, y_hat], [tf.float32, tf.float32, tf.float32, tf.float32])    
    
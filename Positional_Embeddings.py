import numpy as np
import tensorflow as tf

def positional_encoding(position, d_model):

    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

position = 50  
d_model = 512 
pos_encoding = positional_encoding(position, d_model)

print("Positional Encodings Shape:", pos_encoding.shape)
print("Positional Encodings Example:\n", pos_encoding)
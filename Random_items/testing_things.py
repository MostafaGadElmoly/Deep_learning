import tensorflow as tf # FOR Model
import numpy as np
import matplotlib as plt
import pandas as pd 
import seaborn as sns
import os
from tensorflow.keras.layers import Normalization

#normalize the imputs
normalizer = Normalization(axis=-1, mean=5, variance=4)
x_normalized = tf.constant([[3,4,5,6,7],
                            [4,5,6,7,8]])



print(normalizer(x_normalized))
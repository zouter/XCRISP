# Tommasi, T., Orabona, F., & Caputo, B. (2014). Learning Categories From Few 
# Examples With Multi Model Knowledge Transfer. IEEE Transactions on Pattern 
# Analysis and Machine Intelligence, 36(5), 928–941. doi:10.1109/tpami.2013.197
import tensorflow.compat.v2 as tf
from keras import backend
from tensorflow.keras import regularizers

@tf.keras.utils.register_keras_serializable(package='Custom', name='MMKT')
class MMKT(regularizers.Regularizer):

    def __init__(self, w=0., l2=0.):
        self.w = 0. if w is None else w
        self.l2 = 0. if l2 is None else l2

    def __call__(self, x):
        return tf.reduce_sum(tf.square(x - self.l2 * self.w))

    def get_config(self):
        return {'w': self.w, 'l2': self.l2}

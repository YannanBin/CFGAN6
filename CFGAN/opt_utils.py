import tensorflow as tf
import numpy as np



def spectral_normed_weight_1d(weights, num_iters=1, update_collection=None, with_sigma=False):
    """Performs Spectral Normalization on a 1D weight tensor.

    Args:
        weights: The weight tensor which requires spectral normalization.
        num_iters: Number of SN iterations.
        update_collection: The update collection for assigning persisted variable u.
                           If None, the function will update u during the forward pass.
                           Else if the update_collection equals 'NO_OPS', the function
                           will not update the u during the forward. This is useful for
                           the discriminator, since it does not update u in the second pass.
                           Else, it will put the assignment in a collection defined by the user.
                           Then the user needs to run the assignment explicitly.
        with_sigma: For debugging purposes. If True, the function returns
                    the estimated singular value for the weight tensor.
    Returns:
        w_bar: The normalized weight tensor.
        sigma: The estimated singular value for the weight tensor.
    """
    w_shape = weights.shape.as_list()
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # Reshape to [-1, output_channel]
    initializer = tf.initializers.TruncatedNormal()
    u = tf.Variable(initializer(shape=(1, w_shape[-1])), trainable=False, name='u')
    u_ = u
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /= sigma
    if update_collection is None:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)
        if update_collection != 'NO_OPS':
            tf.compat.v1.add_to_collection(update_collection, u.assign(u_))

    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar

def _l2normalize(v, eps=1e-12):
    """L2 normalize a tensor."""
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)




def snconv1d(input_, output_dim, k_w=3, d_w=1, sn_iters=1, update_collection=None, name='snconv1d'):
    with tf.name_scope(name):
        w = tf.Variable(tf.random.truncated_normal([k_w, input_.shape[-1], output_dim], stddev=0.02), name='kernel')
        w_bar = spectral_normed_weight_1d(w, num_iters=sn_iters, update_collection=update_collection)

        conv = tf.nn.conv1d(input_, w_bar, stride=d_w, padding='SAME')
        biases = tf.Variable(tf.zeros([output_dim]), name='bias')
        conv = tf.nn.bias_add(conv, biases)

        conv = tf.nn.leaky_relu(conv, alpha=0.2)

        return conv



def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature, eps=1e-20):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    gumbel_noise = sample_gumbel(tf.shape(logits), eps=eps)
    y = logits + gumbel_noise
    return tf.nn.softmax(y / temperature)

def gumbel_softmax(logits, temperature, hard=False, eps=1e-20):
    """ ST-gumple-softmax
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probability distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, eps)
    
    if hard:
        y_hard = tf.cast(tf.one_hot(tf.argmax(y, axis=-1), tf.shape(logits)[-1]), y.dtype)
        # This trick is borrowed from PyTorch implementation
        y = tf.stop_gradient(y_hard - y) + y
    
    return y








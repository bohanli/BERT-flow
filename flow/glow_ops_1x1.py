"""
modified from 
https://github.com/tensorflow/tensor2tensor/blob/8a084a4d56/tensor2tensor/models/research/glow_ops.py

modifications are as follows:
  1. replace tfp with tf because neither tfp 0.6 or 0.7 is compatible with tf 1.14
  2. remove support for video-related operators like conv3d
  3. remove support for conditional distributions
"""
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
# import tensorflow_probability as tfp

import functools
import numpy as np
import scipy


def get_shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i, dim in enumerate(static):
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

def get_eps(dist, x):
  """Z = (X - mu) / sigma."""
  return (x - dist.loc) / dist.scale


def set_eps(dist, eps):
  """Z = eps * sigma + mu."""
  return eps * dist.scale + dist.loc


# ===============================================
@add_arg_scope
def assign(w, initial_value):
  w = w.assign(initial_value)
  with tf.control_dependencies([w]):
    return w

@add_arg_scope
def get_variable_ddi(name, shape, initial_value, dtype=tf.float32, init=False,
                     trainable=True):
  """Wrapper for data-dependent initialization."""
  # If init is a tf bool: w is assigned dynamically at runtime.
  # If init is a python bool: then w is determined during graph construction.
  w = tf.compat.v1.get_variable(name, shape, dtype, None, trainable=trainable)
  if isinstance(init, bool):
    if init:
      return assign(w, initial_value)
    return w
  else:
    return tf.cond(init, lambda: assign(w, initial_value), lambda: w)

@add_arg_scope
def get_dropout(x, rate=0.0, init=True):
  """Dropout x with dropout_rate = rate.
  Apply zero dropout during init or prediction time.
  Args:
    x: 4-D Tensor, shape=(NHWC).
    rate: Dropout rate.
    init: Initialization.
  Returns:
    x: activations after dropout.
  """
  if init or rate == 0:
    return x
  return tf.layers.dropout(x, rate=rate, training=True) # TODO

def default_initializer(std=0.05):
  return tf.random_normal_initializer(0., std)

# ===============================================

# Activation normalization
# Convenience function that does centering+scaling

@add_arg_scope
def actnorm(name, x, logscale_factor=3., reverse=False, init=False,
            trainable=True):
  """x_{ij} = s x x_{ij} + b. Per-channel scaling and bias.
  If init is set to True, the scaling and bias are initialized such
  that the mean and variance of the output activations of the first minibatch
  are zero and one respectively.
  Args:
    name: variable scope.
    x: input
    logscale_factor: Used in actnorm_scale. Optimizes f(ls*s') instead of f(s)
                     where s' = s / ls. Helps in faster convergence.
    reverse: forward or reverse operation.
    init: Whether or not to do data-dependent initialization.
    trainable:
  Returns:
    x: output after adding bias and scaling.
    objective: log(sum(s))
  """
  var_arg_scope = arg_scope([get_variable_ddi], trainable=trainable)
  var_scope = tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE)

  with var_scope, var_arg_scope:
    if not reverse:
      x = actnorm_center(name + "_center", x, reverse, init=init)
      x, objective = actnorm_scale(
          name + "_scale", x, logscale_factor=logscale_factor,
          reverse=reverse, init=init)
    else:
      x, objective = actnorm_scale(
          name + "_scale", x, logscale_factor=logscale_factor,
          reverse=reverse, init=init)
      x = actnorm_center(name + "_center", x, reverse, init=init)
    return x, objective


@add_arg_scope
def actnorm_center(name, x, reverse=False, init=False):
  """Add a bias to x.
  Initialize such that the output of the first minibatch is zero centered
  per channel.
  Args:
    name: scope
    x: 2-D or 4-D Tensor.
    reverse: Forward or backward operation.
    init: data-dependent initialization.
  Returns:
    x_center: (x + b), if reverse is True and (x - b) otherwise.
  """
  shape = get_shape_list(x)
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
    assert len(shape) == 2 or len(shape) == 4
    if len(shape) == 2:
      x_mean = tf.reduce_mean(x, [0], keepdims=True)
      b = get_variable_ddi("b", (1, shape[1]), initial_value=-x_mean,
                           init=init)
    elif len(shape) == 4:
      x_mean = tf.reduce_mean(x, [0, 1, 2], keepdims=True)
      b = get_variable_ddi(
          "b", (1, 1, 1, shape[3]), initial_value=-x_mean, init=init)

    if not reverse:
      x += b
    else:
      x -= b
    return x


@add_arg_scope
def actnorm_scale(name, x, logscale_factor=3., reverse=False, init=False):
  """Per-channel scaling of x."""
  x_shape = get_shape_list(x)
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):

    # Variance initialization logic.
    assert len(x_shape) == 2 or len(x_shape) == 4
    if len(x_shape) == 2:
      x_var = tf.reduce_mean(x**2, [0], keepdims=True)
      logdet_factor = 1
      var_shape = (1, x_shape[1])
    elif len(x_shape) == 4:
      x_var = tf.reduce_mean(x**2, [0, 1, 2], keepdims=True)
      logdet_factor = x_shape[1]*x_shape[2]
      var_shape = (1, 1, 1, x_shape[3])

    init_value = tf.math.log(1.0 / (tf.sqrt(x_var) + 1e-6)) / logscale_factor
    logs = get_variable_ddi("logs", var_shape, initial_value=init_value,
                            init=init)
    logs = logs * logscale_factor

    # Function and reverse function.
    if not reverse:
      x = x * tf.exp(logs)
    else:
      x = x * tf.exp(-logs)

    # Objective calculation, h * w * sum(log|s|)
    dlogdet = tf.reduce_sum(logs) * logdet_factor
    if reverse:
      dlogdet *= -1
    return x, dlogdet


# ===============================================


@add_arg_scope
def invertible_1x1_conv(name, x, reverse=False, permutation=False):
  """1X1 convolution on x.
  The 1X1 convolution is parametrized as P*L*(U + sign(s)*exp(log(s))) where
  1. P is a permutation matrix.
  2. L is a lower triangular matrix with diagonal entries unity.
  3. U is a upper triangular matrix where the diagonal entries zero.
  4. s is a vector.
  sign(s) and P are fixed and the remaining are optimized. P, L, U and s are
  initialized by the PLU decomposition of a random rotation matrix.
  Args:
    name: scope
    x: Input Tensor.
    reverse: whether the pass is from z -> x or x -> z.
  Returns:
    x_conv: x after a 1X1 convolution is applied on x.
    objective: sum(log(s))
  """
  _, height, width, channels = get_shape_list(x)
  w_shape = [channels, channels]

  if permutation:
    np_w = np.zeros((channels, channels)).astype("float32")
    for i in range(channels):
        np_w[i][channels-1-i] = 1.

    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
      w = tf.compat.v1.get_variable("w", initializer=np_w, trainable=False)

      # If height or width cannot be statically determined then they end up as
      # tf.int32 tensors, which cannot be directly multiplied with a floating
      # point tensor without a cast.
      objective = 0.
      if not reverse:
        w = tf.reshape(w, [1, 1] + w_shape)
        x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format="NHWC")
      else:
        w_inv = tf.reshape(tf.linalg.inv(w), [1, 1] + w_shape)
        x = tf.nn.conv2d(
            x, w_inv, [1, 1, 1, 1], "SAME", data_format="NHWC")
        objective *= -1
    return x, objective
  else:
    # Random rotation-matrix Q
    random_matrix = np.random.rand(channels, channels)
    np_w = scipy.linalg.qr(random_matrix)[0].astype("float32")

    # Initialize P,L,U and s from the LU decomposition of a random rotation matrix
    np_p, np_l, np_u = scipy.linalg.lu(np_w)
    np_s = np.diag(np_u)
    np_sign_s = np.sign(np_s)
    np_log_s = np.log(np.abs(np_s))
    np_u = np.triu(np_u, k=1)

    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
      p = tf.compat.v1.get_variable("P", initializer=np_p, trainable=False)
      l = tf.compat.v1.get_variable("L", initializer=np_l)
      sign_s = tf.compat.v1.get_variable(
          "sign_S", initializer=np_sign_s, trainable=False)
      log_s = tf.compat.v1.get_variable("log_S", initializer=np_log_s)
      u = tf.compat.v1.get_variable("U", initializer=np_u)

      # W = P * L * (U + sign_s * exp(log_s))
      l_mask = np.tril(np.ones([channels, channels], dtype=np.float32), -1)
      l = l * l_mask + tf.eye(channels, channels)
      u = u * np.transpose(l_mask) + tf.linalg.diag(sign_s * tf.exp(log_s))
      w = tf.matmul(p, tf.matmul(l, u))

      # If height or width cannot be statically determined then they end up as
      # tf.int32 tensors, which cannot be directly multiplied with a floating
      # point tensor without a cast.
      objective = tf.reduce_sum(log_s) * tf.cast(height * width, log_s.dtype)
      if not reverse:
        w = tf.reshape(w, [1, 1] + w_shape)
        x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format="NHWC")
      else:
        w_inv = tf.reshape(tf.linalg.inv(w), [1, 1] + w_shape)
        x = tf.nn.conv2d(
            x, w_inv, [1, 1, 1, 1], "SAME", data_format="NHWC")
        objective *= -1
    return x, objective




# ===============================================

def add_edge_bias(x, filter_size):
  """Pad x and concatenates an edge bias across the depth of x.
  The edge bias can be thought of as a binary feature which is unity when
  the filter is being convolved over an edge and zero otherwise.
  Args:
    x: Input tensor, shape (NHWC)
    filter_size: filter_size to determine padding.
  Returns:
    x_pad: Input tensor, shape (NHW(c+1))
  """
  x_shape = get_shape_list(x)
  if filter_size[0] == 1 and filter_size[1] == 1:
    return x
  a = (filter_size[0] - 1) // 2  # vertical padding size
  b = (filter_size[1] - 1) // 2  # horizontal padding size
  padding = [[0, 0], [a, a], [b, b], [0, 0]]
  x_bias = tf.zeros(x_shape[:-1] + [1])

  x = tf.pad(x, padding)
  x_pad = tf.pad(x_bias, padding, constant_values=1)
  return tf.concat([x, x_pad], axis=3)


@add_arg_scope
def conv(name, x, output_channels, filter_size=None, stride=None,
         logscale_factor=3.0, apply_actnorm=True, conv_init="default",
         dilations=None):
  """Convolutional layer with edge bias padding and optional actnorm.
  If x is 5-dimensional, actnorm is applied independently across every
  time-step.
  Args:
    name: variable scope.
    x: 4-D Tensor or 5-D Tensor of shape NHWC or NTHWC
    output_channels: Number of output channels.
    filter_size: list of ints, if None [3, 3] and [2, 3, 3] are defaults for
                 4-D and 5-D input tensors respectively.
    stride: list of ints, default stride: 1
    logscale_factor: see actnorm for parameter meaning.
    apply_actnorm: if apply_actnorm the activations of the first minibatch
                   have zero mean and unit variance. Else, there is no scaling
                   applied.
    conv_init: default or zeros. default is a normal distribution with 0.05 std.
    dilations: List of integers, apply dilations.
  Returns:
    x: actnorm(conv2d(x))
  Raises:
    ValueError: if init is set to "zeros" and apply_actnorm is set to True.
  """
  if conv_init == "zeros" and apply_actnorm:
    raise ValueError("apply_actnorm is unstable when init is set to zeros.")

  x_shape = get_shape_list(x)
  is_2d = len(x_shape) == 4
  num_steps = x_shape[1]

  # set filter_size, stride and in_channels
  if is_2d:
    if filter_size is None:
      filter_size = [1, 1] # filter_size = [3, 3]
    if stride is None:
      stride = [1, 1]
    if dilations is None:
      dilations = [1, 1, 1, 1]
    actnorm_func = actnorm
    x = add_edge_bias(x, filter_size=filter_size)
    conv_filter = tf.nn.conv2d
  else:
    raise NotImplementedError('x must be a NHWC 4-D Tensor!')

  in_channels = get_shape_list(x)[-1]
  filter_shape = filter_size + [in_channels, output_channels]
  stride_shape = [1] + stride + [1]

  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):

    if conv_init == "default":
      initializer = default_initializer()
    elif conv_init == "zeros":
      initializer = tf.zeros_initializer()

    w = tf.compat.v1.get_variable("W", filter_shape, tf.float32, initializer=initializer)
    x = conv_filter(x, w, stride_shape, padding="VALID", dilations=dilations)
    if apply_actnorm:
      x, _ = actnorm_func("actnorm", x, logscale_factor=logscale_factor)
    else:
      x += tf.compat.v1.get_variable("b", [1, 1, 1, output_channels],
                           initializer=tf.zeros_initializer())
      logs = tf.compat.v1.get_variable("logs", [1, output_channels],
                             initializer=tf.zeros_initializer())
      x *= tf.exp(logs * logscale_factor)
    return x


@add_arg_scope
def conv_block(name, x, mid_channels, dilations=None, activation="relu",
               dropout=0.0):
  """2 layer conv block used in the affine coupling layer.
  Args:
    name: variable scope.
    x: 4-D or 5-D Tensor.
    mid_channels: Output channels of the second layer.
    dilations: Optional, list of integers.
    activation: relu or gatu.
      If relu, the second layer is relu(W*x)
      If gatu, the second layer is tanh(W1*x) * sigmoid(W2*x)
    dropout: Dropout probability.
  Returns:
    x: 4-D Tensor: Output activations.
  """
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):

    x_shape = get_shape_list(x)
    is_2d = len(x_shape) == 4
    num_steps = x_shape[1]
    if is_2d:
      first_filter = [1, 1] # first_filter = [3, 3]
      second_filter = [1, 1]
    else:
      raise NotImplementedError('x must be a NHWC 4-D Tensor!')

    # Edge Padding + conv2d + actnorm + relu:
    # [output: 512 channels]
    x = conv("1_1", x, output_channels=mid_channels, filter_size=first_filter,
             dilations=dilations)
    x = tf.nn.relu(x)
    x = get_dropout(x, rate=dropout)

    # Padding + conv2d + actnorm + activation.
    # [input, output: 512 channels]
    if activation == "relu":
      x = conv("1_2", x, output_channels=mid_channels,
               filter_size=second_filter, dilations=dilations)
      x = tf.nn.relu(x)
    elif activation == "gatu":
      # x = tanh(w1*x) * sigm(w2*x)
      x_tanh = conv("1_tanh", x, output_channels=mid_channels,
                    filter_size=second_filter, dilations=dilations)
      x_sigm = conv("1_sigm", x, output_channels=mid_channels,
                    filter_size=second_filter, dilations=dilations)
      x = tf.nn.tanh(x_tanh) * tf.nn.sigmoid(x_sigm)

    x = get_dropout(x, rate=dropout)
    return x


@add_arg_scope
def conv_stack(name, x, mid_channels, output_channels, dilations=None,
               activation="relu", dropout=0.0):
  """3-layer convolutional stack.
  Args:
    name: variable scope.
    x: 5-D Tensor.
    mid_channels: Number of output channels of the first layer.
    output_channels: Number of output channels.
    dilations: Dilations to apply in the first 3x3 layer and the last 3x3 layer.
               By default, apply no dilations.
    activation: relu or gatu.
      If relu, the second layer is relu(W*x)
      If gatu, the second layer is tanh(W1*x) * sigmoid(W2*x)
    dropout: float, 0.0
  Returns:
    output: output of 3 layer conv network.
  """
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):

    x = conv_block("conv_block", x, mid_channels=mid_channels,
                   dilations=dilations, activation=activation,
                   dropout=dropout)

    # Final layer.
    x = conv("zeros", x, apply_actnorm=False, conv_init="zeros",
             output_channels=output_channels, dilations=dilations)
  return x


@add_arg_scope
def additive_coupling(name, x, mid_channels=512, reverse=False,
                      activation="relu", dropout=0.0):
  """Reversible additive coupling layer.
  Args:
    name: variable scope.
    x: 4-D Tensor, shape=(NHWC).
    mid_channels: number of channels in the coupling layer.
    reverse: Forward or reverse operation.
    activation: "relu" or "gatu"
    dropout: default, 0.0
  Returns:
    output: 4-D Tensor, shape=(NHWC)
    objective: 0.0
  """
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
    output_channels = get_shape_list(x)[-1] // 2
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)

    z1 = x1
    shift = conv_stack("nn", x1, mid_channels, output_channels=output_channels,
                       activation=activation, dropout=dropout)

    if not reverse:
      z2 = x2 + shift
    else:
      z2 = x2 - shift
    return tf.concat([z1, z2], axis=3), 0.0


@add_arg_scope
def affine_coupling(name, x, mid_channels=512, activation="relu",
                    reverse=False, dropout=0.0):
  """Reversible affine coupling layer.
  Args:
    name: variable scope.
    x: 4-D Tensor.
    mid_channels: number of channels in the coupling layer.
    activation: Can be either "relu" or "gatu".
    reverse: Forward or reverse operation.
    dropout: default, 0.0
  Returns:
    output: x shifted and scaled by an affine transformation.
    objective: log-determinant of the jacobian
  """
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
    x_shape = get_shape_list(x)
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)

    # scale, shift = NN(x1)
    # If reverse:
    # z2 = scale * (x2 + shift)
    # Else:
    # z2 = (x2 / scale) - shift
    z1 = x1
    log_scale_and_shift = conv_stack(
        "nn", x1, mid_channels, x_shape[-1], activation=activation,
        dropout=dropout)
    shift = log_scale_and_shift[:, :, :, 0::2]
    scale = tf.nn.sigmoid(log_scale_and_shift[:, :, :, 1::2] + 2.0)
    if not reverse:
      z2 = (x2 + shift) * scale
    else:
      z2 = x2 / scale - shift

    objective = tf.reduce_sum(tf.math.log(scale), axis=[1, 2, 3])
    if reverse:
      objective *= -1
    return tf.concat([z1, z2], axis=3), objective


# ===============================================


@add_arg_scope
def single_conv_dist(name, x, output_channels=None):
  """A 1x1 convolution mapping x to a standard normal distribution at init.
  Args:
    name: variable scope.
    x: 4-D Tensor.
    output_channels: number of channels of the mean and std.
  """
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
    x_shape = get_shape_list(x)
    if output_channels is None:
      output_channels = x_shape[-1]
    mean_log_scale = conv("conv2d", x, output_channels=2*output_channels,
                          conv_init="zeros", apply_actnorm=False)
    mean = mean_log_scale[:, :, :, 0::2]
    log_scale = mean_log_scale[:, :, :, 1::2]
    return tf.distributions.Normal(mean, tf.exp(log_scale))


# # ===============================================


@add_arg_scope
def revnet_step(name, x, hparams, reverse=True):
  """One step of glow generative flow.
  Actnorm + invertible 1X1 conv + affine_coupling.
  Args:
    name: used for variable scope.
    x: input
    hparams: coupling_width is the only hparam that is being used in
             this function.
    reverse: forward or reverse pass.
  Returns:
    z: Output of one step of reversible flow.
  """
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
    if hparams.coupling == "additive":
      coupling_layer = functools.partial(
          additive_coupling, name="additive", reverse=reverse,
          mid_channels=hparams.coupling_width,
          activation=hparams.activation, 
          dropout=hparams.coupling_dropout if hparams.is_training else 0)
    else:
      coupling_layer = functools.partial(
          affine_coupling, name="affine", reverse=reverse,
          mid_channels=hparams.coupling_width,
          activation=hparams.activation, 
          dropout=hparams.coupling_dropout if hparams.is_training else 0)

    if "permutation" in hparams and hparams["permutation"] == True:
      ops = [
          functools.partial(actnorm, name="actnorm", reverse=reverse),
          functools.partial(invertible_1x1_conv, name="invertible", reverse=reverse, permutation=True), 
          coupling_layer]
    else:
      ops = [
          functools.partial(actnorm, name="actnorm", reverse=reverse),
          functools.partial(invertible_1x1_conv, name="invertible", reverse=reverse), 
          coupling_layer]

    if reverse:
      ops = ops[::-1]

    objective = 0.0
    for op in ops:
      x, curr_obj = op(x=x)
      objective += curr_obj
    return x, objective


def revnet(name, x, hparams, reverse=True):
  """'hparams.depth' steps of generative flow.
  Args:
    name: variable scope for the revnet block.
    x: 4-D Tensor, shape=(NHWC).
    hparams: HParams.
    reverse: bool, forward or backward pass.
  Returns:
    x: 4-D Tensor, shape=(NHWC).
    objective: float.
  """
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
    steps = np.arange(hparams.depth)
    if reverse:
      steps = steps[::-1]

    objective = 0.0
    for step in steps:
      x, curr_obj = revnet_step(
          "revnet_step_%d" % step, x, hparams, reverse=reverse)
      objective += curr_obj
    return x, objective

# ===============================================

@add_arg_scope
def compute_prior(name, z, latent, hparams, condition=False, state=None,
                  temperature=1.0):
  """Distribution on z_t conditioned on z_{t-1} and latent.
  Args:
    name: variable scope.
    z: 4-D Tensor.
    latent: optional,
            if hparams.latent_dist_encoder == "pointwise", this is a list
            of 4-D Tensors of length hparams.num_cond_latents.
            else, this is just a 4-D Tensor
            The first-three dimensions of the latent should be the same as z.
    hparams: next_frame_glow_hparams.
    condition: Whether or not to condition the distribution on latent.
    state: tf.nn.rnn_cell.LSTMStateTuple.
           the current state of a LSTM used to model the distribution. Used
           only if hparams.latent_dist_encoder = "conv_lstm".
    temperature: float, temperature with which to sample from the Gaussian.
  Returns:
    prior_dist: instance of tfp.distributions.Normal
    state: Returns updated state.
  Raises:
    ValueError: If hparams.latent_dist_encoder is "pointwise" and if the shape
                of latent is different from z.
  """
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
    z_shape = get_shape_list(z)
    h = tf.zeros(z_shape, dtype=tf.float32)
    prior_dist = tf.distributions.Normal(h, tf.exp(h))
    return prior_dist, state



@add_arg_scope
def split(name, x, reverse=False, eps=None, eps_std=None, cond_latents=None,
          hparams=None, state=None, condition=False, temperature=1.0):
  """Splits / concatenates x into x1 and x2 across number of channels.
  For the forward pass, x2 is assumed be gaussian,
  i.e P(x2 | x1) ~ N(mu, sigma) where mu and sigma are the outputs of
  a network conditioned on x1 and optionally on cond_latents.
  For the reverse pass, x2 is determined from mu(x1) and sigma(x1).
  This is deterministic/stochastic depending on whether eps is provided.
  Args:
    name: variable scope.
    x: 4-D Tensor, shape (NHWC).
    reverse: Forward or reverse pass.
    eps: If eps is provided, x2 is set to be mu(x1) + eps * sigma(x1).
    eps_std: Sample x2 with the provided eps_std.
    cond_latents: optionally condition x2 on cond_latents.
    hparams: next_frame_glow hparams.
    state: tf.nn.rnn_cell.LSTMStateTuple.. Current state of the LSTM over z_2.
           Used only when hparams.latent_dist_encoder == "conv_lstm"
    condition: bool, Whether or not to condition the distribution on
               cond_latents.
    temperature: Temperature with which to sample from the gaussian.
  Returns:
    If reverse:
      x: 4-D Tensor, concats input and x2 across channels.
      x2: 4-D Tensor, a sample from N(mu(x1), sigma(x1))
    Else:
      x1: 4-D Tensor, Output of the split operation.
      logpb: log-probability of x2 belonging to mu(x1), sigma(x1)
      eps: 4-D Tensor, (x2 - mu(x1)) / sigma(x1)
      x2: 4-D Tensor, Latent representation at the current level.
    state: Current LSTM state.
           4-D Tensor, only if hparams.latent_dist_encoder is set to conv_lstm.
  Raises:
    ValueError: If latent is provided and shape is not equal to NHW(C/2)
                where (NHWC) is the size of x.
  """
  # TODO(mechcoder) Change the return type to be a dict.
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
    if not reverse:
      x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)

      # objective: P(x2|x1) ~N(x2 ; NN(x1))
      prior_dist, state = compute_prior(
          "prior_on_z2", x1, cond_latents, hparams, condition, state=state)
      logpb = tf.reduce_sum(prior_dist.log_prob(x2), axis=[1, 2, 3])
      eps = get_eps(prior_dist, x2)
      return x1, logpb, eps, x2, state
    else:
      prior_dist, state = compute_prior(
          "prior_on_z2", x, cond_latents, hparams, condition, state=state,
          temperature=temperature)
      if eps is not None:
        x2 = set_eps(prior_dist, eps)
      elif eps_std is not None:
        x2 = eps_std * tf.random_normal(get_shape_list(x))
      else:
        x2 = prior_dist.sample()
      return tf.concat([x, x2], 3), x2, state


@add_arg_scope
def squeeze(name, x, factor=2, reverse=True):
  """Block-wise spatial squeezing of x to increase the number of channels.
  Args:
    name: Used for variable scoping.
    x: 4-D Tensor of shape (batch_size X H X W X C)
    factor: Factor by which the spatial dimensions should be squeezed.
    reverse: Squueze or unsqueeze operation.
  Returns:
    x: 4-D Tensor of shape (batch_size X (H//factor) X (W//factor) X
       (cXfactor^2). If reverse is True, then it is factor = (1 / factor)
  """
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
    shape = get_shape_list(x)
    if factor == 1:
      return x
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])

    if not reverse:
      assert height % factor == 0 and width % factor == 0
      x = tf.reshape(x, [-1, height//factor, factor,
                         width//factor, factor, n_channels])
      x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
      x = tf.reshape(x, [-1, height//factor, width //
                         factor, n_channels*factor*factor])
    else:
      x = tf.reshape(
          x, (-1, height, width, int(n_channels/factor**2), factor, factor))
      x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
      x = tf.reshape(x, (-1, int(height*factor),
                         int(width*factor), int(n_channels/factor**2)))
    return x


def get_cond_latents_at_level(cond_latents, level, hparams):
  """Returns a single or list of conditional latents at level 'level'."""
  if cond_latents:
    if hparams.latent_dist_encoder in ["conv_net", "conv3d_net"]:
      return [cond_latent[level] for cond_latent in cond_latents]
    elif hparams.latent_dist_encoder in ["pointwise", "conv_lstm"]:
      return cond_latents[level]


def check_cond_latents(cond_latents, hparams):
  """Shape checking for cond_latents."""
  if cond_latents is None:
    return
  if not isinstance(cond_latents[0], list):
    cond_latents = [cond_latents]
  exp_num_latents = hparams.num_cond_latents
  if hparams.latent_dist_encoder == "conv_net":
    exp_num_latents += int(hparams.cond_first_frame)
  if len(cond_latents) != exp_num_latents:
    raise ValueError("Expected number of cond_latents: %d, got %d" %
                     (exp_num_latents, len(cond_latents)))
  for cond_latent in cond_latents:
    if len(cond_latent) != hparams.n_levels - 1:
      raise ValueError("Expected level_latents to be %d, got %d" %
                       (hparams.n_levels - 1, len(cond_latent)))


@add_arg_scope
def encoder_decoder(name, x, hparams, eps=None, reverse=False,
                    cond_latents=None, condition=False, states=None,
                    temperature=1.0):
  """Glow encoder-decoder. n_levels of (Squeeze + Flow + Split.) operations.
  Args:
    name: variable scope.
    x: 4-D Tensor, shape=(NHWC).
    hparams: HParams.
    eps: Stores (glow(x) - mu) / sigma during the forward pass.
         Used only to test if the network is reversible.
    reverse: Forward or reverse pass.
    cond_latents: list of lists of tensors.
                  outer length equals hparams.num_cond_latents
                  innter length equals hparams.num_levels - 1.
    condition: If set to True, condition the encoder/decoder on cond_latents.
    states: LSTM states, used only if hparams.latent_dist_encoder is set
            to "conv_lstm.
    temperature: Temperature set during sampling.
  Returns:
    x: If reverse, decoded image, else the encoded glow latent representation.
    objective: log-likelihood.
    eps: list of tensors, shape=(num_levels-1).
         Stores (glow(x) - mu_level(x)) / sigma_level(x)) for each level.
    all_latents: list of tensors, shape=(num_levels-1).
                 Latent representations for each level.
    new_states: list of tensors, shape=(num_levels-1).
                useful only if hparams.latent_dist_encoder="conv_lstm", returns
                the current state of each level.
  """
  # TODO(mechcoder) Change return_type to a dict to be backward compatible.
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):

    if states and len(states) != hparams.n_levels - 1:
      raise ValueError("Expected length of states to be %d, got %d" %
                       (hparams.n_levels - 1, len(states)))
    if states is None:
      states = [None] * (hparams.n_levels - 1)
    if eps and len(eps) != hparams.n_levels - 1:
      raise ValueError("Expected length of eps to be %d, got %d" %
                       (hparams.n_levels - 1, len(eps)))
    if eps is None:
      eps = [None] * (hparams.n_levels - 1)
    check_cond_latents(cond_latents, hparams)

    objective = 0.0
    all_eps = []
    all_latents = []
    new_states = []

    if not reverse:
      # Squeeze + Flow + Split
      for level in range(hparams.n_levels):
        # x = squeeze("squeeze_%d" % level, x, factor=2, reverse=False)

        x, obj = revnet("revnet_%d" % level, x, hparams, reverse=False)
        objective += obj

        if level < hparams.n_levels - 1:
          curr_cond_latents = get_cond_latents_at_level(
              cond_latents, level, hparams)
          x, obj, eps, z, state = split("split_%d" % level, x, reverse=False,
                                        cond_latents=curr_cond_latents,
                                        condition=condition,
                                        hparams=hparams, state=states[level])
          objective += obj
          all_eps.append(eps)
          all_latents.append(z)
          new_states.append(state)

      return x, objective, all_eps, all_latents, new_states

    else:
      for level in reversed(range(hparams.n_levels)):
        if level < hparams.n_levels - 1:

          curr_cond_latents = get_cond_latents_at_level(
              cond_latents, level, hparams)

          x, latent, state = split("split_%d" % level, x, eps=eps[level],
                                   reverse=True, cond_latents=curr_cond_latents,
                                   condition=condition, hparams=hparams,
                                   state=states[level],
                                   temperature=temperature)
          new_states.append(state)
          all_latents.append(latent)

        x, obj = revnet(
            "revnet_%d" % level, x, hparams=hparams, reverse=True)
        objective += obj
        # x = squeeze("squeeze_%d" % level, x, reverse=True)
      return x, objective, all_latents[::-1], new_states[::-1]


# ===============================================


@add_arg_scope
def top_prior(name, z_shape, learn_prior="normal", temperature=1.0):
  """Unconditional prior distribution.
  Args:
    name: variable scope
    z_shape: Shape of the mean / scale of the prior distribution.
    learn_prior: Possible options are "normal" and "single_conv".
                 If set to "single_conv", the gaussian is parametrized by a
                 single convolutional layer whose input are an array of zeros
                 and initialized such that the mean and std are zero and one.
                 If set to "normal", the prior is just a Gaussian with zero
                 mean and unit variance.
    temperature: Temperature with which to sample from the Gaussian.
  Returns:
    objective: 1-D Tensor shape=(batch_size,) summed across spatial components.
  Raises:
    ValueError: If learn_prior not in "normal" or "single_conv"
  """
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
    h = tf.zeros(z_shape, dtype=tf.float32)
    prior_dist = tf.distributions.Normal(h, tf.exp(h))
    return prior_dist
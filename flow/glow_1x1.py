import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
import flow.glow_ops_1x1 as glow_ops
from flow.glow_ops_1x1 import get_shape_list
import flow.glow_init_hook


import numpy as np
import os, sys

arg_scope = tf.contrib.framework.arg_scope
add_arg_scope = tf.contrib.framework.add_arg_scope

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

class Glow():
  
  def __init__(self, hparams):
    self.hparams = hparams

  @property
  def is_predicting(self):
    return not self.is_training

  @staticmethod
  def train_hooks():
    #del hook_context
    return [glow_init_hook.GlowInitHook()]

  def top_prior(self):
    """Objective based on the prior over latent z.

    Returns:
      dist: instance of tfp.distributions.Normal, prior distribution.
    """
    return glow_ops.top_prior(
        "top_prior", self.z_top_shape, learn_prior=self.hparams.top_prior)

  def body(self, features, is_training):
    if is_training:
      init_features = features
      init_op = self.objective_tower(init_features, init=True)
      init_op = tf.Print(
          init_op, [init_op], message="Triggering data-dependent init.",
          first_n=20)
      tf.compat.v1.add_to_collection("glow_init_op", init_op)
    return self.objective_tower(features, init=False)

  def objective_tower(self, features, init=True):
    """Objective in terms of bits-per-pixel. 
    """    
    #features = tf.expand_dims(features, [1, 2])
    features = features[:, tf.newaxis, tf.newaxis, :]
    x = features

    objective = 0

    # The arg_scope call ensures that the actnorm parameters are set such that
    # the per-channel output activations have zero mean and unit variance
    # ONLY during the first step. After that the parameters are learned
    # through optimisation.
    ops = [glow_ops.get_variable_ddi, glow_ops.actnorm, glow_ops.get_dropout]
    with arg_scope(ops, init=init):
      encoder = glow_ops.encoder_decoder

      self.z, encoder_objective, self.eps, _, _ = encoder(
          "flow", x, self.hparams, eps=None, reverse=False)
      objective += encoder_objective

      self.z_top_shape = get_shape_list(self.z)
      prior_dist = self.top_prior()
      prior_objective = tf.reduce_sum(
          prior_dist.log_prob(self.z), axis=[1, 2, 3])
      #self.z_sample = prior_dist.sample()
      objective += prior_objective

    # bits per pixel
    _, h, w, c = get_shape_list(x)
    objective = -objective / (np.log(2) * h * w * c)

    self.z = tf.concat(self.eps + [self.z], axis=-1)
    return objective
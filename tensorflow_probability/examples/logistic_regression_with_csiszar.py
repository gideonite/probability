# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Trains a Bayesian logistic regression model on synthetic data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports
from absl import flags
import matplotlib.colors
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

tfd = tfp.distributions

flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=1500,
                     help="Number of training steps to run.")
flags.DEFINE_integer("batch_size",
                     default=32,
                     help="Batch size.")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                         "logistic_regression/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("num_examples",
                     default=256,
                     help="Number of datapoints to generate.")
flags.DEFINE_integer("num_monte_carlo",
                     default=50,
                     help="Monte Carlo samples to visualize weight posterior.")
flags.DEFINE_integer("seed",
                     default=0,
                     help="Set the random seed for this run")
FLAGS = flags.FLAGS

def toy_logistic_data(num_examples, input_size=2, weights_prior_stddev=5.0):
  """Generates synthetic data for binary classification.

  Args:
    num_examples: The number of samples to generate (scalar Python `int`).
    input_size: The input space dimension (scalar Python `int`).
    weights_prior_stddev: The prior standard deviation of the weight
      vector. (scalar Python `float`).

  Returns:
    random_weights: Sampled weights as a Numpy `array` of shape
      `[input_size]`.
    random_bias: Sampled bias as a scalar Python `float`.
    design_matrix: Points sampled uniformly from the cube `[-1,
       1]^{input_size}`, as a Numpy `array` of shape `(num_examples,
       input_size)`.
    labels: Labels sampled from the logistic model `p(label=1) =
      logistic(dot(inputs, random_weights) + random_bias)`, as a Numpy
      `int32` `array` of shape `(num_examples, 1)`.
  """
  random_weights = weights_prior_stddev * np.random.randn(input_size)
  random_bias = np.random.randn()
  design_matrix = np.random.rand(num_examples, input_size) * 2 - 1
  logits = np.reshape(
      np.dot(design_matrix, random_weights) + random_bias,
      (-1, 1))
  p_labels = 1. / (1 + np.exp(-logits))
  labels = np.int32(p_labels > np.random.rand(num_examples, 1))
  labels = np.squeeze(labels) # TODO why is this necessary? Why does the old version handle this without a problem?
  return random_weights, random_bias, np.float32(design_matrix), labels

def toy_logistic_data_with_ones_col(num_examples, input_size=2, weights_prior_stddev=5.0):
  '''Put the bias into the weights trick by adding a col of 1s.'''
  weights, bias, design_matrix, labels \
      = toy_logistic_data(num_examples, input_size=2, weights_prior_stddev=5.0)

  weights = np.append(weights, bias)
  design_matrix = np.c_[design_matrix, np.ones(design_matrix.shape[0])]

  return weights, np.float32(design_matrix), labels


def visualize_decision(inputs, labels, true_w_b, candidate_w_bs, fname):
  """Utility method to visualize decision boundaries in R^2.

  Args:
    inputs: Input points, as a Numpy `array` of shape `[num_examples, 2]`.
    labels: Numpy `float`-like array of shape `[num_examples, 1]` giving a
      label for each point.
    true_w_b: A `tuple` `(w, b)` where `w` is a Numpy array of
       shape `[2]` and `b` is a scalar `float`, interpreted as a
       decision rule of the form `dot(inputs, w) + b > 0`.
    candidate_w_bs: Python `iterable` containing tuples of the same form as
       true_w_b.
    fname: The filename to save the plot as a PNG image (Python `str`).
  """
  fig = figure.Figure(figsize=(6, 6))
  canvas = backend_agg.FigureCanvasAgg(fig)
  ax = fig.add_subplot(1, 1, 1)
  ax.scatter(inputs[:, 0], inputs[:, 1],
             c=np.float32(labels),
             cmap=matplotlib.colors.ListedColormap(['r','b']))

  def plot_weights(w, b, **kwargs):
    w1, w2 = w
    x1s = np.linspace(-1, 1, 100)
    x2s = -(w1  * x1s + b) / w2
    ax.plot(x1s, x2s, **kwargs)

  for w, b in candidate_w_bs:
    plot_weights(w, b,
                 alpha=1./np.sqrt(len(candidate_w_bs)),
                 lw=1, color="blue")

  if true_w_b is not None:
    plot_weights(*true_w_b, lw=4,
                 color="green", label="true separator")

  ax.set_xlim([-1.5, 1.5])
  ax.set_ylim([-1.5, 1.5])
  ax.legend()

  canvas.print_figure(fname, format="png")
  print("saved {}".format(fname))


def build_input_pipeline(x, y, batch_size):
  """Build a Dataset iterator for supervised classification.

  Args:
    x: Numpy `array` of inputs, indexed by the first dimension.
    y: Numpy `array` of labels, with the same first dimension as `x`.
    batch_size: Number of elements in each training batch.

  Returns:
    batch_data: `Tensor` feed  inputs, of shape
      `[batch_size] + x.shape[1:]`.
    batch_labels: `Tensor` feed of labels, of shape
      `[batch_size] + y.shape[1:]`.
  """
  training_dataset = tf.data.Dataset.from_tensor_slices((x, y))
  training_batches = training_dataset.repeat().batch(batch_size)
  training_iterator = training_batches.make_one_shot_iterator()
  batch_data, batch_labels = training_iterator.get_next()
  return batch_data, batch_labels

def logistic_regression(inputs):
  weights = ed.Normal(loc=tf.zeros(inputs.shape[1]), scale=1., name="weights")
  intercept = ed.Normal(loc=0.,scale=1., name="intercept")
  labels_distribution = ed.Bernoulli(
      logits = tf.tensordot(inputs, weights, [[1], [0]]) + intercept,
      name="labels_distribution")
  return labels_distribution

def logistic_regression_with_ones_col(inputs):
  weights = ed.Normal(loc=tf.zeros(inputs.shape[1]), scale=1., name="weights")
  labels_distribution = ed.Bernoulli(
      logits = tf.tensordot(inputs, weights, [[1], [0]]),
      name="labels_distribution")
  return labels_distribution

def variational_posterior(n_features):
  # TODO omitted the regular `initializer=tf.random_normal` bit.

  # TODO for some reason ed.Normal does not support `sample` but this is
  # required for computing KL-divergence. For some reason tfd.Normal does
  # support log_prob (?). What is the difference between `ed.Normal` and
  # `tfd.Normal`?
  #qweights = tfd.Normal(loc=tf.get_variable("weights_loc", [n_features]),
  #                      scale=tfp.trainable_distributions.softplus_and_shift(
  #                        tf.get_variable("weights_scale", [n_features])),
  #                      name="qweights")

  qweights = tfd.Independent(tfd.Normal(loc=tf.get_variable("fooqweights/loc", [n_features]),
                                        scale=tfp.trainable_distributions.softplus_and_shift(
                                          tf.get_variable("fooqweights/scale", [n_features])),
                                        name="qweights"), reinterpreted_batch_ndims=1)

  qintercept = ed.Normal(loc=tf.get_variable("qintercept/loc", []),
                         scale=tfp.trainable_distributions.softplus_and_shift(
                         tf.get_variable("qintercept/scale", [])))
  return qweights, qintercept

def main(argv):
  del argv  # unused

  tf.set_random_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  if tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warning(
        "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.gfile.MakeDirs(FLAGS.model_dir)

  # Generate (and visualize) a toy classification dataset.
  #w_true, b_true, x, y = toy_logistic_data(FLAGS.num_examples, 2)
  w_true, x, y = toy_logistic_data_with_ones_col(FLAGS.num_examples, 2)

  b_true = w_true[-1]
  #visualize_decision(x, y, (w_true, b_true),
  visualize_decision(x, y, (w_true[:-1], b_true),
                     [],
                     fname=os.path.join(FLAGS.model_dir,
                       "weights_inferred.png"))

  #log_joint = ed.make_log_joint_fn(logistic_regression)
  log_joint = ed.make_log_joint_fn(logistic_regression_with_ones_col)
  def target(weights):
    '''closure over observations, computes p(observations, `weights`).'''
    #intercept = ed.Normal(loc=tf.constant(0.), scale=tf.constant(1.))
    #ret = log_joint(inputs=inputs, weights=weights, intercept=intercept, labels_distribution=labels)
    #ret = log_joint(inputs=x, weights=weights, labels_distribution=y)
    ret = log_joint(inputs=inputs, weights=weights, labels_distribution=labels)
    return ret

  def p_log_prob(qsamples):
    '''qsamples is assumed to be [num_draws x 1 x n_features]'''
    return tf.map_fn(target, tf.squeeze(qsamples))

  # Run Frank-Wolfe
  n_fw_iter = 1
  for iter in range(n_fw_iter):
    with tf.Graph().as_default():
      tf.set_random_seed(FLAGS.seed)
      np.random.seed(FLAGS.seed)
      inputs, labels = build_input_pipeline(x, y, FLAGS.batch_size)

      with tf.variable_scope("current_iterate"):
        #n_features = 2 # TODO globalize
        #sweights, sintercept = variational_posterior(n_features)
        n_features = 3 # TODO globalize
        sweights, _ = variational_posterior(n_features)

      train_with_vimco = True
      if not train_with_vimco: # then train with elbo = -loglik + kl
        sweights_samples =  sweights.sample(2)
        # TODO to get the same order of magnitude, divide by num_samples, but is
        # this correct? Ultimately, It doesn't matter.
        neg_log_likelihood = -tf.reduce_mean(p_log_prob(sweights_samples)) / FLAGS.num_examples
        prior = ed.Normal(loc=tf.zeros(inputs.shape[1]), scale=1.)
        kl = tf.reduce_sum(tfd.kl_divergence(sweights.distribution,
                          prior.distribution)) / FLAGS.num_examples
        elbo_loss = neg_log_likelihood + kl
        loss = elbo_loss
      else:
        f = lambda logu: tfp.vi.kl_reverse(logu, self_normalized=False)
        num_draws = 32
        seed = 0
        klqp_vimco = tfp.vi.csiszar_vimco(
            f=f,
            p_log_prob=p_log_prob,
            q=sweights,
            num_draws=num_draws,
            seed=seed)
        loss = klqp_vimco

      # TODO shouldn't need to redefine this in order to get predictions.
      sweights_prime = ed.as_random_variable(sweights) # TODO for some reason multiplication fails.
      logits = tf.tensordot(inputs, sweights_prime, [[1], [0]])
      predictions = tf.cast(logits > 0, dtype=tf.int32)
      accuracy, accuracy_update_op = tf.metrics.accuracy(
          labels=labels, predictions=predictions)

      with tf.name_scope("train"):
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        train_op = opt.minimize(loss)

        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          sess.run(tf.local_variables_initializer())
          for step in range(FLAGS.max_steps):
            sess.run([train_op, accuracy_update_op])
            if step % 100 == 0:
              print("step {:>3d}, loss {:.3f}, accuracy: {:.3f}".format(step, *sess.run([loss, accuracy])))

          w_draw = sweights.sample()
          #b_draw = 0
          candidate_w_bs = []
          for _ in range(FLAGS.num_monte_carlo):
            #w, b = sess.run((w_draw, b_draw))
            w = sess.run((w_draw))
            b = w[-1]
            candidate_w_bs.append((w[:-1], b))

          #visualize_decision(x, y, (w_true, b_true),
          visualize_decision(x, y, (w_true[:-1], b_true),
              candidate_w_bs,
              fname=os.path.join(FLAGS.model_dir,
                "weights_inferred.png"))

      #f = lambda logu: tfp.vi.kl_reverse(logu, self_normalized=False)
      #num_draws = 32
      #seed = 0
      #klqp_vimco = tfp.vi.csiszar_vimco(
      #    f=f,
      #    p_log_prob=p_log_prob,
      #    q=sweights,
      #    num_draws=num_draws,
      #    seed=seed)

      #sweights_prime = ed.as_random_variable(sweights) # TODO for some reason multiplication fails.
      #logits = tf.tensordot(inputs, sweights_prime, [[1], [0]]) # TODO shouldn't need to redefine this
      #predictions = tf.cast(logits > 0, dtype=tf.int32)
      #accuracy, accuracy_update_op = tf.metrics.accuracy(
      #    labels=labels, predictions=predictions)

      #with tf.name_scope("train"):
      #  opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      #  train_op = opt.minimize(klqp_vimco)

      #  with tf.Session() as sess:
      #    sess.run(tf.global_variables_initializer())
      #    sess.run(tf.local_variables_initializer())
      #    for step in range(FLAGS.max_steps):
      #      _ = sess.run([train_op, accuracy_update_op])
      #      if step % 100 == 0:
      #        klqp, acc = sess.run([klqp_vimco, accuracy])
      #        print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
      #            step, klqp, acc))

      #    w_draw = sweights.sample()
      #    #b_draw = 0
      #    candidate_w_bs = []
      #    for _ in range(FLAGS.num_monte_carlo):
      #      #w, b = sess.run((w_draw, b_draw))
      #      w = sess.run((w_draw))
      #      b = w[-1]
      #      candidate_w_bs.append((w[:-1], b))

      #    #visualize_decision(x, y, (w_true, b_true),
      #    visualize_decision(x, y, (w_true[:-1], b_true),
      #        candidate_w_bs,
      #        fname=os.path.join(FLAGS.model_dir,
      #          "weights_inferred.png"))

if __name__ == "__main__":
  tf.app.run()

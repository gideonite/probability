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
             c=np.float32(labels[:, 0]),
             cmap=cm.get_cmap("binary"))

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
      logits= tf.tensordot(inputs, weights, [[1], [0]]) + intercept,
      name="labels_distribution")
  return labels_distribution

def variational_posterior(n_features):
  # TODO omitted the regular `initializer=tf.random_normal` bit.

  # TODO for some reason ed.Normal does not support `sample` but this is
  # required for computing KL-divergence. For some reason tfd.Normal does
  # support sample. What is the difference between `ed.Normal` and
  # `tfd.Normal`?
  qweights = tfd.Normal(loc=tf.get_variable("weights_loc", [n_features]),
                       scale=tfp.trainable_distributions.softplus_and_shift(
                         tf.get_variable("weights_scale", [n_features])), name="qweights")

  qintercept = ed.Normal(loc=tf.get_variable("intercept_loc", []),
                         scale=tfp.trainable_distributions.softplus_and_shift(
                         tf.get_variable("intercept_scale", [])))
  return qweights, qintercept

def main(argv):
  del argv  # unused
  if tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warning(
        "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.gfile.MakeDirs(FLAGS.model_dir)

  # Generate (and visualize) a toy classification dataset.
  w_true, b_true, x, y = toy_logistic_data(FLAGS.num_examples, 2)

  log_joint = ed.make_log_joint_fn(logistic_regression)
  def target(weights):
    # TODO probably need to support evaluating a number of samples of weights,
    # not just a single weights vector. Make another function which uses this
    # target function as part of it.
    intercept = ed.Normal(loc=tf.constant(0.), scale=tf.constant(1.))
    target = log_joint(inputs=inputs, weights=weights, intercept=intercept, labels_distribution=labels)
    return target

  # Run Frank-Wolfe
  for iter in range(3):
    with tf.Graph().as_default():
      inputs, labels = build_input_pipeline(x, y, FLAGS.batch_size)

      with tf.name_scope("logistic_regression", values=[inputs]):
        labels_distribution = logistic_regression(inputs)

      with tf.variable_scope("current_iterate"):
        n_features = 2 # TODO globalize
        sweights, sintercept = variational_posterior(n_features)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        f = lambda logu: tfp.vi.kl_reverse(logu, self_normalized=False)
        num_draws = 32
        seed = 0
        klqp_vimco = tfp.vi.csiszar_vimco(
            f=f,
            p_log_prob=target,
            q=sweights,
            num_draws=num_draws,
            seed=seed)

        #loc = tf.get_variable("loc", [2],
        #        initializer=tf.random_normal_initializer(mean=0., stddev=1.))

        #scale = tf.get_variable("scale", [2],
        #        initializer=tf.random_normal_initializer(mean=0., stddev=1.))

        #q = tfd.Normal(loc=loc, scale=scale)

      ## Compute the -ELBO as the loss, averaged over the batch size.
      ##neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
      ##kl = sum(layer.losses) / FLAGS.num_examples
      ##elbo_loss = neg_log_likelihood + kl

      #def p_log_prob(q_samples):
      #  q_samples = tf.squeeze(q_samples) # TODO num_batch_draws = 1? What does this mean?
      #  logits = tf.matmul(q_samples, tf.transpose(inputs))
      #  bern = tfd.Independent(tfd.Bernoulli(logits=logits), name="bern")
      #  logprobs = bern.log_prob(labels) # for each q sample
      #  logprobs = tf.expand_dims(logprobs, 1) # mimic num_batch_draws = 1
      #  #ret = tf.reduce_mean(logprobs)  # return the expectation over all variational posterior samples
      #  return logprobs

      ## Define the RELBO objective as a VIMCO with a specific `p.log_prob` function
      #f = lambda logu: tfp.vi.kl_reverse(logu, self_normalized=False)
      #num_draws = 32
      #seed = 0
      #klqp_vimco = tfp.vi.csiszar_vimco(
      #    f=f,
      #    p_log_prob=p_log_prob,
      #    q=q,
      #    num_draws=num_draws,
      #    seed=seed)

      ##relbo_loss = elbo_loss + tf.reduce_mean()

      ## Build metrics for evaluation. Predictions are formed from a single forward
      ## pass of the probabilistic layers. They are cheap but noisy predictions.
      #predictions = tf.cast(logits > 0, dtype=tf.int32)
      #accuracy, accuracy_update_op = tf.metrics.accuracy(
      #    labels=labels, predictions=predictions)

      #with tf.name_scope("train"):
      #  opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

      #  #train_op = opt.minimize(elbo_loss)
      #  train_op = opt.minimize(klqp_vimco)
      #  with tf.Session() as sess:
      #    sess.run(tf.global_variables_initializer())
      #    sess.run(tf.local_variables_initializer())

      #    # Fit the model to data.
      #    for step in range(FLAGS.max_steps):
      #      _ = sess.run([train_op, accuracy_update_op])
      #      if step % 100 == 0:
      #        print( sess.run([klqp_vimco, accuracy]) )
      #        #loss_value, accuracy_value = sess.run([elbo_loss, accuracy])
      #        #loss_value, accuracy_value = sess.run([elbo_loss, accuracy])
      #        #print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
      #        #    step, loss_value, accuracy_value))

      #    # Visualize some draws from the weights posterior.
      #    if False:
      #      w_draw = layer.kernel_posterior.sample()
      #      b_draw = layer.bias_posterior.sample()
      #      candidate_w_bs = []
      #      for _ in range(FLAGS.num_monte_carlo):
      #        w, b = sess.run((w_draw, b_draw))
      #        candidate_w_bs.append((w, b))
      #      visualize_decision(x, y, (w_true, b_true),
      #                        candidate_w_bs,
      #                        fname=os.path.join(FLAGS.model_dir,
      #                                            "weights_inferred.png"))

if __name__ == "__main__":
  tf.app.run()

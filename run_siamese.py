# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner for regression tasks
A large portion of the code is adapted from 
https://github.com/zihangdai/xlnet/blob/master/run_classifier.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import collections
import csv

import modeling
import optimization
import tokenization

import tensorflow as tf

import random
import numpy as np

from flow.glow_1x1 import AttrDict, Glow
from flow.glow_init_hook import GlowInitHook
import optimization_bert_flow
import json

from siamese_utils import StsbProcessor, SickRProcessor, MnliProcessor, QqpProcessor, \
    SnliTrainProcessor, SnliDevTestProcessor, \
    Sts_12_16_Processor, MrpcRegressionProcessor, QnliRegressionProcessor, \
    file_based_convert_examples_to_features, file_based_input_fn_builder, \
    get_input_mask_segment

flags = tf.flags

FLAGS = flags.FLAGS

# model
flags.DEFINE_string("bert_config_file", None,
                    "The config json file corresponding to the pre-trained BERT model. "
                    "This specifies the model architecture.")
flags.DEFINE_integer("max_seq_length", 128,
                    "The maximum total input sequence length after WordPiece tokenization. "
                    "Sequences longer than this will be truncated, and sequences shorter "
                    "than this will be padded.")
flags.DEFINE_string("init_checkpoint", None,
                    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool("do_lower_case", True,
                  "Whether to lower case the input text. Should be True for uncased "
                  "models and False for cased models.")  


# task and data
flags.DEFINE_string("task_name", None, "The name of the task to train.")
flags.DEFINE_string("data_dir", None,
                    "The input data dir. Should contain the .tsv files (or other data files) "
                    "for the task.")
flags.DEFINE_float("label_min", 0., None)
flags.DEFINE_float("label_max", 5., None)

# exp
flags.DEFINE_string("output_parent_dir", None, None)
flags.DEFINE_string("exp_name", None, None)
flags.DEFINE_string("exp_name_prefix", None, None)
flags.DEFINE_integer("log_every_step", 10, None)                
flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")
flags.DEFINE_bool("use_xla", False, None)
flags.DEFINE_integer("seed", 1234, None)
flags.DEFINE_string("cached_dir", None,
                    "Path to cached training and dev tfrecord file. "
                    "The file will be generated if not exist.")

# training
flags.DEFINE_bool("do_train", False, None)
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1,
                  "Proportion of training to perform linear learning rate warmup for. "
                  "E.g., 0.1 = 10% of training.")
flags.DEFINE_bool("early_stopping", False, None)

flags.DEFINE_integer("start_delay_secs", 120, "for tf.estimator.EvalSpec")
flags.DEFINE_integer("throttle_secs", 600, "for tf.estimator.EvalSpec")

# eval
flags.DEFINE_bool("do_eval", False, None)
flags.DEFINE_bool("do_predict", False, None)
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
flags.DEFINE_bool("predict_pool", False, None)
flags.DEFINE_bool("do_predict_on_dev", False, None)
flags.DEFINE_bool("do_predict_on_full", False, None)
flags.DEFINE_string("eval_checkpoint_name", None, "filename of a finetuned checkpoint")
flags.DEFINE_bool("auc", False, None)

# sentence embedding related parameters
flags.DEFINE_string("sentence_embedding_type", "avg", "avg, cls, ...")

# flow parameters
flags.DEFINE_integer("flow", 0, "use flow or not")
flags.DEFINE_integer("flow_loss", 0, "use flow loss or not")
flags.DEFINE_float("flow_learning_rate", 1e-3, "The initial learning rate for Adam.")
flags.DEFINE_string("flow_model_config", "config_l3_d3_w32", None)

# unsupervised or semi-supervised related parameters
flags.DEFINE_integer("num_examples", -1, "# of labeled training examples")
flags.DEFINE_integer("use_full_for_training", 0, None)
flags.DEFINE_integer("dupe_factor", 1, "Number of times to duplicate the input data (with different masks).")

# nli related parameters
# flags.DEFINE_integer("use_snli_full", 0, "augment MNLI training data with SNLI")
flags.DEFINE_float("l2_penalty", -1, "penalize l2 norm of sentence embeddings")

# dimension reduction related parameters
flags.DEFINE_integer("low_dim", -1, "avg pooling over the embedding")

# senteval
flags.DEFINE_bool("do_senteval", False, None)
flags.DEFINE_string("senteval_tasks", "", None)

def get_embedding(bert_config, is_training,
    input_ids, input_mask, segment_ids, scope=None):

  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      scope=scope)
  
  if FLAGS.sentence_embedding_type == "avg":
    sequence = model.get_sequence_output() # [batch_size, seq_length, hidden_size]
    input_mask_ = tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=tf.float32)
    pooled = tf.reduce_sum(sequence * input_mask_, axis=1) / tf.reduce_sum(input_mask_, axis=1)
  elif FLAGS.sentence_embedding_type == "cls":
    pooled = model.get_pooled_output()
  elif FLAGS.sentence_embedding_type.startswith("avg-last-last-"):
    pooled = 0
    n_last = int(FLAGS.sentence_embedding_type[-1])
    input_mask_ = tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=tf.float32)
    sequence = model.all_encoder_layers[-n_last] # [batch_size, seq_length, hidden_size]
    pooled += tf.reduce_sum(sequence * input_mask_, axis=1) / tf.reduce_sum(input_mask_, axis=1)
  elif FLAGS.sentence_embedding_type.startswith("avg-last-"):
    pooled = 0
    n_last = int(FLAGS.sentence_embedding_type[-1])
    input_mask_ = tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=tf.float32)
    for i in range(n_last):
      sequence = model.all_encoder_layers[-i] # [batch_size, seq_length, hidden_size]
      pooled += tf.reduce_sum(sequence * input_mask_, axis=1) / tf.reduce_sum(input_mask_, axis=1)
    pooled /= float(n_last)
  elif FLAGS.sentence_embedding_type.startswith("avg-last-concat-"):
    pooled = []
    n_last = int(FLAGS.sentence_embedding_type[-1])
    input_mask_ = tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=tf.float32)
    for i in range(n_last):
      sequence = model.all_encoder_layers[-i] # [batch_size, seq_length, hidden_size]
      pooled += [tf.reduce_sum(sequence * input_mask_, axis=1) / tf.reduce_sum(input_mask_, axis=1)]
    pooled = tf.concat(pooled, axis=-1)
  else:
    raise NotImplementedError

  # flow
  embedding = None
  flow_loss_batch, flow_loss_example = None, None
  if FLAGS.flow:
    # load model and train config
    with open(os.path.join("./flow/config", FLAGS.flow_model_config + ".json"), 'r') as jp:
        flow_model_config = AttrDict(json.load(jp))
    flow_model_config.is_training = is_training
    flow_model = Glow(flow_model_config)
    flow_loss_example = flow_model.body(pooled, is_training) # no l2 normalization here any more
    flow_loss_batch = tf.math.reduce_mean(flow_loss_example)
    embedding = tf.identity(tf.squeeze(flow_model.z, [1, 2])) # no l2 normalization here any more
  else:
    embedding = pooled

  if FLAGS.low_dim > 0:
    bsz, org_dim = modeling.get_shape_list(embedding)
    embedding = tf.reduce_mean(
        tf.reshape(embedding, [bsz, FLAGS.low_dim, org_dim // FLAGS.low_dim]), axis=-1)

  return embedding, flow_loss_batch, flow_loss_example


def create_model(bert_config, is_regression,
                 is_training, 
                 input_ids_a, input_mask_a, segment_ids_a, 
                 input_ids_b, input_mask_b, segment_ids_b, 
                 labels, num_labels):
  """Creates a classification model."""

  with tf.variable_scope("bert") as scope:
    embedding_a, flow_loss_batch_a, flow_loss_example_a = \
        get_embedding(bert_config, is_training, 
          input_ids_a, input_mask_a, segment_ids_a, scope)
  with tf.variable_scope("bert", reuse=tf.AUTO_REUSE) as scope:
    embedding_b, flow_loss_batch_b, flow_loss_example_b = \
        get_embedding(bert_config, is_training,
          input_ids_b, input_mask_b, segment_ids_b, scope)

  with tf.variable_scope("loss"):
    cos_similarity = tf.reduce_sum(tf.multiply(
        tf.nn.l2_normalize(embedding_a, axis=-1), 
        tf.nn.l2_normalize(embedding_b, axis=-1)), axis=-1)
    if is_regression:
      # changing cos_similarity into (cos_similarity + 1)/2.0 
      #     leads to large performance decrease in practice
      per_example_loss = tf.square(cos_similarity - labels) 
      loss = tf.reduce_mean(per_example_loss)
      logits, predictions = None, None
    else:
      output_layer = tf.concat([
        embedding_a, embedding_b, tf.math.abs(embedding_a - embedding_b)
      ], axis=-1)
      output_size = output_layer.shape[-1].value
      output_weights = tf.get_variable(
          "output_weights", [num_labels, output_size],
          initializer=tf.truncated_normal_initializer(stddev=0.02))
      
      logits = tf.matmul(output_layer, output_weights, transpose_b=True)

      probabilities = tf.nn.softmax(logits, axis=-1)
      predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
      loss = tf.reduce_mean(per_example_loss)
    
    if FLAGS.num_examples == 0:
      per_example_loss = tf.zeros_like(per_example_loss)
      loss = tf.zeros_like(loss)
    elif FLAGS.num_examples > 0:
      per_example_loss = per_example_loss * tf.cast(labels > -1, dtype=tf.float32) 
      loss = tf.reduce_mean(per_example_loss)

    if FLAGS.l2_penalty > 0:
      l2_penalty_loss = tf.norm(embedding_a, axis=-1, keepdims=False)
      l2_penalty_loss += tf.norm(embedding_b, axis=-1, keepdims=False)
      l2_penalty_loss *= FLAGS.l2_penalty

      per_example_loss += l2_penalty_loss
      loss += tf.reduce_mean(l2_penalty_loss)

  model_output = {
      "loss": loss,
      "per_example_loss": per_example_loss,
      "cos_similarity": cos_similarity,
      "embedding_a": embedding_a,
      "embedding_b": embedding_b,
      "logits": logits,
      "predictions": predictions,
  }

  if FLAGS.flow:
    model_output["flow_example_loss"] = flow_loss_example_a + flow_loss_example_b
    model_output["flow_loss"] = flow_loss_batch_a + flow_loss_batch_b

  return model_output


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, is_regression):
  """Returns `model_fn` closure for Estimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for Estimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids_a = features["input_ids_a"]
    input_mask_a = features["input_mask_a"]
    segment_ids_a = features["segment_ids_a"]

    input_ids_b = features["input_ids_b"]
    input_mask_b = features["input_mask_b"]
    segment_ids_b = features["segment_ids_b"]

    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    #### Training or Evaluation
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    #### Get loss from inputs
    model_output = create_model(
        bert_config, is_regression,
        is_training, 
        input_ids_a, input_mask_a, segment_ids_a, 
        input_ids_b, input_mask_b, segment_ids_b, 
        label_ids,
        num_labels)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
      # if "flow" in var.name:
      #   input()

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      if FLAGS.flow_loss:
        train_op = optimization_bert_flow.create_optimizer(
            model_output["loss"], model_output["flow_loss"], 
            learning_rate, FLAGS.flow_learning_rate, 
            num_train_steps, num_warmup_steps, use_tpu=False)
        tf.summary.scalar("loss", model_output["loss"])
        tf.summary.scalar("flow_loss", model_output["flow_loss"])

        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=model_output["loss"] + model_output["flow_loss"],
            train_op=train_op)
      else:
        train_op = optimization.create_optimizer(
            model_output["loss"], learning_rate, 
            num_train_steps, num_warmup_steps, use_tpu=False)
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=model_output["loss"],
            train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(model_output, label_ids, is_real_example):
        predictions = tf.argmax(model_output["logits"], axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=model_output["predictions"],
            weights=is_real_example)
        loss = tf.metrics.mean(
            values=model_output["per_example_loss"], weights=is_real_example)
        metric_output = {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

        if "flow_loss" in model_output:
          metric_output["eval_loss_flow"] = \
            tf.metrics.mean(values=model_output["flow_example_loss"], weights=is_real_example)
          metric_output["eval_loss_total"] = \
            tf.metrics.mean(
              values=model_output["per_example_loss"] + model_output["flow_example_loss"], 
              weights=is_real_example)
        
        return metric_output

      def regression_metric_fn(model_output, label_ids, is_real_example):
        metric_output = {
          "eval_loss": tf.metrics.mean(
            values=model_output["per_example_loss"], weights=is_real_example),
          "eval_pearsonr": tf.contrib.metrics.streaming_pearson_correlation(
              model_output["cos_similarity"], label_ids, weights=is_real_example)
        }

        # metric_output["auc"] = tf.compat.v1.metrics.auc(
        #   label_ids, tf.math.maximum(model_output["cos_similarity"], 0), weights=is_real_example, curve='ROC')

        if "flow_loss" in model_output:
          metric_output["eval_loss_flow"] = \
            tf.metrics.mean(values=model_output["flow_example_loss"], weights=is_real_example)
          metric_output["eval_loss_total"] = \
            tf.metrics.mean(
              values=model_output["per_example_loss"] + model_output["flow_example_loss"], 
              weights=is_real_example)
        
        return metric_output

      if is_regression:
        metric_fn = regression_metric_fn

      eval_metrics = metric_fn(model_output, label_ids, is_real_example)
      
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=model_output["loss"],
          eval_metric_ops=eval_metrics)
    else:
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions= {"embedding_a": model_output["embedding_a"], 
                        "embedding_b": model_output["embedding_b"]} if FLAGS.predict_pool else \
                       {"cos_similarity": model_output["cos_similarity"]})
    return output_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # random seed
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  tf.compat.v1.set_random_seed(FLAGS.seed)
  print("FLAGS.seed", FLAGS.seed)
  # input()

  # prevent double printing of the tf logs
  logger = tf.get_logger()
  logger.propagate = False

  # get tokenizer
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # get bert config
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  # GPU config
  run_config = tf.compat.v1.ConfigProto()
  if FLAGS.use_xla:
    run_config.graph_options.optimizer_options.global_jit_level = \
        tf.OptimizerOptions.ON_1

  run_config.gpu_options.allow_growth = True

  if FLAGS.do_senteval:
    # Set up logger
    import logging
    tf.logging.set_verbosity(0)
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    # load senteval
    import sys
    PATH_TO_SENTEVAL, PATH_TO_DATA = '../SentEval', '../SentEval/data'
    sys.path.insert(0, PATH_TO_SENTEVAL)
    import senteval

    # model
    tf.logging.info("***** Running SentEval *****")
    with tf.Graph().as_default():
      with tf.variable_scope("bert") as scope:
        input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_ids")
        input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_mask")
        segment_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="segment_ids")

        embedding, flow_loss_batch, flow_loss_example = \
            get_embedding(bert_config, False,
              input_ids, input_mask, segment_ids, scope=scope)
        embedding = tf.nn.l2_normalize(embedding, axis=-1)
      
      tvars = tf.trainable_variables()
      initialized_variable_names = {}
      if FLAGS.init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
        tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

      tf.logging.info("**** Trainable Variables ****")
      for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
    
      with tf.train.MonitoredSession(
          session_creator=tf.compat.v1.train.ChiefSessionCreator(config=run_config)) as session:

        # SentEval prepare and batcher
        def prepare(params, samples):
          return

        def batcher(params, batch):
          batch_input_ids, batch_input_mask, batch_segment_ids = [], [], []
          for sent in batch:
            if type(sent[0]) == bytes:
              sent = [_.decode() for _ in sent]
            text = ' '.join(sent) if sent != [] else '.'
            # print(text)

            _input_ids, _input_mask, _segment_ids, _tokens = \
                get_input_mask_segment(text, FLAGS.max_seq_length, tokenizer)
            batch_input_ids.append(_input_ids)
            batch_input_mask.append(_input_mask)
            batch_segment_ids.append(_segment_ids)
          
          batch_input_ids = np.asarray(batch_input_ids)
          batch_input_mask = np.asarray(batch_input_mask)
          batch_segment_ids = np.asarray(batch_segment_ids)

          print(".", end="")

          return session.run(embedding, 
              {input_ids: batch_input_ids,
              input_mask: batch_input_mask,
              segment_ids: batch_segment_ids})

        # Set params for SentEval
        params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                        'tenacity': 3, 'epoch_size': 2}

        # main
        se = senteval.engine.SE(params_senteval, batcher, prepare)

        # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
        #                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
        #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
        #                   'Length', 'WordContent', 'Depth', 'TopConstituents',
        #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
        #                   'OddManOut', 'CoordinationInversion']
        #transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        #transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
        transfer_tasks = FLAGS.senteval_tasks.split(",")
        results = se.eval(transfer_tasks)
        from collections import OrderedDict
        results = OrderedDict(results)
        for key in sorted(results):
          value = results[key]
          if key.startswith("STS"):
            print("'" + key + "':", value["all"])
          else:
            print(key, value)

    return 

  processors = {
      'sts-b': StsbProcessor,
      'sick-r': SickRProcessor,
      'mnli': MnliProcessor,
      'allnli': MnliProcessor,
      'qqp': QqpProcessor,
      'sts-12-16': Sts_12_16_Processor,
      'sts-12': Sts_12_16_Processor,
      'sts-13': Sts_12_16_Processor,
      'sts-14': Sts_12_16_Processor,
      'sts-15': Sts_12_16_Processor,
      'sts-16': Sts_12_16_Processor,
      'mrpc-regression': MrpcRegressionProcessor,
      'qnli-regression': QnliRegressionProcessor,
  }

  task_name = FLAGS.task_name.lower()
  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  if task_name == 'sick-r' or task_name.startswith("sts"):
    is_regression = True
    label_min, label_max = 0., 5.
  elif task_name in ['qqp', 'mrpc-regression', 'qnli-regression']:
    is_regression = True
    label_min, label_max = 0., 1.
  else:
    is_regression = False
    label_min, label_max = 0., 1.

  dupe_factor = FLAGS.dupe_factor

  processor = processors[task_name]()

  label_list = processor.get_labels()

  # this block is moved here for calculating the epoch_step for save_checkpoints_steps
  train_examples = None
  num_train_steps = None
  num_warmup_steps = None

  if task_name == "allnli":
      FLAGS.data_dir = os.path.join(os.path.dirname(FLAGS.data_dir), "MNLI")

  if FLAGS.do_train and FLAGS.num_train_epochs > 1e-6:
    train_examples = processor.get_train_examples(FLAGS.data_dir)

    if task_name == "allnli":
      snli_data_dir = os.path.join(os.path.dirname(FLAGS.data_dir), "SNLI")
      train_examples.extend(SnliTrainProcessor().get_train_examples(snli_data_dir))
      train_examples.extend(SnliDevTestProcessor().get_dev_examples(snli_data_dir))
      train_examples.extend(SnliDevTestProcessor().get_test_examples(snli_data_dir))

    if FLAGS.use_full_for_training:
      eval_examples = processor.get_dev_examples(FLAGS.data_dir)
      predict_examples = processor.get_test_examples(FLAGS.data_dir)
      train_examples.extend(eval_examples + predict_examples)

    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    epoch_step = int(len(train_examples) / FLAGS.train_batch_size)

    if FLAGS.num_examples > 0:
      random.shuffle(train_examples)
      for i in range(FLAGS.num_examples, len(train_examples)):
        train_examples[i].label = -10
    
    random.shuffle(train_examples)


  # ==== #

  if FLAGS.early_stopping:
    save_checkpoints_steps = epoch_step
  else:
    save_checkpoints_steps = FLAGS.save_checkpoints_steps

  keep_checkpoint_max = 3
  save_summary_steps = log_every_step = FLAGS.log_every_step 

  tf.logging.info("save_checkpoints_steps: %d" % save_checkpoints_steps)

  # make exp dir
  if FLAGS.exp_name:  
    output_dir = os.path.join(FLAGS.output_parent_dir, FLAGS.exp_name)
  elif FLAGS.exp_name_prefix:
    output_dir = os.path.join(FLAGS.output_parent_dir, FLAGS.exp_name_prefix)

    output_dir += "_t_%s" % (FLAGS.task_name)
    output_dir += "_ep_%.2f" % (FLAGS.num_train_epochs)
    output_dir += "_lr_%.2e" % (FLAGS.learning_rate)

    if FLAGS.train_batch_size != 32:
      output_dir += "_bsz_%d" % (FLAGS.train_batch_size)

    if FLAGS.sentence_embedding_type != "avg":
      output_dir += "_e_%s" % (FLAGS.sentence_embedding_type)
    
    if FLAGS.flow > 0:
      output_dir += "_f_%d%d" % (FLAGS.flow, FLAGS.flow_loss)

      if FLAGS.flow_loss > 0:
        output_dir += "_%.2e" % (FLAGS.flow_learning_rate)

      if FLAGS.use_full_for_training > 0:
        output_dir += "_allsplits"
  
      if FLAGS.flow_model_config != "config_l3_d3_w32":
        output_dir += "_%s" % (FLAGS.flow_model_config)
    
    if FLAGS.num_examples > 0:
      output_dir += "_n_%d" % (FLAGS.num_examples)
    
    if FLAGS.low_dim > -1:
      output_dir += "_ld_%d" % (FLAGS.low_dim)

    if FLAGS.l2_penalty > 0:
      output_dir += "_l2_%.2e" % (FLAGS.l2_penalty)

  else:
    raise NotImplementedError
  
  if tf.gfile.Exists(output_dir) and FLAGS.do_train:
    tf.io.gfile.rmtree(output_dir)
  tf.gfile.MakeDirs(output_dir)

  # set up estimator
  run_config = tf.estimator.RunConfig(
      model_dir=output_dir,
      save_summary_steps=save_summary_steps,
      save_checkpoints_steps=save_checkpoints_steps,
      keep_checkpoint_max=keep_checkpoint_max,
      log_step_count_steps=log_every_step,
      session_config=run_config)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      is_regression=is_regression)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={
        'train_batch_size': FLAGS.train_batch_size,
        'eval_batch_size': FLAGS.eval_batch_size,
        'predict_batch_size': FLAGS.predict_batch_size})

  def get_train_input_fn():
    cached_dir = FLAGS.cached_dir
    if not cached_dir:
      cached_dir = output_dir

    data_name = task_name

    if FLAGS.num_examples > 0:
      train_file = os.path.join(cached_dir, 
        data_name + "_n_%d" % (FLAGS.num_examples) \
          + "_seed_%d" % (FLAGS.seed) +  "_train.tf_record")
    elif FLAGS.use_full_for_training > 0:
      train_file = os.path.join(cached_dir, data_name + "_allsplits.tf_record")
    else:
      train_file = os.path.join(cached_dir, data_name + "_train.tf_record")

    if not tf.gfile.Exists(train_file):
      file_based_convert_examples_to_features(
          train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, 
          dupe_factor, label_min, label_max, 
          is_training=True)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
  
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        is_regression=is_regression)
    
    return train_input_fn

  def get_eval_input_fn():
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)

    cached_dir = FLAGS.cached_dir
    if not cached_dir:
      cached_dir = output_dir
    eval_file = os.path.join(cached_dir, task_name + "_eval.tf_record")

    if not tf.gfile.Exists(eval_file):
      file_based_convert_examples_to_features(
          eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file,
          dupe_factor, label_min, label_max)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_drop_remainder = False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder,
        is_regression=is_regression)

    return eval_input_fn
  
  def get_predict_input_fn():
    predict_examples = None
    if FLAGS.do_predict_on_dev:
      predict_examples = processor.get_dev_examples(FLAGS.data_dir)
    elif FLAGS.do_predict_on_full:
      train_examples = processor.get_train_examples(FLAGS.data_dir)
      eval_examples = processor.get_dev_examples(FLAGS.data_dir)
      predict_examples = processor.get_test_examples(FLAGS.data_dir)
      predict_examples.extend(eval_examples + train_examples)
    else:
      predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)

    cached_dir = FLAGS.cached_dir
    if not cached_dir:
      cached_dir = output_dir
    predict_file = os.path.join(cached_dir, task_name + "_predict.tf_record")
    
    file_based_convert_examples_to_features(
        predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file, 
        dupe_factor, label_min, label_max)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder,
        is_regression=is_regression)

    return predict_input_fn, num_actual_predict_examples

  eval_steps = None

  if FLAGS.do_train and FLAGS.num_train_epochs > 1e-6:
    train_input_fn = get_train_input_fn()
    if FLAGS.early_stopping:
      eval_input_fn = get_eval_input_fn()
      early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
        estimator, metric_name="eval_pearsonr", 
        max_steps_without_decrease=epoch_step//2, run_every_steps=epoch_step, run_every_secs=None)
      train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, 
                                          hooks=[early_stopping_hook])

      start_delay_secs = FLAGS.start_delay_secs
      throttle_secs = FLAGS.throttle_secs
      tf.logging.info("start_delay_secs: %d; throttle_secs: %d" % (start_delay_secs, throttle_secs))
      eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps,
        start_delay_secs=start_delay_secs, throttle_secs=throttle_secs)
      tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    else:
      estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_input_fn = get_eval_input_fn()
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    
    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_input_fn, num_actual_predict_examples = get_predict_input_fn()
    checkpoint_path = None
    if FLAGS.eval_checkpoint_name:
      checkpoint_path = os.path.join(output_dir, FLAGS.eval_checkpoint_name)
    result = estimator.predict(input_fn=predict_input_fn,
                               checkpoint_path=checkpoint_path)

    def round_float_list(values):
      values = [round(float(x), 6) for x in values.flat]
      return values
    
    fname = ""
    if FLAGS.do_predict_on_full:
      fname += "full"
    elif FLAGS.do_predict_on_dev:
      fname += "dev"
    else:
      fname += "test"

    if FLAGS.predict_pool:
      fname += "_pooled.tsv"
    else:
      fname += "_results.tsv"

    if FLAGS.eval_checkpoint_name:
      fname = FLAGS.eval_checkpoint_name + "." + fname
    output_predict_file = os.path.join(output_dir, fname)
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):

        if is_regression:
          if FLAGS.predict_pool:
            embedding_a = prediction["embedding_a"]
            embedding_b = prediction["embedding_b"]

            output_json = collections.OrderedDict()
            output_json["embedding_a"] = round_float_list(embedding_a)
            output_json["embedding_b"] = round_float_list(embedding_b)

            output_line = json.dumps(output_json) + "\n"
          else:
            cos_similarity = prediction["cos_similarity"]
            if i >= num_actual_predict_examples:
              break
            output_line = str(cos_similarity) + "\n"
        else:
          raise NotImplementedError

        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples
  
  tf.logging.info("*** output_dir ***")
  tf.logging.info(output_dir)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  tf.app.run()

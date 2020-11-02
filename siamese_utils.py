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


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids_a,
               input_mask_a,
               segment_ids_a,
               input_ids_b,
               input_mask_b,
               segment_ids_b,
               label_id,
               is_real_example=True):
    self.input_ids_a = input_ids_a
    self.input_mask_a = input_mask_a
    self.segment_ids_a = segment_ids_a
    self.input_ids_b = input_ids_b
    self.input_mask_b = input_mask_b
    self.segment_ids_b = segment_ids_b
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class GLUEProcessor(DataProcessor):
  def __init__(self):
    self.train_file = "train.tsv"
    self.dev_file = "dev.tsv"
    self.test_file = "test.tsv"
    self.label_column = None
    self.text_a_column = None
    self.text_b_column = None
    self.contains_header = True
    self.test_text_a_column = None
    self.test_text_b_column = None
    self.test_contains_header = True

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.train_file)), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.dev_file)), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    if self.test_text_a_column is None:
      self.test_text_a_column = self.text_a_column
    if self.test_text_b_column is None:
      self.test_text_b_column = self.text_b_column

    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.test_file)), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0 and self.contains_header and set_type != "test":
        continue
      if i == 0 and self.test_contains_header and set_type == "test":
        continue
      guid = "%s-%s" % (set_type, i)

      a_column = (self.text_a_column if set_type != "test" else
          self.test_text_a_column)
      b_column = (self.text_b_column if set_type != "test" else
          self.test_text_b_column)

      # there are some incomplete lines in QNLI
      if len(line) <= a_column:
        tf.logging.warning('Incomplete line, ignored.')
        continue
      text_a = line[a_column]

      if b_column is not None:
        if len(line) <= b_column:
          tf.logging.warning('Incomplete line, ignored.')
          continue
        text_b = line[b_column]
      else:
        text_b = None

      if set_type == "test":
        label = self.get_labels()[0]
      else:
        if len(line) <= self.label_column:
          tf.logging.warning('Incomplete line, ignored.')
          continue
        label = line[self.label_column]
        if len(label) == 0:
          raise Exception
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class StsbProcessor(GLUEProcessor):
  def __init__(self):
    super(StsbProcessor, self).__init__()
    self.label_column = 9
    self.text_a_column = 7
    self.text_b_column = 8

  def get_labels(self):
    return [0.]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0 and self.contains_header and set_type != "test":
        continue
      if i == 0 and self.test_contains_header and set_type == "test":
        continue
      guid = "%s-%s" % (set_type, i)

      a_column = (self.text_a_column if set_type != "test" else
          self.test_text_a_column)
      b_column = (self.text_b_column if set_type != "test" else
          self.test_text_b_column)

      # there are some incomplete lines in QNLI
      if len(line) <= a_column:
        tf.logging.warning('Incomplete line, ignored.')
        continue
      text_a = line[a_column]

      if b_column is not None:
        if len(line) <= b_column:
          tf.logging.warning('Incomplete line, ignored.')
          continue
        text_b = line[b_column]
      else:
        text_b = None

      if set_type == "test":
        label = self.get_labels()[0]
      else:
        if len(line) <= self.label_column:
          tf.logging.warning('Incomplete line, ignored.')
          continue
        label = float(line[self.label_column])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples


class SickRProcessor(StsbProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""
  def __init__(self):
    super(SickRProcessor, self).__init__()
    self.train_file = "SICK_train.txt"
    self.dev_file = "SICK_trial.txt"
    self.test_file = "SICK_test_annotated.txt"
    self.label_column = 3
    self.text_a_column = 1
    self.text_b_column = 2
    self.contains_header = True
    self.test_text_a_column = None
    self.test_text_b_column = None
    self.test_contains_header = True

class Sts_12_16_Processor(GLUEProcessor):
  def __init__(self):
    super(Sts_12_16_Processor, self).__init__()
    self.train_file = "full.txt"
    self.dev_file = "full.txt"
    self.test_file = "full.txt"
    self.text_a_column = 0
    self.text_b_column = 1

  def get_labels(self):
    return [0.]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = line[self.text_a_column]
      text_b = line[self.text_b_column]
      label = self.get_labels()[0]
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples


class QqpProcessor(GLUEProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""
  def __init__(self):
    super(QqpProcessor, self).__init__()
    self.train_file = "train.tsv"
    self.dev_file = "dev.tsv"
    self.test_file = "test.tsv"
    self.label_column = 5
    self.text_a_column = 3
    self.text_b_column = 4
    self.contains_header = True
    self.test_text_a_column = 1
    self.test_text_b_column = 2
    self.test_contains_header = True

  def get_labels(self):
    """See base class."""
    return [0.]


class MrpcRegressionProcessor(StsbProcessor):
  def __init__(self):
    super(MrpcRegressionProcessor, self).__init__()
    self.label_column = 0
    self.text_a_column = 3
    self.text_b_column = 4


class QnliRegressionProcessor(GLUEProcessor):
  def __init__(self):
    super(QnliRegressionProcessor, self).__init__()
    self.label_column = -1
    self.text_a_column = 1
    self.text_b_column = 2

  def get_labels(self):
    return [0.]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0 and self.contains_header and set_type != "test":
        continue
      if i == 0 and self.test_contains_header and set_type == "test":
        continue
      guid = "%s-%s" % (set_type, i)

      a_column = (self.text_a_column if set_type != "test" else
          self.test_text_a_column)
      b_column = (self.text_b_column if set_type != "test" else
          self.test_text_b_column)

      # there are some incomplete lines in QNLI
      if len(line) <= a_column:
        tf.logging.warning('Incomplete line, ignored.')
        continue
      text_a = line[a_column]

      if b_column is not None:
        if len(line) <= b_column:
          tf.logging.warning('Incomplete line, ignored.')
          continue
        text_b = line[b_column]
      else:
        text_b = None

      if set_type == "test":
        label = self.get_labels()[0]
      else:
        if len(line) <= self.label_column:
          tf.logging.warning('Incomplete line, ignored.')
          continue
        label_score_map = { "not_entailment": 0, "entailment": 1 }
        label = label_score_map[line[self.label_column]]
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples


class MnliProcessor(GLUEProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""
  def __init__(self):
    super(MnliProcessor, self).__init__()
    self.train_file = "train.tsv"
    self.dev_file = "dev_matched.tsv"
    self.test_file = "test_matched.tsv"
    self.label_column = 10
    self.text_a_column = 8
    self.text_b_column = 9
    self.contains_header = True
    self.test_text_a_column = None
    self.test_text_b_column = None
    self.test_contains_header = True

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class SnliTrainProcessor(GLUEProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""
  def __init__(self):
    super(SnliTrainProcessor, self).__init__()
    self.train_file = "train.tsv"
    self.dev_file = "dev.tsv"
    self.test_file = "test.tsv"
    self.label_column = -1
    self.text_a_column = 7
    self.text_b_column = 8
    self.contains_header = True

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

class SnliDevTestProcessor(SnliTrainProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""
  def __init__(self):
    super(SnliDevTestProcessor, self).__init__()
    self.label_column = -1

def get_input_mask_segment(text,
      max_seq_length, tokenizer, random_mask=0):
  tokens = tokenizer.tokenize(text)

  # Account for [CLS] and [SEP] with "- 2"
  if len(tokens) > max_seq_length - 2:
    tokens = tokens[0:(max_seq_length - 2)]

  if random_mask:
    tokens[random.randint(0, len(tokens)-1)] = "[MASK]"

  tokens = ["[CLS]"] + tokens + ["[SEP]"]
  segment_ids = [0 for _ in tokens]
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  return (input_ids, input_mask, segment_ids, tokens)

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, random_mask=0):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i
    
  input_ids_a, input_mask_a, segment_ids_a, tokens_a = \
      get_input_mask_segment(example.text_a, max_seq_length, tokenizer, random_mask)
  input_ids_b, input_mask_b, segment_ids_b, tokens_b = \
      get_input_mask_segment(example.text_b, max_seq_length, tokenizer, random_mask)

  if len(label_list) > 1:
    label_id = label_map[example.label]
  else:
    label_id = example.label

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens_a: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens_a]))
    tf.logging.info("tokens_b: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens_b]))
    tf.logging.info("input_ids_a: %s" % " ".join([str(x) for x in input_ids_a]))
    tf.logging.info("input_mask_a: %s" % " ".join([str(x) for x in input_mask_a]))
    tf.logging.info("segment_ids_a: %s" % " ".join([str(x) for x in segment_ids_a]))
    tf.logging.info("input_ids_b: %s" % " ".join([str(x) for x in input_ids_b]))
    tf.logging.info("input_mask_b: %s" % " ".join([str(x) for x in input_mask_b]))
    tf.logging.info("segment_ids_b: %s" % " ".join([str(x) for x in segment_ids_b]))
    tf.logging.info("label: %s (id = %s)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids_a=input_ids_a,
      input_mask_a=input_mask_a,
      segment_ids_a=segment_ids_a,
      input_ids_b=input_ids_b,
      input_mask_b=input_mask_b,
      segment_ids_b=segment_ids_b,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file, 
    dupe_factor, label_min, label_max, is_training=False):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    for t in range(dupe_factor if is_training else 1):
      feature = convert_single_example(ex_index, example, label_list,
                                      max_seq_length, tokenizer, 
                                      random_mask=0 if t == 0 else 1)

      def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

      def create_float_feature(values):
        f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        return f

      features = collections.OrderedDict()
      features["input_ids_a"] = create_int_feature(feature.input_ids_a)
      features["input_mask_a"] = create_int_feature(feature.input_mask_a)
      features["segment_ids_a"] = create_int_feature(feature.segment_ids_a)

      features["input_ids_b"] = create_int_feature(feature.input_ids_b)
      features["input_mask_b"] = create_int_feature(feature.input_mask_b)
      features["segment_ids_b"] = create_int_feature(feature.segment_ids_b)

      if len(label_list) > 1:
        features["label_ids"] = create_int_feature([feature.label_id])
      else:
        features["label_ids"] = create_float_feature(
          [(float(feature.label_id) - label_min) / (label_max - label_min)])
      features["is_real_example"] = create_int_feature(
          [int(feature.is_real_example)])

      tf_example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())
      
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, is_regression):
  """Creates an `input_fn` closure to be passed to Estimator."""

  name_to_features = {
      "input_ids_a": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask_a": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids_a": tf.FixedLenFeature([seq_length], tf.int64),
      "input_ids_b": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask_b": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids_b": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  if is_regression:
    name_to_features["label_ids"] = tf.FixedLenFeature([], tf.float32)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(mode, params):
    """The actual input function."""
    if mode == tf.estimator.ModeKeys.TRAIN:
      batch_size = params["train_batch_size"]
    elif mode == tf.estimator.ModeKeys.EVAL:
      batch_size = params["eval_batch_size"]
    elif mode == tf.estimator.ModeKeys.PREDICT:
      batch_size = params["predict_batch_size"]
    else:
      raise NotImplementedError

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    # if is_training:
    #   d = d.repeat()
    #   d = d.shuffle(buffer_size=100)

    if is_training:
      d = d.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
      d = d.repeat()

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn



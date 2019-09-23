# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from . import modeling
from . import optimization
from . import tokenization
import tensorflow as tf
import numpy as np
import requests
# import datetime
# import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

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

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
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

class polarityProcessor(DataProcessor):
    # read txt
    # 返回InputExample类组成的list
    # text_a是一串字符串，text_b则是另一串字符串。在进行后续输入处理后(BERT代码中已包含，不需要自己完成)
    # text_a和text_b将组合成[CLS] text_a [SEP] text_b [SEP]的形式传入模型
    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'project_train.txt')
        f = open(file_path, 'r', encoding='utf-8')
        train_data = []
        index = 0
        for line in f.readlines():
            guid = 'train-%d' % index  # 参数guid是用来区分每个example的
            line = line.replace("\n", "").split("\t")
            text_a = tokenization.convert_to_unicode(str(line[1]))  # 要分类的文本
            label = str(line[2])  # 文本对应的情感类别
            train_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))  # 加入到InputExample列表中
            index += 1
        return train_data

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'project_dev.txt')
        f = open(file_path, 'r', encoding='utf-8')
        dev_data = []
        index = 0
        for line in f.readlines():
            guid = 'dev-%d' % index
            line = line.replace("\n", "").split("\t")
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = str(line[2])
            dev_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            index += 1
        return dev_data

    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'prediction_origin.txt')
        f = open(file_path, 'r', encoding='utf-8')
        test_data = []
        index = 0
        for line in f.readlines():
            guid = 'test-%d' % index
            line = line.replace("\n", "").split("\t")
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = str(line[2])
            test_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            index += 1
        return test_data

    def get_labels(self):
        return ["pos", "neg"]

class tripleProcessor(DataProcessor):
    # read txt
    # 返回InputExample类组成的list
    # text_a是一串字符串，text_b则是另一串字符串。在进行后续输入处理后(BERT代码中已包含，不需要自己完成)
    # text_a和text_b将组合成[CLS] text_a [SEP] text_b [SEP]的形式传入模型
    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'prediction_origin.txt')
        f = open(file_path, 'r', encoding='utf-8')
        test_data = []
        index = 0
        for line in f.readlines():
            guid = 'test-%d' % index
            line = line.replace("\n", "").split("\t")
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = str(line[2])
            test_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            index += 1
        return test_data

    def get_labels(self):
        return ["愤怒","厌恶","低落"]

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

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

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature

def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

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

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features

def run_classifier(task_name, init_checkpoint, prediction_arr):

    processors = {
        "polarity": polarityProcessor,
        "triple": tripleProcessor
    }

    basic_dir = "sentiment_bert/Bert/"

    data_dir = basic_dir+"roject_data",
    bert_config_file = basic_dir+"chinese_L-12_H-768_A-12/bert_config.json",
    vocab_file = basic_dir+"chinese_L-12_H-768_A-12/vocab.txt",
    output_dir = basic_dir+"output",
    do_lower_case = True,
    max_seq_length = 128,
    do_train = False,
    do_eval =False,
    do_predict = True,
    train_batch_size =32,
    eval_batch_size = 8,
    predict_batch_size =8,
    learning_rate = 5e-5,
    num_train_epochs = 3.0,
    warmup_proportion = 0.1,
    save_checkpoints_steps = 1000,
    iterations_per_loop = 1000,
    use_tpu = False,
    tpu_name = None,
    tpu_zone = None,
    gcp_project = None,
    master = None,
    num_tpu_cores = 8,

    tokenization.validate_case_matches_checkpoint(do_lower_case[0],
                                                  basic_dir+init_checkpoint)

    if not do_train[0] and not do_eval[0] and not do_predict[0]:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(bert_config_file[0])

    # if max_seq_length > bert_config.max_position_embeddings:
    #     raise ValueError(
    #         "Cannot use sequence length %d because the BERT model "
    #         "was only trained up to sequence length %d" %
    #         (max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(output_dir[0])

    task_name = task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file[0], do_lower_case=do_lower_case[0])

    tpu_cluster_resolver = None
    if use_tpu[0] and tpu_name[0]:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_name[0], zone=tpu_zone[0], project=gcp_project[0])

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=master[0],
        model_dir=output_dir[0],
        save_checkpoints_steps=save_checkpoints_steps[0],
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop[0],
            num_shards=num_tpu_cores[0],
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if do_train[0]:
        train_examples = processor.get_train_examples(data_dir[0])
        num_train_steps = int(
            len(train_examples) / train_batch_size[0] * num_train_epochs[0])
        num_warmup_steps = int(num_train_steps * warmup_proportion[0])

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=basic_dir+init_checkpoint,
        learning_rate=learning_rate[0],
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=use_tpu[0],
        use_one_hot_embeddings=use_tpu[0])

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu[0],
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size[0],
        eval_batch_size=eval_batch_size[0],
        predict_batch_size=predict_batch_size[0])

    if do_train[0]:
        train_file = os.path.join(output_dir[0], "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, max_seq_length[0], tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", train_batch_size[0])
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=max_seq_length[0],
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if do_eval[0]:
        eval_examples = processor.get_dev_examples(data_dir[0])
        num_actual_eval_examples = len(eval_examples)
        if use_tpu[0]:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % eval_batch_size[0] != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(output_dir[0], "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, max_seq_length[0], tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", eval_batch_size[0])

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if use_tpu[0]:
            assert len(eval_examples) % eval_batch_size[0] == 0
            eval_steps = int(len(eval_examples) // eval_batch_size[0])

        eval_drop_remainder = True if use_tpu[0] else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=max_seq_length[0],
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(output_dir[0], "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if do_predict[0]:

        # idx = 0
        # predict_examples = []
        # for line in prediction_arr:
        #     guid = 'dev-%d' % idx
        #     text_a = tokenization.convert_to_unicode(str(line))
        #     label = str(label_list[0])
        #     predict_examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        #     idx += 1
        #
        # #predict_examples = processor.get_test_examples(FLAGS.data_dir[0])
        # num_actual_predict_examples = len(predict_examples)
        # if use_tpu[0]:
        #     # TPU requires a fixed batch size for all batches, therefore the number
        #     # of examples must be a multiple of the batch size, or else examples
        #     # will get dropped. So we pad with fake examples which are ignored
        #     # later on.
        #     while len(predict_examples) % predict_batch_size[0] != 0:
        #         predict_examples.append(PaddingInputExample())
        #
        # predict_file = os.path.join(output_dir[0], "predict.tf_record")
        # file_based_convert_examples_to_features(predict_examples, label_list,
        #                                         max_seq_length[0], tokenizer,
        #                                         predict_file)
        # # tf.logging.info("***** Running prediction*****")
        # # tf.logging.info("  Num examples = %d (%d actual, %d padding)",
        # #                 len(predict_examples), num_actual_predict_examples,
        # #                 len(predict_examples) - num_actual_predict_examples)
        # # tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size[0])
        #
        # predict_drop_remainder = True if use_tpu[0] else False
        # predict_input_fn = file_based_input_fn_builder(
        #     input_file=predict_file,
        #     seq_length=max_seq_length[0],
        #     is_training=False,
        #     drop_remainder=predict_drop_remainder)
        #
        # result = estimator.predict(input_fn=predict_input_fn)
        # output_predict_file = os.path.join(output_dir[0], "test_results.tsv")
        # final_results = []
        # with tf.gfile.GFile(output_predict_file, "w") as writer:
        #     num_written_lines = 0
        #     tf.logging.info("***** Predict results *****")
        #     for (i, prediction) in enumerate(result):
        #         probabilities = prediction["probabilities"].tolist()
        #         if i >= num_actual_predict_examples:
        #             break
        #         max_val = max(probabilities)
        #         max_idx = probabilities.index(max_val)
        #         final_results.append(label_list[max_idx])
        #         num_written_lines += 1
        # return final_results
        # assert num_written_lines == num_actual_predict_examples
        for line in prediction_arr:
            text_a = line
            example = InputExample(guid=0, text_a=text_a, label="愤怒")
            feature = convert_single_example(0, example, label_list, max_seq_length, tokenizer)
            input_ids = np.reshape([feature.input_ids], (1, max_seq_length))
            input_mask = np.reshape([feature.input_mask], (1, max_seq_length))
            segment_ids = np.reshape([feature.segment_ids], (max_seq_length))
            label_ids = [feature.label_id]
            dict_curl = {"input_ids": input_ids.tolist(),
                         "input_mask": input_mask.tolist(),
                         "segment_ids": segment_ids.tolist(),
                         "label_ids": label_ids}
            dict_data = {"inputs": dict_curl, "signature_name": "vectorize_service"}



class BertClassifier:

    def run_server(self, task_name, line):
        basic_dir = "sentiment_bert/Bert/"
        vocab_file = basic_dir+"chinese_L-12_H-768_A-12/vocab.txt"
        do_lower_case = True
        max_seq_length = 512

        processors = {
            "polarity": polarityProcessor,
            "triple": tripleProcessor,
        }
        task_name = task_name.lower()
        processor = processors[task_name]()
        label_list = processor.get_labels()
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

        text_a = line
        if task_name == 'triple':
            example = InputExample(guid=0, text_a=text_a, label="愤怒")
        else:
            example = InputExample(guid=0, text_a=text_a, label="pos")
        feature = convert_single_example(0, example, label_list, max_seq_length, tokenizer)
        input_ids = np.reshape([feature.input_ids], (1, max_seq_length))
        input_mask = np.reshape([feature.input_mask], (1, max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (max_seq_length))
        label_ids = [feature.label_id]
        dict_curl = {"input_ids": input_ids.tolist(),
                     "input_mask": input_mask.tolist(),
                     "segment_ids": segment_ids.tolist(),
                     "label_ids": label_ids}
        dict_data = {"inputs": dict_curl, "signature_name": "vectorize_service"}
        if task_name == 'triple':
            url = 'http://124.16.138.75:8501/v1/models/triple_model:predict'
        else:
            url = 'http://124.16.138.75:8501/v1/models/polarity_model:predict'
        requests.adapters.DEFAULT_RETRIES = 5
        s = requests.session()
        s.keep_alive = False
        resp = requests.post(url, json=dict_data)
        label_idx = resp.json()['outputs']['label_predict']
        return label_list[label_idx]
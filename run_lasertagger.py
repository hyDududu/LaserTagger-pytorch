# coding=utf-8

# Lint as: python3
"""BERT-based LaserTagger runner."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

# from tensorflow.contrib import tpu as contrib_tpu
from absl import flags

from src.utils import run_lasertagger_utils
from src.utils import utils

import tensorflow as tf

FLAGS = flags.FLAGS
## Other parameters

# flags.DEFINE_string(
#     "init_checkpoint",
#     '/Users/jiang/Documents/Github/text_scalpel/models/RoBERTa-tiny-clue/bert_model.ckpt',
#     "Initial checkpoint, usually from a pre-trained BERT model. "
#     "导出训练好的模型，为PB用的输出路径"
#     "In the case of exporting, one can optionally provide path to a particular checkpoint to "
#     "be exported here.")
# flags.DEFINE_integer(
#     "max_seq_length", 40,  # contain CLS and SEP
#     "The maximum total input sequence length after WordPiece tokenization. "
#     "Sequences longer than this will be truncated, and sequences shorter than "
#     "this will be padded.")
max_seq_length = 60
do_train = True
do_eval = False
do_export = False
eval_all_checkpoints = False
eval_timeout = 600
# flags.DEFINE_bool("eval_all_checkpoints", False, "Run through all checkpoints.")

# 本地，内存太小了
train_batch_size = 4
eval_batch_size = 2
predict_batch_size = 2
learning_rate = 3e-5
num_train_epochs = 3.0
warmup_proportion = 0.1
save_checkpoints_steps = 20
keep_checkpoint_max = 3
iterations_per_loop = 1000
num_train_examples = 60000
num_eval_examples = 1000

# flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
# flags.DEFINE_string(
#     "tpu_name", None,
#     "The Cloud TPU to use for training. This should be either the name "
#     "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
#     "url.")
# flags.DEFINE_string(
#     "tpu_zone", None,
#     "[Optional] GCE zone where the Cloud TPU is located in. If not "
#     "specified, we will attempt to automatically detect the GCE project from "
#     "metadata.")
flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
master = None  # Optional address of the master for the workers
export_path = './output/models/cefect_export'


def file_based_input_fn_builder(input_file, max_seq_length,
                                is_training, drop_remainder):
    """
    Creates an `input_fn` closure to be passed to TPUEstimator.
    输入函数
    """

    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "labels": tf.FixedLenFeature([max_seq_length], tf.int64),
        "labels_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
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
        d = tf.data.TFRecordDataset(input_file)
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = d.repeat()  # 每ｅｐｏｃｈ重复使用
            d = d.shuffle(buffer_size=100)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=params["batch_size"],
                drop_remainder=drop_remainder))
        return d

    return input_fn


def run_lasertagger(training_file, eval_file, label_map_file, model_config_file, output_dir, init_checkpoint, do_train,
                    keep_checkpoint_max, iterations_per_loop, do_eval, do_export, num_train_epochs, train_batch_size,
                    save_checkpoints_steps, max_seq_length, num_train_examples, num_eval_examples, export_path):
    # tf.logging.set_verbosity(tf.logging.DEBUG)

    if not (do_train or do_eval or do_export):
        raise ValueError("At least one of `do_train`, `do_eval` or `do_export` must"
                         " be True.")

    model_config = run_lasertagger_utils.LaserTaggerConfig.from_json_file(
        model_config_file)

    if max_seq_length > model_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (max_seq_length, model_config.max_position_embeddings))

    if not do_export:
        tf.io.gfile.makedirs(output_dir)

    num_tags = len(utils.read_label_map(label_map_file))

    # tpu_cluster_resolver = None
    # if use_tpu and tpu_name:
    #     tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    #         tpu_name, zone=tpu_zone, project=gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        # cluster=tpu_cluster_resolver,
        master=master,
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=keep_checkpoint_max,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            per_host_input_for_training=is_per_host,
            eval_training_input_configuration=tf.contrib.tpu.InputPipelineConfig.SLICED))

    if do_train:
        num_train_steps, num_warmup_steps = utils._calculate_steps(num_train_examples, train_batch_size,
                                                                   num_train_epochs, warmup_proportion)
    else:
        num_train_steps, num_warmup_steps = None, None

    model_fn = run_lasertagger_utils.ModelFnBuilder(
        config=model_config,
        num_tags=num_tags,
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        # use_tpu=use_tpu,
        # use_one_hot_embeddings=use_tpu,
        max_seq_length=max_seq_length).build()

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        # use_tpu=use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size
    )

    if do_train:
        train_input_fn = file_based_input_fn_builder(
            input_file=training_file,
            max_seq_length=max_seq_length,
            is_training=True,
            drop_remainder=True)
        tensors_to_log = {'train loss': 'loss/Mean:0'}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[logging_hook])

    if do_export:
        tf.logging.info("Exporting the model...")

        def serving_input_fn():
            def _input_fn():
                features = {
                    "input_ids": tf.placeholder(tf.int64, [None, None]),
                    "input_mask": tf.placeholder(tf.int64, [None, None]),
                    "segment_ids": tf.placeholder(tf.int64, [None, None]),
                }
                return tf.estimator.export.ServingInputReceiver(
                    features=features, receiver_tensors=features)

            return _input_fn

        estimator.export_saved_model(
            export_path,
            serving_input_fn(),
            checkpoint_path=init_checkpoint)

# def main(_):
#     tf.logging.set_verbosity(tf.logging.DEBUG)
#
#     if not (FLAGS.do_train or FLAGS.do_eval or FLAGS.do_export):
#         raise ValueError("At least one of `do_train`, `do_eval` or `do_export` must"
#                          " be True.")
#
#     model_config = run_lasertagger_utils.LaserTaggerConfig.from_json_file(
#         FLAGS.model_config_file)
#
#     if FLAGS.max_seq_length > model_config.max_position_embeddings:
#         raise ValueError(
#             "Cannot use sequence length %d because the BERT model "
#             "was only trained up to sequence length %d" %
#             (FLAGS.max_seq_length, model_config.max_position_embeddings))
#
#     if not FLAGS.do_export:
#         tf.io.gfile.makedirs(FLAGS.output_dir)
#
#     num_tags = len(utils.read_label_map(FLAGS.label_map_file))
#
#     tpu_cluster_resolver = None
#     if FLAGS.use_tpu and FLAGS.tpu_name:
#         tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
#             FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
#
#     is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
#     run_config = tf.contrib.tpu.RunConfig(
#         cluster=tpu_cluster_resolver,
#         master=FLAGS.master,
#         model_dir=FLAGS.output_dir,
#         save_checkpoints_steps=FLAGS.save_checkpoints_steps,
#         keep_checkpoint_max=FLAGS.keep_checkpoint_max,
#         tpu_config=tf.contrib.tpu.TPUConfig(
#             iterations_per_loop=FLAGS.iterations_per_loop,
#             per_host_input_for_training=is_per_host,
#             eval_training_input_configuration=tf.contrib.tpu.InputPipelineConfig.SLICED))
#
#     if FLAGS.do_train:
#         num_train_steps, num_warmup_steps = utils._calculate_steps(
#             FLAGS.num_train_examples, FLAGS.train_batch_size,
#             FLAGS.num_train_epochs, FLAGS.warmup_proportion)
#     else:
#         num_train_steps, num_warmup_steps = None, None
#
#     model_fn = run_lasertagger_utils.ModelFnBuilder(
#         config=model_config,
#         num_tags=num_tags,
#         init_checkpoint=FLAGS.init_checkpoint,
#         learning_rate=FLAGS.learning_rate,
#         num_train_steps=num_train_steps,
#         num_warmup_steps=num_warmup_steps,
#         use_tpu=FLAGS.use_tpu,
#         use_one_hot_embeddings=FLAGS.use_tpu,
#         max_seq_length=FLAGS.max_seq_length).build()
#
#     # If TPU is not available, this will fall back to normal Estimator on CPU
#     # or GPU.
#     estimator = tf.contrib.tpu.TPUEstimator(
#         use_tpu=FLAGS.use_tpu,
#         model_fn=model_fn,
#         config=run_config,
#         train_batch_size=FLAGS.train_batch_size,
#         eval_batch_size=FLAGS.eval_batch_size,
#         predict_batch_size=FLAGS.predict_batch_size
#     )
#
#     if FLAGS.do_train:
#         train_input_fn = file_based_input_fn_builder(
#             input_file=FLAGS.training_file,
#             max_seq_length=FLAGS.max_seq_length,
#             is_training=True,
#             drop_remainder=True)
#         tensors_to_log = {'train loss': 'loss/Mean:0'}
#         logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)
#         estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[logging_hook])
#
#     if FLAGS.do_export:
#         tf.logging.info("Exporting the model...")
#
#         def serving_input_fn():
#             def _input_fn():
#                 features = {
#                     "input_ids": tf.placeholder(tf.int64, [None, None]),
#                     "input_mask": tf.placeholder(tf.int64, [None, None]),
#                     "segment_ids": tf.placeholder(tf.int64, [None, None]),
#                 }
#                 return tf.estimator.export.ServingInputReceiver(
#                     features=features, receiver_tensors=features)
#
#             return _input_fn
#
#         estimator.export_saved_model(
#             FLAGS.export_path,
#             serving_input_fn(),
#             checkpoint_path=FLAGS.init_checkpoint)


# if __name__ == "__main__":
#     flags.mark_flag_as_required("model_config_file")
#     flags.mark_flag_as_required("label_map_file")
#     tf.app.run()

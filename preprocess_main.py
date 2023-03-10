# coding=utf-8

# Lint as: python3
"""Convert a dataset into the TFRecord format.
The resulting TFRecord file will be used when training a LaserTagger model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

from absl import logging

from src import bert_example, tagging_converter
from src.utils import utils

import tensorflow as tf

do_lower_case = True
enable_swap_tag = True
output_arbitrary_targets_for_infeasible_examples = True


def _write_example_count(output_tfrecord, count: int) -> Text:
    count_fname = output_tfrecord + '.num_examples.txt'
    with tf.io.gfile.GFile(count_fname, 'w') as count_writer:
        count_writer.write(str(count))
    return count_fname


# def main(argv):
# if len(argv) > 1:
#     raise app.UsageError('Too many command-line arguments.')
# flags.mark_flag_as_required('input_file')
# flags.mark_flag_as_required('input_format')
# flags.mark_flag_as_required('output_tfrecord')
# flags.mark_flag_as_required('label_map_file')
# flags.mark_flag_as_required('vocab_file')
#
# label_map = utils.read_label_map(FLAGS.label_map_file)
# converter = tagging_converter.TaggingConverter(
#     tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
#     FLAGS.enable_swap_tag)
#
# builder = bert_example.BertExampleBuilder(label_map, FLAGS.vocab_file,
#                                           FLAGS.max_seq_length,
#                                           FLAGS.do_lower_case, converter)
#
# num_converted = 0
# with tf.io.TFRecordWriter(FLAGS.output_tfrecord) as writer:
#     for i, (sources, target) in enumerate(utils.yield_sources_and_targets(
#             FLAGS.input_file, FLAGS.input_format)):
#         logging.log_every_n(
#             logging.INFO,
#             f'{i} examples processed, {num_converted} converted to tf.Example.',
#             10000)
#         example = builder.build_bert_example(
#             sources, target,
#             FLAGS.output_arbitrary_targets_for_infeasible_examples)
#         if example is None:
#             continue
#         writer.write(example.to_tf_example().SerializeToString())
#         num_converted += 1
# logging.info(f'Done. {num_converted} examples converted to tf.Example.')
# count_fname = _write_example_count(num_converted)
# logging.info(f'Wrote:\n{FLAGS.output_tfrecord}\n{count_fname}')

def data_process(input_file, input_format, output_tfrecord, label_map_file, vocab_file, max_seq_length,
                 output_arbitrary_targets_for_infeasible_examples):
    label_map = utils.read_label_map(label_map_file)
    converter = tagging_converter.TaggingConverter(
        tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
        enable_swap_tag)

    builder = bert_example.BertExampleBuilder(label_map, vocab_file, max_seq_length, do_lower_case, converter)

    num_converted = 0
    with tf.io.TFRecordWriter(output_tfrecord) as writer:
        for i, (sources, target) in enumerate(utils.yield_sources_and_targets(input_file, input_format)):
            logging.log_every_n(
                logging.INFO,
                f'{i} examples processed, {num_converted} converted to tf.Example.',
                10000)
            example = builder.build_bert_example(sources, target, output_arbitrary_targets_for_infeasible_examples)
            if example is None:
                continue
            writer.write(example.to_tf_example().SerializeToString())
            num_converted += 1
    logging.info(f'Done. {num_converted} examples converted to tf.Example.')
    count_fname = _write_example_count(output_tfrecord, num_converted)
    logging.info(f'Wrote:\n{output_tfrecord}\n{count_fname}')

# if __name__ == '__main__':
#     app.run(main)

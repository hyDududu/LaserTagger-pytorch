# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import math, time
from termcolor import colored
import tensorflow as tf

from src import bert_example, tagging_converter
from src.utils import predict_utils
from src.utils import utils

from src.curLine_file import curLine

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file', '/Users/jiang/Documents/Github/text_scalpel/corpus/rephrase_corpus/test.txt',
    'Path to the input file containing examples for which to compute '
    'predictions.')
flags.DEFINE_enum(
    'input_format', 'wikisplit', ['wikisplit'],
    'Format which indicates how to parse the input_file.')
flags.DEFINE_string(
    'output_file',
    '/Users/jiang/Documents/Github/text_scalpel/output/models/cefect/pred.tsv',
    'Path to the TSV file where the predictions are written to.')
flags.DEFINE_string(
    'label_map_file', '/Users/jiang/Documents/Github/text_scalpel/corpus/rephrase_corpus/output/label_map.txt',
    'Path to the label map file. Either a JSON file ending with ".json", that '
    'maps each possible tag to an ID, or a text file that has one tag per '
    'line.')
flags.DEFINE_string('vocab_file', '/Users/jiang/Documents/Github/text_scalpel/bert_base/RoBERTa-tiny-clue/vocab.txt',
                    'Path to the BERT vocabulary file.')
flags.DEFINE_integer('max_seq_length', 40, 'Maximum sequence length.')
flags.DEFINE_bool(
    'do_lower_case', False,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_bool('enable_swap_tag', True, 'Whether to enable the SWAP tag.')
flags.DEFINE_string('saved_model',
                    '/Users/jiang/Documents/Github/text_scalpel/output/cefect/export/1591764354',
                    'Path to an exported TF model.')


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('input_format')
    flags.mark_flag_as_required('output_file')
    flags.mark_flag_as_required('label_map_file')
    flags.mark_flag_as_required('vocab_file')
    flags.mark_flag_as_required('saved_model')

    label_map = utils.read_label_map(FLAGS.label_map_file)
    converter = tagging_converter.TaggingConverter(
        tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
        FLAGS.enable_swap_tag)
    builder = bert_example.BertExampleBuilder(label_map, FLAGS.vocab_file,
                                              FLAGS.max_seq_length,
                                              FLAGS.do_lower_case, converter)
    predictor = predict_utils.LaserTaggerPredictor(
        tf.contrib.predictor.from_saved_model(FLAGS.saved_model), builder,
        label_map)
    print(colored("%s input file:%s" % (curLine(), FLAGS.input_file), "red"))
    sources_list = []
    target_list = []
    with tf.io.gfile.GFile(FLAGS.input_file) as f:
        for line in f:
            sources, target = line.rstrip('\n').replace('\ufeff', '').split('\t')
            sources_list.append([sources])
            target_list.append(target)
    number = len(sources_list)  # ????????????
    predict_batch_size = min(64, number)
    batch_num = math.ceil(float(number) / predict_batch_size)

    start_time = time.time()
    num_predicted = 0
    with tf.gfile.Open(FLAGS.output_file, 'w') as writer:
        writer.write(f'source\tprediction\ttarget\n')
        for batch_id in range(batch_num):
            sources_batch = sources_list[batch_id * predict_batch_size: (batch_id + 1) * predict_batch_size]
            prediction_batch = predictor.predict_batch(sources_batch=sources_batch)
            assert len(prediction_batch) == len(sources_batch)
            num_predicted += len(prediction_batch)
            for id, [prediction, sources] in enumerate(zip(prediction_batch, sources_batch)):
                target = target_list[batch_id * predict_batch_size + id]
                writer.write(f'{"".join(sources)}\t{prediction}\t{target}\n')
            if batch_id % 20 == 0:
                cost_time = (time.time() - start_time) / 60.0
                print("%s batch_id=%d/%d, predict %d/%d examples, cost %.2fmin." %
                      (curLine(), batch_id + 1, batch_num, num_predicted, number, cost_time))
    cost_time = (time.time() - start_time) / 60.0
    logging.info(
        f'{curLine()} {num_predicted} predictions saved to:{FLAGS.output_file}, cost {cost_time} min, ave {cost_time / num_predicted} min.')


if __name__ == '__main__':
    app.run(main)

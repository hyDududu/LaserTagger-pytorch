import argparse
from phrase_vocabulary_optimization import vocab_build
from preprocess_main import data_process
from run_lasertagger import run_lasertagger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 第一步
    print('run:phrase_vocabulary_optimization.py,开始建立，优化词汇表')
    parser.add_argument('--input_file', type=str, default="./corpus/rephrase_corpus/train.txt",
                        help='Path to the input file containing examples to be converted to tf.Examples.')
    parser.add_argument('--input_format', type=str, default="wikisplit",
                        help='Format which indicates how to parse the input_file')
    parser.add_argument('--vocabulary_size', type=int, default=500, help='size of vocabulary')
    parser.add_argument('--max_input_examples', type=int, default=100000, help='max number of the input examples')
    parser.add_argument('--enable_swap_tag', type=bool, default=False, help='Whether to enable the SWAP tag')
    parser.add_argument('--output_file', type=str, default="./output/label_map.txt", help='output file')
    args, _ = parser.parse_known_args()
    vocab_build(args.input_file, args.input_format, args.vocabulary_size, args.max_input_examples,
                args.enable_swap_tag, args.output_file)

    print("run:preprocess_main.py,开始整理数据")
    # parser.add_argument('--input_file', type=str, default="./corpus/rephrase_corpus/train.txt", help='')
    # parser.add_argument('--input_format', type=str, default="wikisplit", help='Format which indicates how to parse the input_file')
    parser.add_argument('--output_tfrecord', type=str, default="./output/train.tf_record", help='')
    parser.add_argument('--label_map_file', type=str, default="./output/label_map.txt",
                        help="Path to the label map file. Either a JSON file ending with '.json', that maps each possible tag to an ID, or a text file that has one tag per line")
    parser.add_argument('--vocab_file', type=str, default="./bert_base/RoBERTa-tiny-clue/vocab.txt",
                        help='')
    parser.add_argument('--max_seq_length', type=int, default=60, help='Maximum sequence length')
    parser.add_argument('--output_arbitrary_targets_for_infeasible_examples', type=bool, default=False,
                        help='Set this to True when preprocessing the development set. Determines '
                             'whether to output a TF example also for sources that can not be converted to target via the available tagging operations. In these cases, the target ids will correspond to the tag sequence KEEP-DELETE-KEEP-DELETE... which should be very unlikely to be predicted by chance. This will be useful for getting more accurate eval scores during training')
    args, _ = parser.parse_known_args()
    data_process(args.input_file, args.input_format, args.output_tfrecord, args.label_map_file, args.vocab_file,
                 args.max_seq_length, args.output_arbitrary_targets_for_infeasible_examples)

    parser.add_argument('--input_file', type=str, default="./corpus/rephrase_corpus/tune.txt", help='')
    # parser.add_argument('--input_format', type=str, default="wikisplit", help='Format which indicates how to parse the input_file')
    parser.add_argument('--output_tfrecord', type=str, default="./output/tune.tf_record", help='')
    # parser.add_argument('--label_map_file', type=str, default="./output/label_map.txt", help='')
    # parser.add_argument('--vocab_file', type=str, default="./bert_base/RoBERTa-tiny-clue/vocab.txt",
    #                     help='')
    # parser.add_argument('--max_seq_length', type=int, default=60, help='Maximum sequence length')
    # parser.add_argument('--output_arbitrary_targets_for_infeasible_examples', type=bool, default=False,
    #                     help='')
    args, _ = parser.parse_known_args()
    data_process(args.input_file, args.input_format, args.output_tfrecord, args.label_map_file, args.vocab_file,
                 args.max_seq_length, args.output_arbitrary_targets_for_infeasible_examples)

    # 第二步：训练模型
    parser.add_argument('--training_file', type=str, default="./output/train.tf_record",
                        help='Path to the TFRecord training file')
    parser.add_argument('--eval_file', type=str, default="./output/tune.tf_record",
                        help='Path to the the TFRecord dev file')
    # parser.add_argument('--label_map_file', type=str, default="./output/label_map.txt", help="Path to the label map file. Either a JSON file ending with '.json', that maps each possible tag to an ID, or a text file that has one tag per line")
    parser.add_argument('--model_config_file', type=str, default="./configs/lasertagger_config.json",
                        help='The config json file specifying the model architecture')
    parser.add_argument('--output_dir', type=str, default="./output/models/",
                        help="The output directory where the model checkpoints will be written. If `init_checkpoint' is not provided when exporting, the latest checkpoint from this directory will be exported")
    parser.add_argument('--init_checkpoint', type=str, default="./bert_base/RoBERTa-tiny-clue/bert_model.ckpt", help='')
    parser.add_argument('--do_train', type=bool, default=True, help='Whether to run training')
    parser.add_argument('--keep_checkpoint_max', type=int, default=3, help='')
    parser.add_argument('--iterations_per_loop', type=int, default=1000, help='')
    parser.add_argument('--do_eval', type=bool, default=False, help='Whether to run eval on the dev set')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='')
    parser.add_argument('--train_batch_size', type=int, default=256, help='')
    parser.add_argument('--save_checkpoints_steps', type=int, default=200, help='')
    # parser.add_argument('--max_seq_length', type=int, default=60, help='Maximum sequence length')
    parser.add_argument('--num_train_examples', type=int, default=1097189, help='size of vocabulary')
    parser.add_argument('--num_eval_examples', type=int, default=1723, help='max number of the input examples')
    args, _ = parser.parse_known_args()
    run_lasertagger(args.training_file, args.eval_file, args.label_map_file, args.model_config_file, args.output_dir,
                    args.init_checkpoint, args.do_train, args.keep_checkpoint_max, args.iterations_per_loop,
                    args.do_eval, False, args.num_train_epochs, args.train_batch_size, args.save_checkpoints_steps,
                    args.max_seq_length, args.num_train_examples, args.num_eval_examples, '')

    # 第三步：Export the model
    # parser.add_argument('--label_map_file', type=str, default="./output/label_map.txt", help="Path to the label map file. Either a JSON file ending with '.json', that maps each possible tag to an ID, or a text file that has one tag per line")
    parser.add_argument('--init_checkpoint', type=str, default="./output/models/cefect/model.ckpt-20", help='')
    parser.add_argument('--model_config_file', type=str, default="./configs/lasertagger_config.json",
                        help='The config json file specifying the model architecture')
    parser.add_argument('--output_dir', type=str, default="./output/models/cefect",
                        help="The output directory where the model checkpoints will be written. If `init_checkpoint' is not provided when exporting, the latest checkpoint from this directory will be exported")
    parser.add_argument('--do_train', type=bool, default=False, help='Whether to run training')
    parser.add_argument('--do_eval', type=bool, default=False, help='Whether to run eval on the dev set')
    parser.add_argument('--do_export', type=bool, default=True, help='Whether to export a trained model')
    parser.add_argument('--export_path', type=str, default="./output/models/cefect_export",
                        help="Path to save the exported model")
    args, _ = parser.parse_known_args()
    run_lasertagger(args.training_file, args.eval_file, args.label_map_file, args.model_config_file, args.output_dir,
                    args.init_checkpoint, args.do_train, args.keep_checkpoint_max, args.iterations_per_loop,
                    args.do_eval, args.do_export, args.num_train_epochs, args.train_batch_size,
                    args.save_checkpoints_steps, args.max_seq_length, args.num_train_examples, args.num_eval_examples,
                    args.export_path)

# run_toolkit(args.model, args.dataset, args.language, config_dict)

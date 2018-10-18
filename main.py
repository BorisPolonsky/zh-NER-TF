import tensorflow as tf
import numpy as np
import os
import argparse
import yaml
import sys
import time
import random
from model import BiLSTM_CRF, BiDirectionalStackedLSTM_CRF, VariationalBiRNN_CRF
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, get_tag2label, random_embedding

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # need ~700MB GPU memory


## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrained_embedding',
                    type=str,
                    default=None,
                    help='Path to serialized pretrained embedding. By default it would be initialized randomly.')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
parser.add_argument('--num_rnn_layer', type=int, default=1,
                    help='Number of RNN cells to be stacked.')
parser.add_argument('--output_path', type=str, default=None, help='Directory for saving model, summaries, etc..')
parser.add_argument('--model_type', type=str, default="bi-lstm-crf",
                    help='bi-lstm-crf/bi-stacked-lstm-crf/variational-bi-lstm-crf')
parser.add_argument('--word2id', type=str, default=None, help='Serialized word2id dictionary.')
parser.add_argument('--digit_token', type=str, default=None,
                    help='If specified (e.g. "<NUM>"), '
                         'all digits in the original text will be overridden with this token.')
parser.add_argument('--latin_char_token', type=str, default=None,
                    help='If specified (e.g. "<ENG>"), '
                         'all latin characters in the original text will be overridden with this token.')
parser.add_argument('--unknown_word_token', type=str, default='<UNK>',
                    help='If specified (e.g. "<UNK>"), '
                    'all characters beyond vocabulary will be overridden with this token.')
parser.add_argument("--entity_tokens", type=lambda x: x.split(","), default="PER,LOC,ORG",
                    help="List of ordered entity tokens to take into account. Split by commas. Default: PER,LOG,ORG")
parser.add_argument("--config_path", default=None, type=str,
                    help="A yaml file for overriding parameters specification in this module.")
args = parser.parse_args()

# Override parameters
if args.config_path is not None:
    with open(args.config_path, "r") as f:
        yml_config = yaml.safe_load(f)
    for k, v in yml_config.items():
        if k in args.__dict__:
            args.__dict__[k] = v
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

# get char embeddings
if args.word2id is None:
    word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
else:
    word2id = read_dictionary(args.word2id)
if args.pretrained_embedding is None:
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = os.path.normpath(args.pretrained_embedding)
    embedding = np.load(embedding_path)
    if isinstance(embedding, np.lib.npyio.NpzFile):
        embedding = embedding['embedding']
    embeddings = np.array(embedding, dtype='float32')

# read corpus and get training data
if args.mode not in ('demo', 'predict'):
    train_path = os.path.join('.', args.train_data, 'train_data')
    test_path = os.path.join('.', args.test_data, 'test_data')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path)
    test_size = len(test_data)



# paths setting
paths = {}
if args.output_path is None:
    output_path = os.path.join('.', "data_path_baseline", '1521112368') # Basline model
else:
    output_path = os.path.normpath(args.output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path):
    os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path):
    os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


# Model selection
model_constructor = {"bi-lstm-crf": BiLSTM_CRF,
                     "bi-stacked-lstm-crf": BiDirectionalStackedLSTM_CRF,
                     "variational-bi-lstm-crf": VariationalBiRNN_CRF}[args.model_type]
tag2label = get_tag2label(args.entity_tokens)

# training model
if args.mode == 'train':
    # hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    # train model on the whole training data
    model = model_constructor(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    print("test data: {}".format(test_size))
    model = model_constructor(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    label_list = model.test(test_data)
    label2tag = {label: tag for tag, label in model.tag2label.items()}
    for sentence_label, (sentence, sentence_tag_real) in zip(label_list, test_data):
        sentence_tag_predict = [label2tag[label] for label in sentence_label]
        if sentence_tag_real != sentence_tag_predict:
            real_entities = dict(
                zip(args.entity_tokens,
                    get_entity(sentence_tag_real, sentence, suffixes=args.entity_tokens, strict=False)))
            detected_entities = dict(
                zip(args.entity_tokens,
                    get_entity(sentence_tag_predict, sentence, suffixes=args.entity_tokens, strict=False)))
            human_readable_result = []
            for ch, tag_r, tag_p in zip(sentence, sentence_tag_real, sentence_tag_predict):
                if tag_r == tag_p:
                    human_readable_result.append("{}({})".format(ch, tag_r))
                else:
                    human_readable_result.append("{}({}|{})".format(ch, tag_r, tag_p))
            human_readable_result = "".join(human_readable_result)
            print("Sentence:\n{}\nGround truth:\n{}\nDetection:\n{}".format(human_readable_result,
                                                                            real_entities,
                                                                            detected_entities))

elif args.mode == "corpus-check":
    for data_name, data in zip(("train", "dev"), (train_data, test_data)):
        print(data_name)
        for sentence, tags in data:
            try:
                get_entity(tags, sentence, strict=True)
            except ValueError:
                print("Corrupted tag sequence")
                print(" ".join("{}({}).".format(ch, tag) for ch, tag in zip(sentence, tags)))

elif args.mode == "predict":
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = model_constructor(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    test_path = os.path.join('.', args.test_data, 'test_data')
    with tf.Session(config=config) as sess:
        saver.restore(sess, model.model_path)
        predict_result_path = os.path.join(paths["result_path"], "predict_only")
        with open(os.path.join('.', args.test_data, 'test_raw_data'), "r") as fi, open(predict_result_path, "w") as fo:
            for line in fi:
                line = line.rstrip()
                if len(line) == 0:
                    continue
                test_data = [(line, ("O",) * len(line))]
                tag = model.demo_one(sess, test_data)
                entities = get_entity(tag, line, suffixes=args.entity_tokens)
                human_readable_msg = ['Sentence:\n{}'.format(line)] + \
                                     ["{}:\n{}".format(token, entity) for \
                                      token, entity in zip(args.entity_tokens, entities)]
                human_readable_msg = "\n".join(human_readable_msg)
                fo.write(human_readable_msg + "\n")
                print(human_readable_msg)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = model_constructor(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while True:
            try:
                print('Please input your sentence:')
                demo_sent = input()
                if demo_sent == '' or demo_sent.isspace():
                    break
                else:
                    demo_sent = list(demo_sent.strip())
                    demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                    tag = model.demo_one(sess, demo_data)
                    entities = get_entity(tag, demo_sent, suffixes=args.entity_tokens)
                    human_readable_msg = "\n".join(["{}:\n{}".format(token, entity)
                                                    for token, entity in zip(args.entity_tokens, entities)])
                    print(human_readable_msg)
            except KeyboardInterrupt:
                break
        print('See you next time!')

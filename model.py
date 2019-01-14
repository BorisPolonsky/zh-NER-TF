import numpy as np
import os
import time
import sys
import tensorflow as tf
from tensorflow.nn.rnn_cell import GRUCell, LSTMCell, DropoutWrapper
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield, sentence2id
from utils import get_logger
from eval import conlleval
from abc import abstractmethod


class Model:
    """
    The Basic model class.
    """
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch if 'epoch' in args else None
        self.embeddings = embeddings
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout if 'dropout' in args else 1
        self.optimizer = args.optimizer if 'optimizer' in args else 'SGD'
        self.lr_decay = args.lr_decay if 'lr_decay' in args else 1
        self.clip_grad = args.clip if 'clip' in args else None
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args.shuffle if 'shuffle' in args else False
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'], logger_name=__name__)
        self.result_path = paths['result_path']
        self.config = config
        self.digit_token = args.digit_token
        self.latin_token = args.latin_char_token
        self.unknown_word_token = args.unknown_word_token
        self.lr = args.lr if 'lr' in args else None
        self.lr_decay = 0.0 if ('lr_decay' not in args) or (args.lr_decay is None) else args.lr_decay
        self.embedding_dim = args.embedding_dim
        self.word_embeddings = None
        self.loss = None
        self.merged = None
        self.file_writer = None
        self.word_ids = None
        self.labels = None
        self.sequence_lengths = None
        self.dropout_pl = None
        self.lr_pl = None
        self.train_op = None
        self.global_step = None

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        embedding_shape = None if self.embeddings is not None else (len(self.vocab), self.embedding_dim)
        with tf.variable_scope("words", reuse=False):
            _word_embeddings = tf.get_variable(initializer=self.embeddings,
                                               dtype=tf.float32,
                                               trainable=self.update_embedding,
                                               name="_word_embeddings",
                                               shape=embedding_shape)
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            if self.clip_grad is not None:
                grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            else:
                grads_and_vars_clip = grads_and_vars  # Leave it as is
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        """

        :param train:
        :param dev:
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)
            if self.epoch_num is None:
                raise RuntimeError("Please specify epoch_num.")
            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        """

        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        if self.lr is None:
            raise RuntimeError("Please specify lr.")
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label,
                              shuffle=self.shuffle,
                              digit_token=self.digit_token,
                              latin_char_token=self.latin_token,
                              unknown_word_token=self.unknown_word_token)
        lr = self.lr * (1 - self.lr_decay) ** epoch
        for step, (seqs, labels) in enumerate(batches):
            sys.stdout.write(' processing: {} batch / {} batches.\r'.format(step + 1, num_batches))
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op,
                                                          self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, '
                    'global_step: {}, learning_rate: {}'.format(start_time,
                                                                epoch + 1, step + 1,
                                                                loss_train,
                                                                step_num,
                                                                lr))
            self.file_writer.add_summary(summary, step_num)
            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def demo_one(self, sess, sent):
        """

        :param sess:
        :param sent:
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent,
                                        self.batch_size,
                                        self.vocab,
                                        self.tag2label,
                                        shuffle=False,
                                        digit_token=self.digit_token,
                                        latin_char_token=self.latin_token,
                                        unknown_word_token=self.unknown_word_token):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def predict_only(self, sess, single_batch):
        """

        :param sess: tf.Session
        :param single_batch: A single batch where each slot contains a char-sequence.
        :return: tag_seq_list
        """
        def get_sent2id():
            return lambda sentence: sentence2id(sent=sentence,
                                                word2id=self.vocab,
                                                digit_token_override=self.digit_token,
                                                latin_char_token_override=self.latin_token,
                                                unknown_word_token=self.unknown_word_token)

        sent2id = get_sent2id()
        input_batch = list(map(sent2id, single_batch))
        label_seq_list, _ = self.predict_one_batch(sess, input_batch)
        label2tag = {}
        for tag, label in self.tag2label.items():
            # print('test', tag, label)
            label2tag[label] = tag
        tag_seq_list = []
        for label_seq in label_seq_list:
            tag_seq_list.append([label2tag[label] for label in label_seq])
        return tag_seq_list

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)
        return label_list

    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label,
                                        shuffle=False,
                                        digit_token=self.digit_token,
                                        latin_char_token=self.latin_token,
                                        unknown_word_token=self.unknown_word_token):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """

        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch is not None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)

    @abstractmethod
    def predict_one_batch(self, seqs):
        pass

    @abstractmethod
    def build_graph(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_feed_dict(self, *args, **kwargs):
        pass


class BiLSTM_CRF(Model):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        super().__init__(args, embeddings, tag2label, vocab, paths, config)
        self.hidden_dim = args.hidden_dim
        self.transition_params = None
        self.loss = None
        self.global_step = None
        self.logits = None
        self.log_likelihood = None
        self.labels_softmax_ = None
        self.word_embeddings = None
        self.word_ids = None
        self.merged = None
        self.train_op = None

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.pred_op()
        self.loss_op()
        self.trainstep_op()

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        if self.CRF:
            self.loss = -tf.reduce_mean(self.log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)
        else:
            self.log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                             tag_indices=self.labels,
                                                                             sequence_lengths=self.sequence_lengths)

    @property
    def init_op(self):
        return tf.global_variables_initializer()

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """

        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list


class BiDirectionalStackedLSTM_CRF(BiLSTM_CRF):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        super().__init__(args, embeddings, tag2label, vocab, paths, config)
        self._layer_num = args.num_rnn_layer

    def biLSTM_layer_op(self):
        """
        Bi-(Stacked)LSTM layer
        :return:
        """
        def inner_cells():
                return [LSTMCell(self.hidden_dim) for _ in range(self.layer_num)]
        with tf.variable_scope("bi-stacked-lstm"):
            cell_fw = tf.nn.rnn_cell.MultiRNNCell(inner_cells())
            cell_bw = tf.nn.rnn_cell.MultiRNNCell(inner_cells())
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    @property
    def layer_num(self):
        return self._layer_num


class VariationalBiRNN_CRF(BiLSTM_CRF):
    """
    An BiRNN-CRF architecture with the same dropout mask applied at every step for
    forward/backward rnn cells, as described in:
    Y. Gal, Z Ghahramani. "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks".
    https://arxiv.org/abs/1512.05287
    """
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        super().__init__(args, embeddings, tag2label, vocab, paths, config)
        if "cell_type" in args:
            if isinstance(args.cell_type):
                self._cell_type = {"lstm": LSTMCell, "gru": GRUCell}[args.cell_type.lower()]
            elif callable(args.cell_type):
                self._cell_type = args.cell_type
            else:
                raise ValueError("Unsupported cell_type specification.")
        else:
            self._cell_type = LSTMCell

    def lookup_layer_op(self):
        """
        This method does not create dropout layer for word embedding, as DropoutWrapper applies
        dropout for RNN inputs.
        :return:
        """
        embedding_shape = None if self.embeddings is not None else (len(self.vocab), self.embedding_dim)
        with tf.variable_scope("words"):
            _word_embeddings = tf.get_variable(initializer=self.embeddings,
                                               dtype=tf.float32,
                                               trainable=self.update_embedding,
                                               name="_word_embeddings",
                                               shape=embedding_shape)

            self.word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                          ids=self.word_ids,
                                                          name="word_embeddings")

    def biLSTM_layer_op(self):
        def get_cell():
            return DropoutWrapper(self._cell_type(self.hidden_dim),
                                  input_keep_prob=self.dropout_pl,
                                  output_keep_prob=self.dropout_pl,
                                  state_keep_prob=self.dropout_pl,
                                  variational_recurrent=True,
                                  dtype=tf.float32,
                                  input_size=self.embedding_dim)
        with tf.variable_scope("variational-bi-rnn"):
            cell_fw, cell_bw = get_cell(), get_cell()
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            # output = tf.nn.dropout(output, self.dropout_pl) # Redundancy

        with tf.variable_scope("projection-layer"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])


class IDCNN_CRF(Model):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        super().__init__(args, embeddings, tag2label, vocab, paths, config)
        self.num_filter = args.hidden_dim
        self.filter_width = args.filter_width
        self.transition_params = None
        self.loss = None
        self.global_step = None
        self.logits = None
        self.log_likelihood = None
        self.labels_softmax_ = None
        self.word_embeddings = None
        self.word_ids = None
        self.merged = None
        self.train_op = None
        self.n_repeat = args.num_repeat
        self.dilation_rate = args.dilation
        self.n_repeat = args.num_repeat

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.idcnn_layer_op()
        self.pred_op()
        self.loss_op()
        self.trainstep_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)
        else:
            self.log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                             tag_indices=self.labels,
                                                                             sequence_lengths=self.sequence_lengths)

    def idcnn_layer_op(self):
        # self.word_embeddings [batch_size, max_time, embedding_dim]
        layer_input = tf.expand_dims(self.word_embeddings, axis=1)
        with tf.variable_scope("idcnn"):
            with tf.variable_scope("input_proj"):
                input_filter = tf.get_variable("input_mapping_filter",
                                               shape=[1, self.filter_width, self.embedding_dim, self.num_filter])
                layer_input = tf.nn.convolution(layer_input, input_filter,
                                                padding="SAME", strides=[1, 1],
                                                dilation_rate=None, name="input")  # [batch_size, 1, max_time, self.num_filter]
                """
                tf.nn.conv2d(layer_input,
                             input_filter,
                             strides=[1, 1, 1, 1],
                             padding="SAME",
                             name="init_layer")
                """
            with tf.variable_scope("conv"):
                final_output, conv_output_dim = [], 0
                for repeat_i in range(self.n_repeat):
                    for layer_j, dilation in enumerate(self.dilation_rate):
                        with tf.variable_scope("dilated-conv-{}".format(layer_j + 1), reuse=tf.AUTO_REUSE):
                            filter_w = tf.get_variable("filter_w",
                                                       shape=[1, self.filter_width, self.num_filter, self.num_filter],
                                                       initializer=tf.contrib.layers.xavier_initializer())
                            filter_b = tf.get_variable("filter_b", shape=[self.num_filter])
                            layer_output = tf.nn.atrous_conv2d(layer_input, filter_w, dilation, padding="SAME")
                            layer_output = tf.nn.bias_add(layer_output, filter_b)
                            layer_output = tf.nn.relu(layer_output)  # [batch_size, 1, max_time, self.num_filter]
                            layer_input = layer_output
                    final_output.append(layer_output)
                    conv_output_dim += self.num_filter
                final_output = tf.concat(final_output, axis=-1)
                final_output = tf.squeeze(final_output, axis=1)  # [batch_size, max_time, self.num_filter]
                final_output = tf.nn.dropout(final_output, self.dropout_pl)
            with tf.variable_scope("output_proj"):
                conv_out_shape = tf.shape(final_output)
                final_output = tf.reshape(final_output, shape=[-1, conv_output_dim])
                proj_w = tf.get_variable(name="w", shape=[conv_output_dim, self.num_tags])
                proj_b = tf.get_variable(name="b", shape=[self.num_tags])
                final_output = tf.nn.xw_plus_b(final_output, proj_w, proj_b)
                self.logits = tf.reshape(final_output, shape=[-1, conv_out_shape[1], self.num_tags])

    @property
    def init_op(self):
        return tf.global_variables_initializer()

    def loss_op(self):
        if self.CRF:
            self.loss = -tf.reduce_mean(self.log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def predict_one_batch(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """

        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

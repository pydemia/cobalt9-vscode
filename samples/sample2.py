import os
import sys
import shutil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sentencepiece as spm
import tf_sentencepiece as tfspm

from gensim.models import Word2Vec


class NoValidationException(Exception):
    def __init__(self, valid_value):
        self.valid_value = str(valid_value)
    def __str__(self):
        msg = "Training with NO-VALIDATION: 'VALIDATION_RATIO=%s'"
        return repr(msg % self.valid_value)


class NumericalTrainingHelper(tf.contrib.seq2seq.TrainingHelper):

    def sample(self, time, outputs, name=None, **unused_kwargs):
        with tf.name_scope(name, "NumericalTrainingHelperSample", [time, outputs]):
            sample_ids = tf.identity(outputs)
            return sample_ids


tf.reset_default_graph()


class NeuralNetworkModel(object):
    def __init__(self, *args, **kwargs):
        super().__init__()


    def input(self, *args, **kwargs):
        pass


    def _layer_rnn(self, *args, **kwargs):
        pass

    def _graph_rnn(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        pass


    def train(self, *args, **kwargs):
        pass


    def evaluate(self, *args, **kwargs):
        pass


    def loss_plot(self, *args, **kwargs):
        pass

# Functions in module --------------------------------------------------------

def get_conv_output_size(input_size, filter_size, stride_size):
    "Assumption: Input & Filter are Square-shaped."
    output_size = ((input_size - filter_size) / stride_size) + 1
    return output_size


# Seq2Seq --------------------------------------------------------------------

class Seq2Seq(NeuralNetworkModel):
    """Varitional RNN.

    TBD

    Attributes
    ----------
    TBD


    Example
    -------
    >>> "TBD"

    """
    def __init__(
        self,
        input_f_dtype=tf.string,
        input_f_shape=(None, ),
        input_x_dtype=tf.int32,
        input_y_dtype=tf.int32,
        input_x_shape=(None, ),
        input_y_shape=(None, ),
        sp_model_dir=None,  # 'data/news/news_spm_200000.model',
        sp_vocab_dir=None,
        sp_vocab_threshold=30,
        reverse=False,
        add_bos=True,
        add_eos=True,
        embedding_size=200,
        gensim_embeddings_binfile=None,
        use_tfrecords=True,
        use_gpu=True,
        bidirectional=True,
        use_attention=True,
        use_residual=False,
        use_normalize=False,
        attention_method='bahdanau',
        input_seq_length=500,
        input_channel=200,
        MAX_OUTPUT_SEQ_LENGTH=300,
        output_channel=200,
        unit_num=32,
        batch_size=128,
        dropout=None,
        proj_activation=tf.nn.tanh,
        start_token=-1.,
        buffer_size=1000,
        learning_rate=.0002,
        lr_decaying=True,
        decay_lr_step=1000,
        adam_beta1=.5,
        adam_beta2=.8,
        validation_ratio=.2,
        time_major=False,
        ):

        super().__init__()
        self.input_f_dtype = input_f_dtype
        self.input_f_shape = input_f_shape

        self.input_x_dtype = input_x_dtype
        self.input_y_dtype = input_y_dtype

        self.input_x_shape = input_x_shape
        self.input_y_shape = input_y_shape

        self.USE_TFRECORDS = use_tfrecords

        self.EMBEDDING_SIZE = embedding_size
        self.GENSIM_EMBEDDINGS_BINFILE = gensim_embeddings_binfile

        self.USE_GPU = use_gpu

        self.SP_MODEL_DIR = sp_model_dir
        self.SP_VOCAB_DIR = sp_vocab_dir
        self.SP_VOCAB_THRESHOLD_INT = self.sp_vocab_threshold
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.SP_MODEL_DIR)
        self.sp.SetEncodeExtraOptions('eos:bos')
        self.sp.LoadVocabulary(
            self.SP_VOCAB_DIR,
            self.SP_VOCAB_THRESHOLD_INT,
        )
        self.VOCAB_SIZE = self.sp.GetPieceSize()

        self.REVERSE = reverse
        self.ADD_BOS = add_bos
        self.ADD_EOS = add_eos

        self.INPUT_SEQ_LENGTH = input_seq_length          # 1hour x 12 = 12hours
        self.INPUT_CHANNEL = input_channel        # Each pixel has KPIs as channels
        # self.INPUT_SEQ_LENGTH = input_x_shape[1]
        # self.INPUT_CHANNEL = input_x_shape[2]

        self.MAX_OUTPUT_SEQ_LENGTH = MAX_OUTPUT_SEQ_LENGTH if MAX_OUTPUT_SEQ_LENGTH else input_seq_length
        self.OUTPUT_CHANNEL = output_channel if output_channel else input_channel
        # self.MAX_OUTPUT_SEQ_LENGTH = input_y_shape[1]
        # self.OUTPUT_CHANNEL = input_y_shape[2]

        self.BIDIRECTIONAL = bidirectional
        self.USE_ATTENTION = use_attention
        self.ATTENTION_METHOD = attention_method
        self.USE_RESIDUAL = use_residual
        self.USE_NORMALIZE = use_normalize
        self.UNIT_NUM = unit_num

        self.BATCH_SIZE = batch_size              # Training Batch Size
        self.DROPOUT = dropout
        self.PROJ_ACTIVATION = proj_activation
        self.START_TOKEN = start_token

        self.BUFFER_SIZE = buffer_size            # For tf.Dataset.suffle(buffer_size)
        self.LEARNING_RATE = learning_rate        # Learning rate (Fixed for now)
        self.DECAY_LR_STEP = decay_lr_step
        self.LR_DECAYING = lr_decaying
        self.ADAM_BETA1 = adam_beta1
        self.ADAM_BETA2 = adam_beta2
        self.VALIDATION_RATIO = validation_ratio  # Validation Ratio
        self.EPOCH_NUM = 0                        # Cumulative Epoch Number
        self.TIME_MAJOR = time_major

        self._x_input_tensor = None
        self._y_input_tensor = None
        self._x_batch_tensor = None
        self._z_batch_tensor = None
        self.data_init_op = None

        self.writer = None
        self.variable_init_op = None
        self.train_op = None
        self.train_loss_history = None
        self.train_valid_history = None

        self.prediction = None
        self.loss = None
        self.accuracy = None
        self.global_step = None

        self.metrics_init_op = None
        self.metrics = None
        self.update_metrics_op = None

        self.summary_op = None

        self.history = {}

        tf.reset_default_graph()

        if self.GENSIM_EMBEDDINGS_BINFILE:
            embed = self._load_gensim_embed(
                self.GENSIM_EMBEDDINGS_BINFILE
            )
            self.embed_vector = embed.wv.vectors

        # self.input(
        #     batch_size=self.BATCH_SIZE,
        #     input_x_dtype=self.input_x_dtype,
        #     input_y_dtype=self.input_y_dtype,
        #     input_x_shape=self.input_x_shape,
        #     input_y_shape=self.input_y_shape,
        #     is_training=True,
        # )

        self.build(
            lr_decaying=self.LR_DECAYING,
            learning_rate=self.LEARNING_RATE,
            adam_beta1=self.ADAM_BETA1,
            adam_beta2=self.ADAM_BETA2,
            dropout=self.DROPOUT,
            bidirectional=self.BIDIRECTIONAL,
        )

    def input(
        self,
        batch_size=None,
        USE_TFRECORDS=True,
        input_f_dtype=tf.string,
        input_f_shape=(None, ),
        # input_x_dtype=tf.float32,
        # input_y_dtype=tf.float32,
        # input_x_shape=(None, 64, 64, 3),
        # input_y_shape=(None, 100),
        is_training=True,
        drop_remainder=False,
        ):
        """Input function.

        TBD

        Parameter
        ---------
        mode: (string) {'train', 'eval'}
            or any other mode you can think of.
            At training, we shuffle the data and have multiple epochs
        input_x: numpy.ndarray

        input_y: numpy.ndarray

        Example
        -------
        >>> vrnn = VRNN()
        >>> vrnn.input_fn('train', x_train, y_train)
        >>> eval_input_spec = input_fn('evaluate', x_test, y_test)

        """
        with tf.name_scope('input'):
            if batch_size is None:
                batch_size = self.BATCH_SIZE

            buffer_size = self.BUFFER_SIZE if is_training else 1

            # X_t = tf.placeholder(self.input_x_dtype, self.input_x_shape,
            #                      name='x_tensor_interface')
            #
            # Y_t = tf.placeholder(self.input_y_dtype, self.input_y_shape,
            #                      name='y_tensor_interface')

            F_t = tf.placeholder(
                self.input_f_dtype,
                self.input_f_shape,
                name='x_tensor_interface',
            )
            if USE_TFRECORDS:
                dataset = tf.data.Dataset.from_tensors(F_t)
                dataset = dataset.flat_map(
                    lambda filename: (
                        tf.data.TFRecordDataset(filename)
                    )
                )
                # dataset = tf.data.TFRecordDataset(X_t)

                feature_desc = {
                    'body': tf.VarLenFeature(tf.int64),
                    'title': tf.VarLenFeature(tf.int64),

                }

                def _parse_example(example_proto):
                    res = tf.parse_single_example(
                        example_proto,
                        feature_desc,
                    )
                    # res = res.values()  # Unpacking (res['body'].values, res['title'].values)
                    return res

                dataset = dataset.map(_parse_example)
                dataset = dataset.shuffle(
                    buffer_size,
                    reshuffle_each_iteration=True,
                )
                dataset = dataset.batch(
                    batch_size,
                    drop_remainder=drop_remainder,
                )

                def _extract_ids(parsed):
                    """
                    ids: tf.Tensor (2D).
                    len: tf.Tensor (1D).
                    """
                    x_ids = parsed['body']
                    y_ids = parsed['title']

                    x_ids_dense = tf.sparse_to_dense(
                        sparse_indices=x_ids.indices,
                        output_shape=x_ids.dense_shape,
                        sparse_values=x_ids.values,
                    )
                    y_ids_dense = tf.sparse_to_dense(
                        sparse_indices=y_ids.indices,
                        output_shape=y_ids.dense_shape,
                        sparse_values=y_ids.values,
                    )

                    x_seq_len = tf.where(tf.equal(x_ids_dense, 2))[:, 1]
                    y_seq_len = tf.where(tf.equal(y_ids_dense, 2))[:, 1]
                    # x_len = x_ids.shape[-1]
                    # y_len = y_ids.shape[-1]

                    return x_ids_dense, y_ids_dense, x_seq_len, y_seq_len

                dataset = dataset.map(_extract_ids)
                # parsed_dataset = parsed_dataset.flat_map(
                #     lambda parsed: tf.data.Dataset.from_tensor_slices(
                #         _extract_ids_with_len(parsed)
                #     )
                # )
                # dataset = dataset.flat_map(
                #     lambda data_x, data_y: tf.data.Dataset.zip(
                #         (
                #             tf.data.Dataset.from_tensors(data_x),
                #             tf.data.Dataset.from_tensors(data_y),
                #             tf.data.Dataset.from_tensors(tf.size(data_x)),
                #             tf.data.Dataset.from_tensors(tf.size(data_y)),
                #         )
                #     )#.repeat(repeat_num)
                # )
                # dataset = dataset.prefetch(buffer_size=batch_size * 1_000)

                data_op = dataset.make_initializable_iterator()
                data_init_op = data_op.initializer
                # next_batch = data_op.get_next()
                next_batch = data_op.get_next()
                X_batch, Y_batch, X_batch_length, Y_batch_length = next_batch

                print('[dtype] X: %s' % (X_batch.dtype))
                print('[shape] X: %s' % (X_batch.get_shape()))

                print('[dtype] Y: %s' % (Y_batch.dtype))
                print('[shape] Y: %s' % (Y_batch.get_shape()))

            else:

                raise NotImplementedError()

        # self._x_input_tensor = X_t
        # self._y_input_tensor = Y_t
        self._f_input_tensor = F_t
        self._x_batch_tensor = X_batch
        self._y_batch_tensor = Y_batch
        self._x_batch_length_tensor = X_batch_length
        self._y_batch_length_tensor = Y_batch_length
        self.next_batch = next_batch
        self.data_init_op = data_init_op


    def _uprint(*args, print_ok=True, **kwargs):
        """Print option interface.

        This function is equal to ``print`` function but added ``print_ok``
        option. This allows you to control printing in a function.

        Parameters
        ----------
        *args: whatever ``print`` allows.
          It is same as ``print`` does.

        print_ok: Boolean (default: True)
          An option whether you want to print something out or not.

        arg_zip: zip (default: None)
          A ``zip`` instance.

        """
        if print_ok:
            print(*args, **kwargs)

    def _print_scope(
        self,
        name,
        sep_length=50,
        ):
        print(
            '{{{name:^18s}}}'.format(name=name) + ' ' + '-' * sep_length
        )

    def _print_layer(
        self,
        name,
        input_shape=None,  # input_.get_shape(),
        output_shape=None,  # final_layer.get_shape(),
        ):

        string_format = " ".join(
            [
                "{name:20s}  |",
                "{input_shape:20s}",
                " -> ",
                "{output_shape:20s}",
                "|",
            ]
        )
        layer_str = string_format.format(
            name=name,
            input_shape=str(input_shape),
            output_shape=str(output_shape),
        )
        layer_str_len = len(layer_str)
        print('-' * layer_str_len)
        print(layer_str)
        print('-' * layer_str_len)


    def _print_parameter(
        self,
        parameter_dict,
        ):

        parameter_str = "\n".join(
            f"| {prmt_key:18s}: {str(prmt_value):>8s} |"
            #if isinstance(prmt_value, (int, float, complex))
            #else f"{prmt_key:15s}\t: {prmt_value:>8s}"
            for prmt_key, prmt_value in parameter_dict.items()
        )
        print(
            "=" * 7 + " Given Parameters " +  "=" * 7,
            parameter_str,
            "=" * 32,
            sep='\n',
        )

    def _load_gensim_embed(self, w2v_trained_binfile):

        # w2v_trained_file = 'data/news/news_bpe_200000_w2v.bin'

        if not os.path.isfile(w2v_trained_binfile):
            raise Exception(
                f'Pre-trained W2V "{w2v_trained_binfile}" not exist.'
            )
        else:
            embed = Word2Vec.load(w2v_trained_binfile)
            print('W2V Loaded: ', embed.wv.vectors.shape)

            return embed


    def _string_encoder(
            self,
            input_x,
            model_file=None,
            mode='sentence',
            name='string_encoder',
            ):

        if mode == 'sentence':
            encoded = tfspm.encode_dense(
                input_x,
                model_file=model_file,
                nbest_size=0,
                alpha=1.0,
                model_proto=None,
                reverse=self.REVERSE,
                add_bos=self.ADD_BOS,
                add_eos=self.ADD_EOS,
                out_type=tf.int32,
                name=name,
            )

        elif mode == 'word':

            encoded = tfspm.encode_dense(
                input_x,
                model_file=model_file,
                nbest_size=0,
                alpha=1.0,
                model_proto=None,
                reverse=False,
                add_bos=False,
                add_eos=False,
                out_type=tf.int32,
                name=name,
            )

        encoded_vector = encoded.values
        encoded_seq_length = encoded.sequence_length

        return encoded_vector, encoded_seq_length


    def _string_decoder(
            self,
            encoded_vector,
            sequence_length=None,
            model_file=None,
            name='string_decoder',
            ):

        decoded = tfspm.decode(
            encoded_vector,
            sequence_length=sequence_length,
            # sequence_length=tf.tile([8], [tf.shape(encoded_vector)[0]]),
            model_file=model_file,
            name=name,
        )

        return decoded

    def _cell_layer_norm_lstm(
        self,
        initializer=None,
        dropout=None,
        use_residual=True,
        is_training=True,
        name='lstm_cell',
        ):

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if initializer is None:
                initializer = tf.random_normal_initializer(0., .1)

            if dropout:
                keep_prob = tf.cond(
                    tf.cast(is_training, tf.bool),
                    lambda : tf.constant((1. - dropout)),
                    lambda : tf.constant(1.),
                    name='choose_prob_if_training'
                )
            else:
                keep_prob = tf.constant(1.),

            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.UNIT_NUM,
                #initializer=initializer,
                forget_bias=.5,
                input_size=None,
                activation=tf.tanh,
                layer_norm=True,
                norm_gain=1.0,
                norm_shift=0.0,
                dropout_keep_prob=keep_prob,
                dropout_prob_seed=None,
                #name='lstm_cell',
            )
            final_cell = cell


            if use_residual:
                # residualed = tf.nn.rnn_cell.ResidualWrapper(
                #     final_cell,
                #     residual_fn=lambda input_t, output_t: tf.concat(
                #         [input_t, output_t],
                #         axis=-1,
                #     ),
                # )
                # residualed = tf.nn.rnn_cell.ResidualWrapper(
                #     final_cell,
                #     residual_fn=lambda input_, output_:
                #         self._layer_bilinear(
                #             input_[1:],
                #             output_[-1],
                #         ),
                # )
                # final_cell = residualed
                final_cell = final_cell

        return final_cell


    def _cell_lstm(
        self,
        initializer=None,
        dropout=None,
        use_residual=True,
        is_training=True,
        name='lstm_cell',
        ):

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if initializer is None:
                initializer = tf.random_normal_initializer(0., .1)

            cell = tf.nn.rnn_cell.LSTMCell(
                self.UNIT_NUM,
                initializer=initializer,
                forget_bias=.5,
                name='lstm_cell',
            )
            final_cell = cell

            if dropout:

                keep_prob = tf.cond(
                    tf.cast(is_training, tf.bool),
                    lambda : tf.constant((1. - dropout)),
                    lambda : tf.constant(1.),
                    name='choose_prob_if_training'
                )
                # keep_prob = tf.case(
                #     {
                #         tf.equal(is_training, tf.constant(True)): tf.constant((1. - dropout)),
                #     },
                #     default=tf.constant(1.),
                #     name='choose_prob_if_training',
                #     exclusive=True,
                # )

                # TBD: tf.layers.dropout()
                dropoutted = tf.contrib.rnn.DropoutWrapper(
                    final_cell,
                    input_keep_prob=keep_prob,
                    output_keep_prob=keep_prob,
                    #name='dropout',
                )
                final_cell = dropoutted


            if use_residual:
                # residualed = tf.nn.rnn_cell.ResidualWrapper(
                #     final_cell,
                #     residual_fn=lambda input_t, output_t: tf.concat(
                #         [input_t, output_t],
                #         axis=-1,
                #     ),
                # )
                # residualed = tf.nn.rnn_cell.ResidualWrapper(
                #     final_cell,
                #     residual_fn=lambda input_, output_:
                #         self._layer_bilinear(
                #             input_[1:],
                #             output_[-1],
                #         ),
                # )
                # final_cell = residualed
                final_cell = final_cell

        return final_cell


    def _layer_linear(
        self,
        input_,
        output_size,
        is_training=True,
        stddev=0.02,
        bias_start=0.,
        batch_norm_ok=False,
        activation=None,
        dropout=.3,
        reuse=tf.AUTO_REUSE,
        name='linear',
        ):

        with tf.variable_scope(name, reuse=reuse):

            input_shape = input_.get_shape().as_list()

            with tf.variable_scope('linear', reuse=reuse):
                weight = tf.get_variable(
                    'weight',
                    shape=(input_shape[1], output_size),
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer(
                        #mean=0.,
                        stddev=stddev,
                    ),
                )
                bias = tf.get_variable(
                    'bias',
                    shape=(output_size,),
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(bias_start),
                )
                lineared = tf.matmul(
                    input_,
                    weight,
                )
                lineared_biased = tf.nn.bias_add(
                    lineared,
                    bias,
                    #data_format='NHWC',
                    name='linear_function',
                )
                final_layer = lineared_biased

            # (Optional) Batch Normalization
            if batch_norm_ok:
                # TBD: tf.layers.batch_normalization()
                with tf.variable_scope('batch_norm', reuse=reuse) as scope:
                    batch_normed = tf.contrib.layers.batch_norm(
                        final_layer,
                        epsilon=1e-5,
                        decay=.9,
                        updates_collections=None,
                        scale=True,
                        is_training=is_training,
                        scope=scope,
                    )
                final_layer = batch_normed

            # (Optional) Activation Layer
            if activation:
                activated = activation(
                    final_layer,
                    name='activation',
                )
                final_layer = activated

            # (Optional) Dropout Layer
            if dropout:

                keep_prob = tf.cond(
                    tf.cast(is_training, tf.bool),
                    lambda : tf.constant((1. - dropout)),
                    lambda : tf.constant(1.),
                    name='choose_prob_if_training'
                )
                # keep_prob = tf.case(
                #     {
                #         tf.equal(is_training, tf.constant(True)): tf.constant((1. - dropout)),
                #     },
                #     default=tf.constant(1.),
                #     name='choose_prob_if_training',
                #     exclusive=True,
                # )

                # TBD: tf.layers.dropout()
                dropoutted = tf.nn.dropout(
                    final_layer,
                    keep_prob=keep_prob,
                    name='dropout',
                )
                final_layer = dropoutted

        self._print_layer(
            name=name,
            input_shape=input_.get_shape(),
            output_shape=final_layer.get_shape(),
        )

        return final_layer


    def _layer_bilinear(
        self,
        input_a,
        input_b,
        use_bias=True,
        stddev=.05,
        bias_start=0.,
        batch_norm_ok=False,
        activation=None,
        dropout=None,
        is_training=True,
        reuse=tf.AUTO_REUSE,
        name='bilinear',
        ):
        with tf.variable_scope(name, reuse=reuse):

            input_a_shape = input_a.get_shape().as_list()
            input_b_shape = input_b.get_shape().as_list()

            weight = tf.get_variable(
                'weight',
                shape=(input_a_shape[-1], input_b_shape[0]),
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(
                    #mean=0.,
                    stddev=stddev,
                ),
            )

            if use_bias:
                bias_b = tf.get_variable(
                    'bias',
                    shape=(input_b_shape[0],),
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(bias_start),
                )
                bias_a = tf.get_variable(
                    'bias',
                    shape=(input_b_shape[0],),
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(bias_start),
                )

            lineared = weight @ bias_b

            if use_bias:
                lineared = tf.nn.bias_add(lineared, bias_b)

            bilineared = input_a @ lineared

            if use_bias:
                bilineared = tf.nn.bias_add(bilineared, bias_a)

            final_layer = bilineared

            # (Optional) Batch Normalization
            if batch_norm_ok:
                # TBD: tf.layers.batch_normalization()
                with tf.variable_scope('batch_norm', reuse=reuse) as scope:
                    batch_normed = tf.contrib.layers.batch_norm(
                        final_layer,
                        epsilon=1e-5,
                        decay=.9,
                        updates_collections=None,
                        scale=True,
                        is_training=is_training,
                        scope=scope,
                    )
                final_layer = batch_normed

            # (Optional) Activation Layer
            if activation:
                activated = activation(
                    final_layer,
                    name='activation',
                )
                final_layer = activated

            # (Optional) Dropout Layer
            if dropout:

                keep_prob = tf.cond(
                    tf.cast(is_training, tf.bool),
                    lambda : tf.constant((1. - dropout)),
                    lambda : tf.constant(1.),
                    name='choose_prob_if_training'
                )
                # keep_prob = tf.case(
                #     {
                #         tf.equal(is_training, tf.constant(True)): tf.constant((1. - dropout)),
                #     },
                #     default=tf.constant(1.),
                #     name='choose_prob_if_training',
                #     exclusive=True,
                # )

                # TBD: tf.layers.dropout()
                dropoutted = tf.nn.dropout(
                    final_layer,
                    keep_prob=keep_prob,
                    name='dropout',
                )
                final_layer = dropoutted

        self._print_layer(
            name=name,
            input_shape=[input_a_shape, input_b_shape],
            output_shape=final_layer.get_shape(),
        )

        return final_layer


    def _layer_dense(
        self,
        input_x,
        output_size,
        is_training=True,
        stddev=1.,
        bias_start=0.,
        reuse=tf.AUTO_REUSE,
        name='dense',
        ):

        with tf.variable_scope(name, reuse=reuse):
            densed = tf.layers.dense(
                input_x,
                units=output_size,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.,
                    stddev=stddev,
                ),
                bias_initializer=tf.constant_initializer(bias_start),
                name=name,
            )

        print(name, densed.get_shape())

        return densed

    def _graph_token_padder(
        self,
        input_x,
        input_y,
        value=-1.,
        name='padder',
        ):
        with tf.variable_scope(name):
            x_padding = tf.constant(
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                name='x_padding',
            )

            y_padding = tf.constant(
                [
                    [0, 0],
                    [1, 0],
                    [0, 0],
                ],
                name='y_padding',
            )

            input_x_padded = tf.pad(
                input_x,
                paddings=x_padding,
                mode='CONSTANT',
                constant_values=0,
                name='padded_x',
            )

            input_y_padded = tf.pad(
                input_y,
                paddings=y_padding,
                mode='CONSTANT',
                constant_values=value,
                name='padded_y',
            )
            print('Y has been padded with %s' % value)

        return input_x_padded, input_y_padded

    def _graph_embedding(
            self,
            trained_embeddings=None,
            name='embedding',
            reuse=tf.AUTO_REUSE,
            ):

        with tf.variable_scope(
            name or 'embedding',
            reuse=reuse,
            ) as scope:

            # embeddings = tf.get_variable(
            #     'embedding_vector',
            #     shape=(
            #         vocab_size,
            #         embedding_size,
            #     ),
            #     initializer=tf.random_uniform_initializer(-1., 1.),
            # )
            embeddings = tf.get_variable(
                'embedding_vector',
                shape=(
                    self.VOCAB_SIZE,
                    self.EMBEDDING_SIZE,
                ),
                initializer=tf.constant_initializer(trained_embeddings),
                trainable=False,
            )
            # W = tf.get_variable(name="W", shape=embedding.shape, initializer=tf.constant_initializer(embedding), trainable=False)

            # embeddings = tf.convert_to_tensor(
            #     trained_embeddings,
            # )
            # normalized_embeddings = tf.nn.l2_normalize(embeddings, 1)
            res_embeddings = embeddings
            print('embedding_vector: ', res_embeddings.shape)

            return res_embeddings


    def _graph_id2embedding_query(
            self,
            input_ids,
            embeddings=None,
            name='embedding_query',
            ):

        with tf.name_scope(name):
            embedded_word_ids = tf.nn.embedding_lookup(
                embeddings,
                input_ids,
            )
            return embedded_word_ids


    # def _graph_embedding2id_query(
    #         self,
    #         input_ids,
    #         embeddings=None,
    #         name='embedding_query',
    #         ):
    #
    #     with tf.name_scope(name):
    #         embedded_word_ids = tf.nn.embedding_lookup(
    #             embeddings,
    #             input_ids,
    #         )
    #         return embedded_word_ids
    #

    # def _graph_embedding_at_encoding(
    #     input_x_as_ids,
    #     vocab_size=None,
    #     embed_dim=150,
    #     reuse=tf.AUTO_REUSE,
    #     name='embedding',
    #     ):
    #
    #     with tf.variable_scope(
    #         name or 'embedding',
    #         reuse=reuse
    #         ) as scope:
    #          # Encoder embedding
    #          enc_embed_input = tf.contrib.layers.embed_sequence(
    #             input_x_as_ids,
    #             vocab_size=vocab_size,
    #             embed_dim=embed_dim,
    #             unique=False,
    #             initializer=None,
    #             regularizer=None,
    #             trainable=True,
    #             scope=scope,
    #             reuse=None,
    #         )

    def _graph_encoder(
        self,
        input_x,
        x_seq_length,
        reuse=tf.AUTO_REUSE,
        bidirectional=False,
        dropout=None,
        use_residual=True,
        use_normalize=True,
        is_training=True,
        name="encoder",
        ):

        #dynamic_batch_size = tf.shape(input_x)[0]
        with tf.variable_scope(
            name,
            reuse=reuse,
            dtype=input_x.dtype,
            ) as scope:

            if bidirectional:

                if use_normalize:
                    encoder_forward_cell = self._cell_layer_norm_lstm(
                        initializer=tf.random_normal_initializer(0., .1),
                        dropout=dropout,
                        use_residual=use_residual,
                        is_training=is_training,
                        name='encoder_forward_lstm_cell',
                    )
                    encoder_backward_cell = self._cell_layer_norm_lstm(
                        initializer=tf.random_normal_initializer(0., .1),
                        dropout=dropout,
                        use_residual=use_residual,
                        is_training=is_training,
                        name='encoder_backward_lstm_cell',
                    )

                else:
                    encoder_forward_cell = self._cell_lstm(
                        initializer=tf.random_normal_initializer(0., .1),
                        dropout=dropout,
                        use_residual=use_residual,
                        is_training=is_training,
                        name='encoder_forward_lstm_cell',
                    )
                    encoder_backward_cell = self._cell_lstm(
                        initializer=tf.random_normal_initializer(0., .1),
                        dropout=dropout,
                        use_residual=use_residual,
                        is_training=is_training,
                        name='encoder_backward_lstm_cell',
                    )

                encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=encoder_forward_cell,
                    cell_bw=encoder_backward_cell,
                    inputs=input_x,
                    sequence_length=x_seq_length,
                    #sequence_length=x_seq_length,#tf.tile([self.INPUT_SEQ_LENGTH], [self.BATCH_SIZE]),
                    time_major=self.TIME_MAJOR,
                    dtype=tf.float32,
                    scope=scope,
                )
                encoder_outputs = tf.concat(encoder_outputs, 2)

            else:

                encoder_basic_cell = self._cell_lstm(
                    initializer=tf.random_normal_initializer(0., .1),
                    dropout=dropout,
                    use_residual=use_residual,
                    is_training=is_training,
                    name='encoder_basic_lstm_cell',
                )

                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell=encoder_basic_cell,
                    inputs=input_x,
                    #sequence_length=x_seq_length,#tf.tile([self.INPUT_SEQ_LENGTH], [dynamic_batch_size]),
                    time_major=self.TIME_MAJOR,
                    dtype=tf.float32,
                    scope=scope,
                )

            self._print_layer(
                name=name,
                input_shape=input_x.get_shape(),
                output_shape=' '.join(
                    [
                        str(encoder_outputs.get_shape()),
                        str(
                            [
                                [
                                    tuple(state.c.get_shape().as_list()),
                                    tuple(state.h.get_shape().as_list()),
                                ]
                             for state in encoder_state],
                        ).replace('None', '?')
                    ]
                ),
            )

        return encoder_outputs, encoder_state

    def _graph_embedding_generator_at_decoding(
        target_letter_to_int,
        target_vocab_size,
        decoding_embedding_size,
        dec_embeddings,
        dec_input,
        name='embedding_dec',
        ):

        with tf.variable_scope(
            name or 'embedding_dec',
            ):
            # 1. Decoder Embedding
            target_vocab_size = len(target_letter_to_int)
            dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
            dec_embed_input = tf.nn.embedding_lookup(
                dec_embeddings,
                dec_input,
            )

            return dec_embed_input

    def _graph_decoder_cell(
        self,
        input_y,
        encoder_outputs,
        encoder_state,
        reuse=tf.AUTO_REUSE,
        #cell=None,
        #bidirectional=False,
        #use_attention=True,
        attention_method=None,
        dropout=None,
        use_residual=True,
        #projection_layer=None,
        is_training=True,
        name='decoder_cell',
        ):

        with tf.variable_scope(name, reuse=reuse):

            dynamic_batch_size = tf.shape(encoder_outputs)[0]

            # decoder_cell_0 = self._cell_lstm(
            #     initializer=tf.random_normal_initializer(0., .1),
            #     dropout=dropout,
            #     use_residual=use_residual,
            #     is_training=is_training,
            #     name='decoder_basic_lstm_cell_0',
            # )
            # decoder_cell_1 = self._cell_lstm(
            #     initializer=tf.random_normal_initializer(0., .1),
            #     dropout=dropout,
            #     use_residual=use_residual,
            #     is_training=is_training,
            #     name='decoder_basic_lstm_cell_1',
            # )
            decoder_cell_0 = self._cell_layer_norm_lstm(
                initializer=tf.random_normal_initializer(0., .1),
                dropout=dropout,
                use_residual=use_residual,
                is_training=is_training,
                name='decoder_basic_lstm_cell_0',
            )
            decoder_cell_1 = self._cell_layer_norm_lstm(
                initializer=tf.random_normal_initializer(0., .1),
                dropout=dropout,
                use_residual=use_residual,
                is_training=is_training,
                name='decoder_basic_lstm_cell_1',
            )
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                [decoder_cell_0, decoder_cell_1],
            )

            if attention_method == 'bahdanau':
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.UNIT_NUM,
                    memory=encoder_outputs,
                    memory_sequence_length=None,
                    normalize=self.USE_NORMALIZE,
                    probability_fn=None,  # Default: `tf.nn.softmax`
                    score_mask_value=None,
                    dtype=tf.float32,
                    name='BahdanauAttention',
                )
            elif attention_method == 'luong':
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.UNIT_NUM,
                    memory=encoder_outputs,
                    memory_sequence_length=None,
                    scale=self.USE_NORMALIZE,
                    probability_fn=None,  # Default: `tf.nn.softmax`
                    score_mask_value=None,
                    dtype=tf.float32,
                    name='LuongAttention',
                )
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell,
                attention_mechanism,
                attention_layer_size=self.UNIT_NUM,  #int(self.UNIT_NUM / 2),
                alignment_history=True,
                cell_input_fn=None,
                # A callable. The default is: lambda inputs, attention: array_ops.concat([inputs, attention], -1).
                output_attention=True,
                initial_cell_state=encoder_state,
                attention_layer=None,
                name='attention_layer',
            )

            attention_initial_state = attention_cell.zero_state(
                dtype=tf.float32,
                batch_size=dynamic_batch_size,
                #batch_size=true_batch_size * beam_width,
            )
            attention_initial_state = attention_initial_state.clone(
                cell_state=encoder_state,
            )


        return attention_cell, attention_initial_state


    # Decoder
    # def _graph_decoder_training(
    #     self,
    #     input_y,
    #     y_seq_length,
    #     encoder_state,
    #     cell=None,
    #     projection_layer=None,
    #     is_training=True,
    #     name='decoder_training',
    #     ):
    #
    #     decoder_cell = cell
    #     reuse = tf.AUTO_REUSE
    #     with tf.variable_scope(
    #         name,
    #         reuse=reuse,
    #         dtype=tf.float32,
    #         ) as scope:
    #
    #         # seq_length = tf.reshape(
    #         #     tf.tile(
    #         #         [self.MAX_OUTPUT_SEQ_LENGTH],
    #         #         [batch_size],
    #         #     ),
    #         #     [batch_size, ],
    #         # )
    #         # seq_length = tf.tile(
    #         #         [self.MAX_OUTPUT_SEQ_LENGTH],
    #         #         [y_seq_length],
    #         # )
    #
    #         training_helper = NumericalTrainingHelper(
    #             input_y,
    #             sequence_length=y_seq_length,
    #             time_major=self.TIME_MAJOR,
    #             name='training_helper',
    #         )
    #
    #         training_decoder = tf.contrib.seq2seq.BasicDecoder(
    #             cell=decoder_cell,
    #             helper=training_helper,
    #             initial_state=encoder_state,
    #             output_layer=projection_layer,
    #         )
    #
    #         (final_outputs,
    #          final_states,
    #          final_sequence_len) = tf.contrib.seq2seq.dynamic_decode(
    #             training_decoder,
    #             output_time_major=self.TIME_MAJOR,
    #             impute_finished=False,
    #             swap_memory=True,
    #             maximum_iterations=self.MAX_OUTPUT_SEQ_LENGTH,
    #             #parallel_iterations=32,
    #             scope=scope,
    #         )
    #         training_logits = final_outputs.rnn_output
    #         #print('training_logits: ', training_logits.get_shape())
    #
    #         print('\n[%s]: ' % name  + '-'*30)
    #         print(
    #
    #             input_y.get_shape(),
    #             '->',
    #             training_logits.get_shape(),
    #         )
    #         print('-'*50)
    #
    #         # self._print_layer(
    #         #     name=name,
    #         #     input_shape=(
    #         #         str(input_y.get_shape()) +
    #         #         str([state.shape.as_list() for state in encoder_state])
    #         #     ),
    #         #     output_shape=training_logits.get_shape(),
    #         # )
    #
    #         return training_logits
    #
    #
    # def _graph_decoder_inference(
    #     self,
    #     batch_size_tensor,
    #     encoder_state,
    #     cell=None,
    #     projection_layer=None,
    #     is_training=False,
    #     name='decoder_inference',
    #     ):
    #
    #     batch_size = batch_size_tensor
    #     decoder_cell = cell
    #     reuse = True
    #     with tf.variable_scope(
    #         name,
    #         reuse=reuse,
    #         dtype=tf.float32,
    #         ) as scope:
    #
    #         def _decoder_initialize_func():
    #             with tf.variable_scope('inference_decoder_init_func'):
    #                 finished = tf.tile([False], [batch_size])
    #                 start_inputs = tf.fill(
    #                     [
    #                         batch_size,
    #                         self.OUTPUT_CHANNEL,
    #                     ],
    #                     self.START_TOKEN,
    #                 )
    #                 return (finished, start_inputs)
    #
    #         def _decoder_sample_func(time, outputs, state):
    #             with tf.variable_scope('inference_decoder_sample_func'):
    #                 sample_ids = tf.identity(outputs)
    #                 return sample_ids
    #
    #         def _decoder_next_inputs_func(time, outputs, state, sample_ids):
    #             with tf.variable_scope('inference_decoder_next_input_func'):
    #                 next_time = time + 1
    #                 seq_ended = (next_time >= self.MAX_OUTPUT_SEQ_LENGTH)
    #                 all_ended = tf.reduce_all(seq_ended)
    #                 finished = tf.cond(
    #                     all_ended,
    #                     lambda : tf.tile([True], [batch_size]),
    #                     lambda : tf.tile([False], [batch_size]),
    #                 )
    #                 next_inputs = sample_ids
    #
    #                 return (finished, next_inputs, state)
    #
    #     inference_helper = tf.contrib.seq2seq.CustomHelper(
    #         initialize_fn=_decoder_initialize_func,
    #         sample_fn=_decoder_sample_func,
    #         next_inputs_fn=_decoder_next_inputs_func,
    #     )
    #     inference_decoder = tf.contrib.seq2seq.BasicDecoder(
    #         cell=decoder_cell,
    #         helper=inference_helper,
    #         initial_state=encoder_state,
    #         output_layer=projection_layer,
    #     )
    #
    #     (final_outputs,
    #      final_states,
    #      final_sequence_len) = tf.contrib.seq2seq.dynamic_decode(
    #         inference_decoder,
    #         output_time_major=self.TIME_MAJOR,
    #         impute_finished=False,
    #         swap_memory=True,
    #         maximum_iterations=self.MAX_OUTPUT_SEQ_LENGTH,
    #         #parallel_iterations=32,
    #         scope=scope,
    #     )
    #     inference_logits = final_outputs.rnn_output
    #     #print('inference_logits: ', inference_logits.get_shape())
    #
    #     print('\n[%s]: ' % name  + '-'*30)
    #     print(
    #         [state.shape.as_list() for state in encoder_state],
    #         '->',
    #         inference_logits.get_shape(),
    #     )
    #     print('-'*50)
    #
    #     return inference_logits


    # Decoder
    def _graph_attn_decoder_training(
        self,
        input_y,
        y_seq_length,
        encoder_state,
        cell=None,
        projection_layer=None,
        is_training=True,
        reuse=tf.AUTO_REUSE,
        name='decoder_attn_training',
        ):

        decoder_cell = cell
        with tf.variable_scope(
            name,
            reuse=reuse,
            dtype=tf.float32,
            ) as scope:

            # seq_length = tf.reshape(
            #     tf.tile(
            #         [self.MAX_OUTPUT_SEQ_LENGTH],
            #         [batch_size],
            #     ),
            #     [batch_size, ],
            # )
            # seq_length = tf.tile(
            #         [self.MAX_OUTPUT_SEQ_LENGTH],
            #         [y_seq_length],
            # )

            # training_helper = NumericalTrainingHelper(
            #     input_y,
            #     sequence_length=y_seq_length,
            #     time_major=self.TIME_MAJOR,
            #     name='training_helper',
            # )
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=input_y,
                sequence_length=y_seq_length,
                time_major=self.TIME_MAJOR,
                name='training_helper',
            )

            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=training_helper,
                initial_state=encoder_state,
                output_layer=projection_layer,
            )

            (final_outputs,
             final_states,
             final_sequence_len) = tf.contrib.seq2seq.dynamic_decode(
                training_decoder,
                output_time_major=self.TIME_MAJOR,
                impute_finished=True,
                # swap_memory=True,
                maximum_iterations=self.MAX_OUTPUT_SEQ_LENGTH,
                #parallel_iterations=32,
                scope=scope,
            )
            training_ids = final_outputs.sample_id
            training_logits = final_outputs.rnn_output
            training_alignments = final_states.alignment_history.stack()
            #print('training_logits: ', training_logits.get_shape())

            self._print_layer(
                name=name,
                input_shape=' '.join(
                    [
                        str(input_y.get_shape()),
                        str(
                            # [tuple(tf.shape(state).shape.as_list())
                            #  for state in encoder_state],
                            [
                                [
                                    tuple(state.c.get_shape().as_list()),
                                    tuple(state.h.get_shape().as_list()),
                                ]
                             for state in encoder_state.cell_state],
                        ).replace('None', '?')
                    ]
                ),
                output_shape=training_logits.get_shape(),
            )

            return training_ids, training_logits, training_alignments


    def _graph_attn_decoder_inference(
        self,
        dynamic_batch_size,
        encoder_state,
        embedding=None,
        cell=None,
        projection_layer=None,
        is_training=False,
        reuse=True,
        name='decoder_attn_inference',
        ):

        #batch_size = batch_size_tensor
        with tf.variable_scope(
            name,
            reuse=reuse,
            dtype=tf.float32,
            ) as scope:

            #dynamic_batch_size = tf.shape(encoder_state[0])[0] # LSTMCell

            with tf.name_scope('inference_helper'):
                # def _decoder_initialize_func():
                #     with tf.variable_scope('inference_decoder_init_func'):
                #         finished = tf.tile([False], [dynamic_batch_size])
                #         start_inputs = tf.fill(
                #             [
                #                 dynamic_batch_size,
                #                 self.OUTPUT_CHANNEL,
                #             ],
                #             self.START_TOKEN,
                #         )
                #         return (finished, start_inputs)
                #
                # def _decoder_sample_func(time, outputs, state):
                #     with tf.variable_scope('inference_decoder_sample_func'):
                #         sample_ids = tf.identity(outputs)
                #         return sample_ids
                #
                # def _decoder_next_inputs_func(
                #     time,
                #     outputs,
                #     state,
                #     sample_ids
                #     ):
                #     with tf.variable_scope('inference_decoder_next_input_func'):
                #         next_time = time + 1
                #         seq_ended = (next_time >= self.MAX_OUTPUT_SEQ_LENGTH)
                #         all_ended = tf.reduce_all(seq_ended)
                #         finished = tf.cond(
                #             all_ended,
                #             lambda : tf.tile([True], [dynamic_batch_size]),
                #             lambda : tf.tile([False], [dynamic_batch_size]),
                #         )
                #         next_inputs = sample_ids
                #
                #         return (finished, next_inputs, state)

        #     inference_helper = tf.contrib.seq2seq.CustomHelper(
        #         initialize_fn=_decoder_initialize_func,
        #         sample_fn=_decoder_sample_func,
        #         next_inputs_fn=_decoder_next_inputs_func,
        # )
                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=embedding,
                    start_tokens=tf.tile([1], [dynamic_batch_size]),  # tf.ones_like(x_seq_length, dtype=tf.int64)
                    end_token=2,  # [2],
                )

            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cell,
                helper=inference_helper,
                initial_state=encoder_state,
                output_layer=projection_layer,
            )

        (final_outputs,
         final_states,
         final_sequence_len) = tf.contrib.seq2seq.dynamic_decode(
            inference_decoder,
            output_time_major=self.TIME_MAJOR,
            impute_finished=True,
            # swap_memory=True,
            maximum_iterations=self.MAX_OUTPUT_SEQ_LENGTH,
            #parallel_iterations=32,
            scope=scope,
        )
        inference_ids = final_outputs.sample_id
        inference_logits = final_outputs.rnn_output
        inference_alignments = final_states.alignment_history.stack()
        #print('inference_logits: ', inference_logits.get_shape())

        self._print_layer(
            name=name,
            input_shape=' '.join(
                [
                    str(
                        [
                            [
                                tuple(state.c.get_shape().as_list()),
                                tuple(state.h.get_shape().as_list()),
                            ]
                         for state in encoder_state.cell_state],
                    ).replace('None', '?')
                ]
            ),
            # output_shape=inference_logits.get_shape(),
            output_shape=' '.join(
                [
                    str(
                        [
                            inference_ids.get_shape().as_list(),# inference_ids.get_shape(),
                            inference_logits.get_shape().as_list(),
                        ],
                    ).replace('None', '?')
                ]
            ),
        )

        return inference_ids, inference_logits, inference_alignments


    def build(
        self,
        learning_rate=.0002,
        lr_decaying=True,
        adam_beta1=.5,
        adam_beta2=.8,
        bidirectional=True,
        dropout=.3,
        ):
        reuse_ok =  tf.AUTO_REUSE  # (mode != 'train')
        print('\n[seq2seq_model]: ' + '='*30)

        # Objective Functions ================================================
        with tf.variable_scope('seq2seq_model', reuse=reuse_ok):


            with tf.name_scope('input_placeholders'):
                # Input_x_encoder = tf.placeholder(
                #     self.input_x_dtype,
                #     self.input_x_shape,
                #     name='input_x_encoder',
                # )
                # Input_y_decoder = tf.placeholder(
                #     self.input_y_dtype,
                #     self.input_y_shape,
                #     name='input_y_decoder',
                # )
                if self.USE_TFRECORDS:
                    Input_x_encoder = tf.placeholder(
                        tf.int32,
                        shape=(None, None),
                        name='input_x_encoder',
                    )
                    Input_y_decoder = tf.placeholder(
                        tf.int32,
                        shape=(None, None),
                        name='input_y_decoder',
                    )
                    # Input_x_encoder_final = tf.expand_dims(
                    #     Input_x_encoder,
                    #     axis=-1,
                    # )
                    # Input_y_decoder_final = tf.expand_dims(
                    #     Input_y_decoder,
                    #     axis=-1,
                    # )
                    Input_x_encoder_final = Input_x_encoder
                    Input_y_decoder_final = Input_y_decoder
                else:
                    Input_x_encoder = tf.placeholder(
                        tf.int32,
                        shape=(None, 1),
                        name='input_x_encoder',
                    )
                    Input_y_decoder = tf.placeholder(
                        tf.int32,
                        shape=(None, 1),
                        name='input_y_decoder',
                    )
                    Input_x_encoder_final = Input_x_encoder
                    Input_y_decoder_final = Input_y_decoder

                X_seq_length = tf.placeholder(
                    tf.int32,
                    (None, ),
                    #self.input_x_shape[:1],
                    #(None, ),
                    name='X_seq_length',
                )
                Y_seq_length = tf.placeholder(
                    tf.int32,
                    (None, ),
                    # self.input_y_shape[:1],
                    #(None, ),
                    name='Y_seq_length',
                )

            with tf.name_scope('parameter_placeholders'):
                Start_learning_rate_tensor = tf.placeholder(
                    tf.float32,
                    [],
                    name='start_learning_rate_tensor',
                )
                Decay_lr_step_tensor = tf.placeholder(
                    tf.int32,
                    [],
                    name='decay_lr_step_tensor',
                )


            # (Input_x_encoder_padded,
            #  Input_y_decoder_padded) = self._graph_token_padder(
            #     Input_x_encoder_final,
            #     Input_y_decoder_final,
            #     value=self.START_TOKEN,
            #     name='padder',
            # )
            # Input_x_encoder_padded = Input_x_encoder_final
            # Input_y_decoder_padded = Input_y_decoder_final

            tf_embed = self._graph_embedding(
                trained_embeddings=self.embed_vector
            )

            Input_x_encoder_padded = self._graph_id2embedding_query(
                input_ids=Input_x_encoder_final,
                embeddings=tf_embed,
                name='input_x_id2vec',
            )
            Input_y_decoder_padded = self._graph_id2embedding_query(
                input_ids=Input_y_decoder_final,
                embeddings=tf_embed,
                name='input_y_id2vec',
            )

            print('Input_x_encoder_padded: ', Input_x_encoder_padded.shape)
            encoder_outputs, encoder_state = self._graph_encoder(
                Input_x_encoder_padded,
                X_seq_length,
                bidirectional=bidirectional,
                dropout=dropout,
                use_residual=self.USE_RESIDUAL,
                name="encoder",
            )

            attention_cell, attention_initial_state = self._graph_decoder_cell(
                Input_y_decoder_padded,
                encoder_outputs,
                encoder_state,
                reuse=tf.AUTO_REUSE,
                #cell=None,
                #bidirectional=False,
                #use_attention=self.USE_ATTENTION,
                attention_method=self.ATTENTION_METHOD,
                dropout=dropout,
                use_residual=self.USE_RESIDUAL,
                #projection_layer=None,
                is_training=True,
                name='decoder_cell',
            )

            projection_layer = tf.layers.Dense(
                self.OUTPUT_CHANNEL,
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                # kernel_regularizer=tf.contrib.layers.l2_regularizer(
                #     scale=0.1,
                # ),
                bias_initializer=tf.constant_initializer(.0),
                activation=self.PROJ_ACTIVATION,
                name='projection_layer',
            )

            with tf.name_scope('dynamic_batch_size'):
                dynamic_batch_size = tf.shape(Input_x_encoder)[0]


            (
                decoder_training_ids,
                decoder_training_logits,
                decoder_training_aligns,
            ) = self._graph_attn_decoder_training(
                Input_y_decoder_padded,
                y_seq_length=Y_seq_length,
                encoder_state=attention_initial_state,
                cell=attention_cell,
                projection_layer=projection_layer,
                is_training=True,
                reuse=tf.AUTO_REUSE,
                name='decoder_attn_training',
            )

            (
                decoder_inference_ids,
                decoder_inference_logits,
                decoder_inference_aligns,
            ) = self._graph_attn_decoder_inference(
                dynamic_batch_size,
                encoder_state=attention_initial_state,
                embedding=tf_embed,
                cell=attention_cell,
                projection_layer=projection_layer,
                is_training=False,
                reuse=True,
                name='decoder_attn_inference',
            )

            decoder_inference_strings = self._string_decoder(
                decoder_inference_ids,
                # sequence_length=decoder_inference_ids.shape[-1],
                sequence_length=tf.tile([self.MAX_OUTPUT_SEQ_LENGTH], [dynamic_batch_size]),
                model_file=self.SP_MODEL_DIR,
                name='decoder_inference_strings',
            )

        # ====================================================================


        # Optimization =======================================================

        with tf.variable_scope('seq2seq_optimization'):

            print('Y_seq_length: ', Y_seq_length.shape)
            # Create the weights for sequence_loss
            decoder_y_masks = tf.sequence_mask(
                Y_seq_length,
                self.MAX_OUTPUT_SEQ_LENGTH,  # tf.tile([self.MAX_OUTPUT_SEQ_LENGTH], [dynamic_batch_size]), # max_target_sequence_length,
                dtype=tf.float32,
                name='output_masks',
            )
            print('decoder_y_masks: ', decoder_y_masks.shape)
            # loss = tf.losses.mean_squared_error(
            #         labels=Input_y_decoder_padded,
            #         predictions=decoder_training_logits,
            # )# * 100
            #
            # valid_loss = tf.losses.mean_squared_error(
            #         labels=Input_y_decoder_padded,
            #         predictions=decoder_inference_logits,
            # )# * 100

            # loss = tf.reduce_mean(
            #     tf.losses.sparse_softmax_cross_entropy(
            #         logits=decoder_training_logits,
            #         labels=Input_y_decoder_padded,
            #         weights=decoder_y_masks,
            #     )
            # )
            # valid_loss = tf.reduce_mean(
            #     tf.losses.sparse_softmax_cross_entropy(
            #         logits=decoder_inference_logits,
            #         labels=Input_y_decoder_padded,
            #         weights=decoder_y_masks,
            #     )
            # )

            loss = tf.contrib.seq2seq.sequence_loss(
                logits=decoder_training_logits,  # tf.expand_dims(decoder_training_ids, -1),
                targets=Input_y_decoder_final,  # Input_y_decoder_padded,
                weights=decoder_y_masks,  # (dynamic_batch_size, Y_seq_length),
                # average_across_timesteps=True,
                # average_across_batch=False,
                # softmax_loss_function=None,
            )
            valid_loss = tf.contrib.seq2seq.sequence_loss(
                logits=decoder_inference_logits,  # tf.expand_dims(decoder_inference_ids, -1),
                targets=Input_y_decoder_final,  # Input_y_decoder_padded,
                weights=decoder_y_masks,  # (dynamic_batch_size, Y_seq_length),
                # average_across_timesteps=True,
                # average_across_batch=False,
                # softmax_loss_function=None,
            )

            summary_loss = tf.summary.scalar(
                'loss',
                loss,
            )

            summary_op = tf.summary.merge([
                # summary_output_loss,
                # summary_reg_loss,
                summary_loss,
            ])

            global_step = tf.Variable(
                0,
                trainable=False,
                name='global_step',
            )

            if lr_decaying:
                learning_rate_decay = tf.train.exponential_decay(
                    learning_rate=Start_learning_rate_tensor,
                    global_step=global_step,
                    decay_steps=Decay_lr_step_tensor,
                    decay_rate=.9,
                    staircase=True,
                    # decay_steps=20,
                    # decay_rate=.0005,
                    # staircase=True,
                    name='learning_rate_decay',
                )
            else:
                learning_rate_decay = Start_learning_rate_tensor

            summary_learning_rate_decay = tf.summary.scalar(
                'learning_rate',
                learning_rate_decay,
            )

            optimizer = tf.train.AdamOptimizer(
                #learning_rate=self.LEARNING_RATE,
                learning_rate=learning_rate_decay,
                #learning_rate=Learning_rate_tensor,
                beta1=self.ADAM_BETA1,
                beta2=self.ADAM_BETA2,
                epsilon=1e-08,
                name='optimizer',
            )


            train_op = optimizer.minimize(
                loss,
                #var_list=None,
                global_step=global_step,
            )

        variable_seq2seq = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='seq2seq',
        )
        variable_init_op = tf.group(
            *[
                tf.variables_initializer(
                    var_list = tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope='seq2seq_model',
                    )
                ),
                tf.variables_initializer(
                    var_list = tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope='seq2seq_optimization',
                    )
                ),
            ],
        )

        # ====================================================================

        with tf.variable_scope("metrics_seq2seq", reuse=reuse_ok):
            metrics_train = {
                'Train_loss': tf.metrics.mean(loss),
                'Learning_rate': tf.metrics.mean(learning_rate_decay),

            }
            metrics_valid = {
                'Valid_loss': tf.metrics.mean(valid_loss),
            }

        # Group the update ops for the tf.metrics
        update_metrics_op_train = tf.group(
            *[op for _, op in metrics_train.values()]
        )
        update_metrics_op_valid = tf.group(
            *[op for _, op in metrics_valid.values()]
        )

        # Get the op to reset the local variables used in tf.metrics
        metrics_init_op = tf.variables_initializer(
            var_list=tf.get_collection(
                tf.GraphKeys.LOCAL_VARIABLES,
                scope="metrics_seq2seq",
            ),
            name='metrics_init_op',
        )

        self.train_op = train_op

        self.variable_seq2seq = variable_seq2seq
        self.variable_init_op = variable_init_op

        self.metrics_train = metrics_train
        self.metrics_valid = metrics_valid
        self.update_metrics_op_train = update_metrics_op_train
        self.update_metrics_op_valid = update_metrics_op_valid
        self.metrics_init_op = metrics_init_op

        self.summary_op = summary_op

        self.loss = loss

        self.Input_X = Input_x_encoder
        self.Input_Y = Input_y_decoder
        self.X_seq_length = X_seq_length
        self.Y_seq_length = Y_seq_length
        self.Start_learning_rate_tensor = Start_learning_rate_tensor
        self.Decay_lr_step_tensor = Decay_lr_step_tensor
        self.summary_learning_rate_decay = summary_learning_rate_decay

        self.encoder = encoder_outputs, encoder_state
        self.decoder = decoder_inference_logits
        self.decoder_alignment = decoder_inference_aligns
        self.decoder_inference_strings = decoder_inference_strings


    def train(
        self,
        # input_x=None,
        # input_y=None,
        USE_TFRECORDS=True,
        file_list=None,
        batch_size=None,
        step_limit=None,
        drop_remainder=False,
        epoch_num=2,
        validation_ratio=.2,
        learning_rate=.0002,
        decay_lr_step=2000,
        model_save_dir='./model_save_seq2seq',
        pre_trained_path=None,
        verbose=False,
        writer=None,
        ):
        """Short Description.

        TBD

        Attributes
        ----------
        TBD


        Example
        -------
        >>> "TBD"
        >>> train_fn(x_train, y_train, train_model_spec,
                 3, None, hyper_param,
                 model_save_dir='model_save', pre_trained_path=None,
                 verbose=True)

        >>> train_fn(x_train, y_train, train_model_spec,
                 10, None, hyper_param,
                 model_save_dir='model_save', pre_trained_path='model_save')

        """
        metrics_train = self.metrics_train
        metrics_valid = self.metrics_valid

        # --------------------------------------------------------------------
        #global_step = tf.train.get_global_step()

        self.LEARNING_RATE = learning_rate
        self.DECAY_LR_STEP = decay_lr_step
        self.VALIDATION_RATIO = validation_ratio

        try:
            if not self.VALIDATION_RATIO or not (0 < self.VALIDATION_RATIO < 1):
                raise NoValidationException(self.VALIDATION_RATIO)
            else:
                pass
                # print("Training Parameter\n" +
                #       "'VALIDATION_RATIO=%s'" % self.VALIDATION_RATIO
                # )
        except KeyError:
            raise NoValidationException(
                'validation_ratio',
                None,
            )
        except NoValidationException as err:
            # print(err)
            self.VALIDATION_RATIO = 0

        if batch_size is None:
            batch_size_int = self.BATCH_SIZE
        else:
            batch_size_int = int(batch_size)


        parameter_dict = {
            'VALIDATION_RATIO': self.VALIDATION_RATIO,
            'DROPOUT': self.DROPOUT,
            'LEARNING_RATE': self.LEARNING_RATE,
            'BATCH_SIZE': batch_size_int,
            'EPOCH_NUM': epoch_num,
            'ADAM_BETA1': self.ADAM_BETA1,
            'ADAM_BETA2': self.ADAM_BETA2,
        }
        self._print_parameter(parameter_dict)


        self.input(
            batch_size=batch_size_int,
            USE_TFRECORDS=True,
            input_f_dtype=tf.string,
            input_f_shape=(None, ),
            # input_x_dtype=self.input_x_dtype,
            # input_y_dtype=self.input_y_dtype,
            # input_x_shape=self.input_x_shape,
            # input_y_shape=self.input_y_shape,
            is_training=True,
            drop_remainder=drop_remainder,
        )


        # Initialize tf.Saver instances to save weights during training
        last_saver = tf.train.Saver(
            var_list=self.variable_seq2seq,
            max_to_keep=2,  # will keep last 5 epochs as default
            keep_checkpoint_every_n_hours=1.,
            name='saver',
        )
        begin_at_epoch = 0

        with tf.Session() as sess:


            if pre_trained_path is None:
                if os.path.isdir(model_save_dir):
                    print(
                        "'%s' Exists: Remove the old one and create a new one.\n" %
                        model_save_dir
                    )
                    shutil.rmtree(model_save_dir, ignore_errors=False)
                os.makedirs(model_save_dir, exist_ok=False)

            # Reload weights from directory if specified
            elif pre_trained_path is not None:
                #logging.info("Restoring parameters from {}".format(restore_from))
                if os.path.isdir(pre_trained_path):
                    last_save_path = os.path.join(
                        pre_trained_path,
                        'last_weights',
                    )
                    saved_model = tf.train.latest_checkpoint(last_save_path)
                elif not os.path.isdir(pre_trained_path) or not saved_model:
                    print(
                        'Warning: ' +
                        'Pre-trained model not Found. ' +
                        'Trying to reset...'
                    )
                else:
                    begin_at_epoch = int(saved_model.split('-')[-1])
                    epoch_num = begin_at_epoch + epoch_num

                    last_saver.restore(sess, saved_model)
                    print('Pre-trained model loaded')
                    print("Restoring parameters from {}".format(saved_model))


            # For TensorBoard (takes care of writing summaries to files)
            train_writer = tf.summary.FileWriter(
                logdir=os.path.join(
                    model_save_dir,
                    'train_summaries',
                ),
                graph=sess.graph,
            )
            if self.VALIDATION_RATIO:
                valid_writer = tf.summary.FileWriter(
                    logdir=os.path.join(
                        model_save_dir,
                        'valid_summaries',
                        #'train_summaries',
                    ),
                    graph=sess.graph,
                )


            # Initialize model variables
            sess.run(self.variable_init_op)

            for epoch in range(begin_at_epoch, epoch_num):

                # Load the training dataset into the pipeline
                # and initialize the metrics local variables
                sess.run(
                    self.data_init_op,
                    feed_dict={
                        self._f_input_tensor: file_list,
                        #self._y_input_tensor: input_y,
                    }
                )
                sess.run(self.metrics_init_op)

                if not verbose:
                    epoch_msg = "\rEpoch %d/%d " % (epoch + 1, epoch_num)
                    sys.stdout.flush()
                else:
                    epoch_msg = "Epoch %d/%d \n" % (epoch + 1, epoch_num)
                sys.stdout.write(epoch_msg)

                # BATCH : Optimized by each chunk
                batch_num = 0

                total_len = sum(
                    1
                    for _ in tf.python_io.tf_record_iterator(file_list[0])
                )
                batch_len = int(np.ceil(total_len / batch_size_int))

                if self.VALIDATION_RATIO:
                    valid_len = max(int(batch_len * self.VALIDATION_RATIO), 1)
                    train_len = batch_len - valid_len
                else:
                    train_len = batch_len

                # global_step = tf.train.get_or_create_global_step(
                #     graph=sess.graph,
                # )
                global_step = epoch + 1

                batch_remains_ok = True
                while batch_remains_ok and (batch_num <= batch_len):
                    try:
                        for batch in range(train_len):

                            #X_batch, Y_batch = sess.run(self.next_batch)
                            (
                                X_batch,
                                Y_batch,
                                X_seq_length_batch,
                                Y_seq_length_batch,
                            ) = sess.run(self.next_batch)
                            # X_seq_length_batch = [len(x) for x in X_batch]
                            # Y_seq_length_batch = [len(y) for y in Y_batch]
                            # print(X_seq_length_batch)
                            (_,
                             summary_str_train,
                             summary_learning_rate,
                              _) = sess.run(
                                [
                                    self.train_op,
                                    self.summary_op,
                                    self.summary_learning_rate_decay,
                                    self.update_metrics_op_train,
                                ],
                                feed_dict={
                                    self.Input_X: X_batch,
                                    self.Input_Y: Y_batch,
                                    self.X_seq_length: X_seq_length_batch,
                                    self.Y_seq_length: Y_seq_length_batch,
                                    self.Start_learning_rate_tensor: self.LEARNING_RATE,
                                    self.Decay_lr_step_tensor: self.DECAY_LR_STEP,
                                    #self.Learning_rate_tensor: learning_rate,
                                },
                            )
                            # -----------------------------------------------

                            # Write summaries for tensorboard
                            # writer.add_summary(summ, global_step_val)

                            batch_num += 1

                            if verbose:

                                batch_pct = int(20 * batch_num / train_len)
                                batch_bar = "[%s]  " % (("#" * batch_pct) + ("-" * (20 - batch_pct)))
                                batch_msg = "\rBatch [%s/%s]  " % (batch_num, train_len)
                                #batch_err = 'Loss: %.5f ' % loss_val

                                batch_msg = batch_msg + batch_bar# + batch_err

                                sys.stdout.flush()
                                sys.stdout.write(batch_msg)
                            # -----------------------------------------------

                        if self.VALIDATION_RATIO:

                            for valid in range(valid_len):
                                (X_batch,
                                 Y_batch,
                                 X_seq_length_batch,
                                 Y_seq_length_batch) = sess.run(self.next_batch)

                                summary_str_valid, _ = sess.run(
                                    [
                                        self.summary_op,
                                        self.update_metrics_op_valid,
                                    ],
                                    feed_dict={
                                        self.Input_X: X_batch,
                                        self.Input_Y: Y_batch,
                                        self.X_seq_length: X_seq_length_batch,
                                        self.Y_seq_length: Y_seq_length_batch,
                                    },
                                )
                        else:
                            continue

                            # -----------------------------------------------

                    except tf.errors.OutOfRangeError:
                        batch_remains_ok = False

                        #result_msg = "finished.\n"
                        #sys.stdout.write(result_msg)
                        continue

                train_writer.add_summary(
                    summary_str_train,
                    epoch,
                )
                train_writer.add_summary(
                    summary_learning_rate,
                    epoch,
                )

                if self.VALIDATION_RATIO:
                    valid_writer.add_summary(
                        summary_str_valid,
                        epoch,
                    )

                if verbose:

                    metrics_values_train = {k: value[0]
                        for k, value in metrics_train.items()}
                    metrics_values_valid = {k: value[0]
                        for k, value in metrics_valid.items()}
                    metrics_res_train, metrics_res_valid = sess.run([
                        metrics_values_train,
                        metrics_values_valid,
                    ])
                    metrics_res = {
                        **metrics_res_train,
                        **metrics_res_valid,
                    }
                    metrics_str = "  ".join(
                        "{metric_key}: {metric_value:.5f}".format(
                                metric_key=k,
                                metric_value=value,
                        ) for k, value in metrics_res.items()
                    )
                    # print(
                    #     "\n-- Metrics -- ",
                    #     metrics_str,
                    #     sep='\n',
                    # )
                    sys.stdout.write(metrics_str + '\n')

                else:

                    if epoch == epoch_num-1:

                        metrics_values_train = {k: value[0]
                            for k, value in metrics_train.items()}
                        metrics_values_valid = {k: value[0]
                            for k, value in metrics_valid.items()}
                        metrics_res_train, metrics_res_valid = sess.run([
                            metrics_values_train,
                            metrics_values_valid,
                        ])
                        metrics_res = {
                            **metrics_res_train,
                            **metrics_res_valid,
                        }
                        metrics_str = "  ".join(
                            "{metric_key}: {metric_value:.5f}".format(
                                    metric_key=k,
                                    metric_value=value,
                            ) for k, value in metrics_res.items()
                        )

                # Save weights
                if model_save_dir:
                    last_save_path = os.path.join(
                        model_save_dir,
                        'last_weights',
                        'after-epoch',
                    )
                    last_saver.save(
                        sess,
                        last_save_path,
                        global_step=epoch + 1,
                    )
                    # if verbose:
                    #     print(
                    #         'Model Saved >> %s' %
                    #         (last_save_path)
                    #     )

        print('\nTraining has been Finished.')
        print(
            "Metrics " +  "=" * 20,
            metrics_str,
            sep='\n',
        )
        print(
            'Latest Model Saved >> %s-%s' %
            (last_save_path, epoch + 1)
        )


    def evaluate(
        self,
        input_x=None,
        pre_trained_path='./model_save_seq2seq',
        target_epoch=None,
        ):

        dropout = None
        assert pre_trained_path is not None, "`pre_trained_path` is mandatory."
        #global_step = tf.train.get_global_step()

        if self.decoder is None:
            tf.reset_default_graph()
            self.build(
                learning_rate = self.LEARNING_RATE,
                adam_beta1 = self.ADAM_BETA1,
                dropout=self.DROPOUT,
            )

        with tf.Session() as sess:
            ## Initialize model variables
            sess.run(self.variable_init_op)

            # Reload weights from directory if specified
            if pre_trained_path is not None:
                #logging.info("Restoring parameters from {}".format(restore_from))
                if os.path.isdir(pre_trained_path):
                    last_save_path = os.path.join(
                        pre_trained_path,
                        'last_weights',
                    )

                    if target_epoch:
                        saved_model = ''.join(
                            [
                                last_save_path + '/',
                                'after-epoch-',
                                str(target_epoch),
                            ],
                        )
                        last_saver = tf.train.import_meta_graph(
                            saved_model + '.meta'
                        )
                    else:
                        last_saver = tf.train.Saver()
                        saved_model = tf.train.latest_checkpoint(
                            last_save_path,
                        )
                    #begin_at_epoch = int(saved_model.split('-')[-1])
                    last_saver.restore(sess, saved_model)

            if self.BIDIRECTIONAL:
                pred_str, pred, aligns = sess.run(
                    [
                        self.decoder_inference_strings,
                        self.decoder,
                        self.decoder_alignment,
                    ],
                    feed_dict={
                        self.Input_X: input_x,
                        self.X_seq_length: [len(x) for x in input_x],
                    }
                )
                # TIME_MAJOR: True -> False
                aligns = np.swapaxes(aligns, 0, 1)
                # Transpose for X:Y shape
                #aligns = np.swapaxes(aligns, 1, 2)

                # self.embed_vector
                return pred_str, pred, aligns

            else:
                pred = sess.run(
                    self.decoder,
                    feed_dict={
                        self.Input_X: input_x,
                        self.X_seq_length: [len(x) for x in input_x],
                    }
                )
                return pred

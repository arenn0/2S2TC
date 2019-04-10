import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import io
import json
import bert_optimization

import collections
Embedding = "Bert"
# Embedding = "ELMo"
# Embedding = "Doc2Vec"
# Embedding = "word2vec"
# Embedding = "GloVe"
# Embedding = "fastText"

import bert
import string

TRAIN = True

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = bert_modeling.BertModel(
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
      ) = bert_modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
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

      train_op = bert_optimization.create_optimizer(
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


if Embedding == "Bert":
    import bert
    import bert_modeling
    import bert_tokenization
    max_seq_length = 128
    d = 768
if Embedding == "GloVe":
    d = 300
    path = "embeddings/glove.840B.300d.txt"
elif Embedding == "fastText":
    d = 300
    path = "embeddings/crawl-300d-2M-subword.vec"
elif Embedding == "ELMo":
    # ELMo = "embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
    d = 1024
elif Embedding == "Doc2Vec":
    d = 300
elif Embedding == "word2vec":
    d = 300
    from gensim.models import KeyedVectors

embeddings = []
x_text = []
word2idx = {}

if Embedding == "GloVe" or Embedding == "fastText":
    weights = []
    word2idx = {'PAD': 0}

    # LOAD THE DATASET
    if Embedding == "fastText":
        f = io.open(path, 'r', encoding='utf8', newline='\n', errors='ignore')
        n, d = map(int, f.readline().split())  # FastText

    else:
        f = io.open(path, 'r', encoding='utf8', newline='\n', errors='ignore')
    for index, line in enumerate(f):
        # print(line)
        values = line.split()
        i = 0
        word = ""

        while i < len(values):
            try:
                float(values[i])
                break
            except:
                word += values[i]
                i += 1
        word_weights = np.asarray(values[-d:], dtype=np.float32)
        # print(len(word_weights))
        word2idx[word] = index + 1
        weights.append(word_weights)
        if index + 1 == 400:
            break
    embedding_size = len(weights[0])
    weights.insert(0, np.random.randn(embedding_size))
    UNKNOWN_TOKEN = len(weights)
    word2idx['UNK'] = UNKNOWN_TOKEN
    weights.append(np.random.randn(embedding_size))
if Embedding == "Bert":

    weights = []
    tokenizer = bert_tokenization.FullTokenizer(vocab_file="./bert_checkpoint/vocab.txt", do_lower_case=True)
    tokenized = [tokenizer.tokenize(j) for i, j in enumerate(x_text)]
    for i, j in enumerate(tokenized):
        if len(j) > max_seq_length - 2:
            tokenized[i] = tokenized[i][0:(max_seq_length - 2)]
    bert_config = []
    i = -1
    word2idx = {('PAD', 0): -1}
    j = 0
    for line in open('CrisisLexT26_english_output2.txt', 'r'):
        record = line.split()
        if record[0] == "[CLS]":
            i += 1
        word2idx[(record[0], i)] = j
        j += 1
        weights.append(record[1:])

    embedding_size = len(weights[0])
    print(embedding_size)
    weights.insert(0, np.random.randn(embedding_size))
    UNKNOWN_TOKEN = len(weights)
    word2idx[('UNK', 0)] = UNKNOWN_TOKEN
    weights.append(np.random.randn(embedding_size))


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
        self, sequence_length, num_classes, vocab_size,
        embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, binary=True):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.binary = binary
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            global weights
            global word2idx
            global embeddings
            global Embedding


            if Embedding == "Doc2Vec":
                from gensim.test.utils import common_texts
                from gensim.models.doc2vec import Doc2Vec, TaggedDocument

                documents = []
                for i, doc in enumerate(x_text):
                    #print(i, doc.split())
                    documents.append(TaggedDocument(doc.split(), [i]))
                #print(documents)
                #exit()
                model = Doc2Vec(documents, vector_size=d, window=2, min_count=1, workers=4)
                model.train(documents, total_examples=len(documents), epochs=100)
                self.embedded_chars = tf.nn.embedding_lookup(model.wv.syn0, self.input_x)
                # print("Hello", self.embedded_chars.shape)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # exit()
            elif Embedding == "word2vec":
                from gensim.test.utils import common_texts, get_tmpfile
                from gensim.models import Word2Vec
                path = get_tmpfile("word2vec.model")
                #print(common_texts)
                documents = []
                for i, doc in enumerate(x_text):
                    documents.append(doc.split())
                model = Word2Vec(documents, size=d, window=5, min_count=1, workers=4)
                model.save("word2vec.model")

                model.train(documents, total_examples=len(documents), epochs=100)
                self.embedded_chars = tf.nn.embedding_lookup(model.wv.syn0, self.input_x)
                # print("Hello", self.embedded_chars.shape)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # exit()
            elif Embedding == "ELMo":
                elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
                l = max([len(x_text[i]) for i in range(len(x_text))])

                for i in range(len(x_text)):
                    np.pad(x_text[i], (0, l), 'constant')
                    # print(x_text[i])
                self.embedded_chars = elmo(x_text, signature="default", as_dict=True)["elmo"]
                # self.embedded_chars = tf.nn.embedding_lookup(embeddings, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            else:
                self.weights = np.asarray(weights, dtype=np.float32)
                vocab_size = self.weights.shape[0]
                # print(weights.shape)
                # glove_weights_initializer = tf.constant_initializer(self.weights)
                #embedding_weights = tf.constant(weights,name='embedding_weights',shape=(vocab_size, embedding_size))
                print("Bert")
                self.W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), dtype=np.float32,  name="W", trainable=True)
                self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
                max_seq = max([len(i.split()) for i in x_text])

                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                # print("Hello", self.embedded_chars.shape)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                #print(W.shape)
                #print(self.embedded_chars_expanded.shape)
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            with tf.name_scope("precision"):
                self.confusion = tf.confusion_matrix(tf.argmax(self.input_y, 1), self.predictions,
                                                     num_classes=num_classes, name="confusion")
                if self.binary:
                    self.precision = self.confusion[0][0] / (self.confusion[0][0] + self.confusion[0][1])
                else:
                    self.precision = [self.confusion[i][i] / tf.reduce_sum(self.confusion, 1)[i] for i in
                                   range(self.confusion.shape[0])]  # with tf.name_scope("f1"):
            with tf.name_scope("recall"):
                self.confusion = tf.confusion_matrix(tf.argmax(self.input_y, 1), self.predictions,
                                                     num_classes=num_classes,
                                                     name="confusion")
                if self.binary:
                    self.recall = self.confusion[0][0] / (
                            self.confusion[0][0] + self.confusion[1][0])  # with tf.name_scope("f1"):
                else:
                    self.recall = [self.confusion[i][i] / tf.reduce_sum(self.confusion, 0)[i] for i in
                                   range(self.confusion.shape[0])]  # with tf.name_scope("f1"):
            # Confusion
            with tf.name_scope("confusion"):
                self.confusion = tf.confusion_matrix(tf.argmax(self.input_y, 1), self.predictions,
                                                     num_classes=num_classes, name="confusion")

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# import bert_optimization
idx2word={}
word2idx = {}
weights = []


ITALIAN = True
TRAIN = True
# CONFIDENCE_THRESHOLD = 0.9999
# CONFIDENCE_THRESHOLD = 0.99999 Good with 50 epochs
# CONFIDENCE_THRESHOLD = 0.999999 Good with 1 epoch (too few added to labeled?)
# CONFIDENCE_THRESHOLD = 0.999995 Not good with one epoch of training(too many, check classification)
CONFIDENCE_THRESHOLD = 0.999


# Embedding = "Bert"
# Embedding = "ELMo"
# Embedding = "Doc2Vec"
Embedding = "word2vec"
# Embedding = "GloVe"
# Embedding = "fastText"
# import bert

if Embedding == "Bert":
    import bert_modeling
    import bert_optimization
    import bert_tokenization

    max_seq_length = 128
    d = 768
if Embedding == "GloVe":
    d = 300
elif Embedding == "fastText":
    d = 300
elif Embedding == "ELMo":
    # ELMo = "embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
    d = 1024
elif Embedding == "Doc2Vec":
    d = 300
elif Embedding == "word2vec":
    d = 300

embeddings = []
x_text = []
word2idx = {}


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

class TextCNN(object):

    def initialize(self, x_text):
        if Embedding == "Bert":
            import bert_modeling
            import bert_optimization
            import bert_tokenization
            max_seq_length = 128
            d = 768
            if ITALIAN:
                path = "../Data/ItalianData/italian_output2.txt"  ## Related?
            else:
                path = "../Data/EnglishData/CrisisLexT26_english_output2.txt"  ## Related?
            from gensim.models import KeyedVectors
            # model = KeyedVectors.load_word2vec_format(path)
        if Embedding == "GloVe":
            d = 300
            path = "../Data/embeddings/glove.840B.300d.txt"
            from gensim.models import KeyedVectors

            # glove_file = datapath(path)
            # tmp_file = get_tmpfile("test_word2vec.txt")
            # _ = glove2word2vec(glove_file, tmp_file)
            model = KeyedVectors.load_word2vec_format(path)
            print("GloVe model loaded")
        elif Embedding == "fastText":
            d = 300
            if ITALIAN:
                path = "../Data/italian_embeddings/cc.it.300.vec"
            else:
                path = "../Data/embeddings/crawl-300d-2M-subword.vec"
            from gensim.models import KeyedVectors
            model = KeyedVectors.load_word2vec_format(path)
            print("fastText model loaded")
        elif Embedding == "ELMo":
            # ELMo = "../Data/embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
            d = 1024
        elif Embedding == "Doc2Vec":
            d = 300
        elif Embedding == "word2vec":
            d = 300
            from gensim.models import KeyedVectors

        embeddings = []
        word2idx = {}



        self.word2idx = {}
        self.weights = []
        if Embedding == "fastText" or Embedding == "GloVe":
            self.weights = []
            word_set = set()
            index = 0
            for i in x_text:
                for j in i.split():
                    if j not in word_set:
                        if j in model.vocab:
                            word_set.add(j)
                            self.weights.append(model.vectors[model.vocab[j].index])
                            # print(j, model.vocab[j].index)  # Print the correspondences between indices and embedding table

                            self.word2idx[j] = index
                            index += 1
            self.weights.append(np.random.randn(d))
            self.word2idx['UNK'] = index
        if Embedding == "Bert":
            self.weights = []
            if ITALIAN:
                tokenizer = bert_tokenization.FullTokenizer(vocab_file="../Data/bert_checkpoint_multilingual/vocab.txt",
                                                            do_lower_case=True)
            else:
                tokenizer = bert_tokenization.FullTokenizer(vocab_file="../Data/bert_checkpoint/vocab.txt",
                                                            do_lower_case=True)

            tokenized = [tokenizer.tokenize(j) for i, j in enumerate(x_text)]
            for i, j in enumerate(tokenized):
                if len(j) > max_seq_length - 2:
                    tokenized[i] = tokenized[i][0:(max_seq_length - 2)]
            bert_config = []
            i = -1
            self.word2idx = {}
            j = 0

            if ITALIAN:
                file = '../Data/ItalianData/italian_output2_related.txt'
            else:
                file = '../Data/EnglishData/CrisisLexT26_english_output2_related.txt'
            for line in open(file, 'r'):
                record = line.split()
                if record[0] == "[CLS]":
                    i += 1
                word2idx[(record[0], i)] = j
                j += 1
                self.weights.append(record[1:])

            embedding_size = len(self.weights[0])
            print(embedding_size)
            self.weights.insert(0, np.random.randn(embedding_size))
            UNKNOWN_TOKEN = len(self.weights)
            self.word2idx[('UNK', 0)] = UNKNOWN_TOKEN
            self.word2idx['UNK'] = UNKNOWN_TOKEN
            self.weights.append(np.random.randn(embedding_size))

    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(

        self, sequence_length, num_classes, vocab_size,
        embedding_size, filter_sizes, num_filters, text, l2_reg_lambda=0.0, binary=True, italian=False):
        lang = 1 if italian else 0


        # Placeholders for input, output and dropout
        self.n = tf.placeholder(tf.int32, name="n")
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.unlabeled_training = tf.placeholder(tf.int32, [None, sequence_length], name="x_unlabeled")
        self.y_unlabeled = tf.placeholder(tf.float32, [None, num_classes], name="y_unlabeled")
        self.x_text = tf.placeholder(tf.string, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.binary = binary
        self.results = tf.Variable(tf.zeros([1, sequence_length], tf.int32), name="co_training")
        self.results_y = tf.Variable(tf.zeros([1, 2], tf.float32), name="co_training")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        self.initialize(text)



        # Embedding layer
        with tf.name_scope("embedding"):
            global weights
            global embeddings
            global Embedding
            global model

            if Embedding == "Doc2Vec":
                from gensim.models.doc2vec import Doc2Vec, TaggedDocument

                documents = []
                for i, doc in enumerate(text):
                    #print(i, doc.split())
                    documents.append(TaggedDocument(doc.split(), [i]))
                #print(documents)
                #print(documents)
                #exit()
                model = Doc2Vec(documents, vector_size=d, window=2, min_count=1, workers=4)
                model.train(documents, total_examples=len(documents), epochs=100)
                self.embedded_chars = tf.nn.embedding_lookup(model.wv.syn0, self.input_x)
                # print("Hello", self.embedded_chars.shape)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            elif Embedding == "word2vec":
                from gensim.test.utils import get_tmpfile
                from gensim.models import Word2Vec
                path = get_tmpfile("word2vec.model")
                #print(common_texts)
                documents = []
                for i, doc in enumerate(text):
                    documents.append(doc.split())
                model = Word2Vec(documents, size=d, window=5, min_count=1, workers=4)
                model.save("word2vec.model")

                model.train(documents, total_examples=len(documents), epochs=100)
                self.W = model.wv.syn0
                # new = tf.concat([self.results, self.input_x], 0)
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                # print("Hello", self.embedded_chars.shape)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            elif Embedding == "ELMo":
                elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
                print("ELMo module loaded from tensorflow-hub")
                l = max([len(x_text[i]) for i in range(len(x_text))])
                max_document_length = max([len(x.split(" ")) for x in x_text])
                # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
                for i in range(len(x_text)):
                    np.pad(x_text[i], (0, l), 'constant')
                    # print(x_text[i])

                # x = np.array(list(vocab_processor.fit_transform(x_text)))
                global idx2word
                word2idx = {}
                idx2word = {}
                index = 0
                for i in x_text:  # TODO: extract as global for all corpus
                    for j in i.split():
                        if j not in self.word2idx:
                            self.word2idx[j] = index
                            idx2word[index] = j
                            index += 1

                self.x_text = tf.reshape(self.x_text, [-1])
                self.embedded_chars = elmo(self.x_text, signature="default", as_dict=True)["elmo"]
                # self.embedded_chars = tf.nn.embedding_lookup(embeddings, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            elif Embedding == "Bert":
                self.weights = np.array(self.weights)
                self.W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), dtype=np.float32, name="W",
                                     trainable=TRAIN)
                self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
                # max_seq = max([len(i.split()) for i in x_text])
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.x_text)
                # print("Hello", self.embedded_chars.shape)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            else:
                self.weights = np.array(self.weights)
                vocab = set()
                for i in x_text:
                    for j in i.split():
                        vocab.add(j)
                # vocab_size = len(vocab)
                vocab_size = self.weights.shape[0]

                self.W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), dtype=np.float32,  name="W", trainable=TRAIN)
                self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])

                # new = tf.concat([self.results, self.input_x], 0)

                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                # print("Hello", self.embedded_chars.shape)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            self.embedded_chars_unlabeled = tf.nn.embedding_lookup(self.W, self.unlabeled_training)
            self.embedded_chars_expanded_unlabeled = tf.expand_dims(self.embedded_chars_unlabeled, -1)


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        pooled_outputs_unlabeled = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                conv_unlabeled = tf.nn.conv2d(self.embedded_chars_expanded_unlabeled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                h_unalabeled = tf.nn.relu(tf.nn.bias_add(conv_unlabeled, b), name="relu_unlabeled")
                # print(sequence_length - filter_size + 1)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID',name="pool")
                pooled_unlabeled = tf.nn.max_pool(h_unalabeled, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID',name="pool")
                pooled_outputs.append(pooled)
                pooled_outputs_unlabeled.append(pooled_unlabeled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        self.h_pool_unlabeled = tf.concat(pooled_outputs_unlabeled, 3)
        self.h_pool_flat_unlabeled = tf.reshape(self.h_pool_unlabeled, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W",shape=[num_filters_total, num_classes])
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")  # TODO scores give confidence
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

            self.scores_unlabeled = tf.nn.softmax(tf.nn.xw_plus_b(self.h_pool_flat_unlabeled, W, b, name="scores_unlabeled"))
            self.predictions_unlabeled = tf.argmax(self.scores_unlabeled, 1, name="predictions")

            with tf.name_scope("co-training"):
                maxs = tf.reduce_max(self.scores_unlabeled, reduction_indices=[1])
                zeros = tf.cast(tf.zeros_like(maxs), dtype=tf.bool)
                ones = tf.cast(tf.ones_like(maxs), dtype=tf.bool)
                mask = tf.where(maxs > CONFIDENCE_THRESHOLD, ones, zeros)
                # self.predictions_unlabeled = tf.argmax(self.scores_unlabeled, 1, name="predictions_unlabeled")

                self.results = tf.boolean_mask(self.unlabeled_training, mask)

                self.new_y = tf.boolean_mask(self.scores_unlabeled, mask)
                self.predict2 = tf.argmax(self.new_y, 1, name="predictions_unlabeled")
                self.results_y = tf.one_hot(self.predict2, 2, dtype=tf.float32) # predicted labels above average
                self.actual_unlabeled_labels = tf.boolean_mask(self.y_unlabeled, mask)
                # take filtered maxs
                # argmax
                # onehot

                # self.y = tf.concat()
                self.next = tf.boolean_mask(self.unlabeled_training, tf.where(maxs > CONFIDENCE_THRESHOLD, zeros, ones))
                self.next_y = tf.boolean_mask(self.y_unlabeled, tf.where(maxs > CONFIDENCE_THRESHOLD, zeros, ones))
                self.maxs = maxs
        # Calculate mean cross-entropy loss

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                correct_predictions_unlabeled = tf.equal(self.predict2, tf.argmax(self.actual_unlabeled_labels, 1))

                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
                self.accuracy_unlabeled = tf.reduce_mean(tf.cast(correct_predictions_unlabeled, "float"), name="accuracy")
            with tf.name_scope("precision"):
                self.precision = tf.metrics.precision(tf.argmax(self.input_y, 1), self.predictions)
            with tf.name_scope("recall"):
                self.recall = tf.metrics.recall(tf.argmax(self.input_y, 1), self.predictions)
            with tf.name_scope("confusion"):
                self.confusion = tf.confusion_matrix(tf.argmax(self.input_y, 1), self.predictions,
                                                     num_classes=num_classes, name="confusion")
                self.confusion_unlabeled = tf.confusion_matrix(tf.argmax(self.actual_unlabeled_labels, 1), self.predict2,
                                                     num_classes=num_classes, name="confusion")
            with tf.name_scope("f1"):
                self.f1 = tf.contrib.metrics.f1_score(tf.argmax(self.input_y, 1), self.predictions)
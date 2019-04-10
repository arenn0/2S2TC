#! /usr/bin/env python
# import nltk
# nltk.download('stopwords')
# import nltk
# nltk.download('averaged_perceptron_tagger')

import datetime
import os
import time

import numpy as np
import tensorflow as tf

import preprocess

BINARY = True
PRETRAINEDEMBEDDING = True
TRAIN = False
CNN = True
SVM= False
NB = False
# Parameters
# ==================================================

weights = []
start_up = 2
confidence = 0.99

if PRETRAINEDEMBEDDING:
    import main_pre_trained_embeddings
    ITALIAN = main_pre_trained_embeddings.ITALIAN
    from main_pre_trained_embeddings import TextCNN
    print("Pretrained Embedding: {}".format(main_pre_trained_embeddings.Embedding))
    print("Italian: {}".format(ITALIAN))

else:
    import main_with_embeddings
    from main_with_embeddings import TextCNN
    ITALIAN = main_with_embeddings.ITALIAN
    print("No Pretrained Embedding")
    print("Italian: {}".format(ITALIAN))
from tensorflow.contrib import learn
# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
if PRETRAINEDEMBEDDING and main_pre_trained_embeddings.Embedding!="ELMo":
    tf.flags.DEFINE_integer("embedding_dim", main_pre_trained_embeddings.d, "Dimensionality of character embedding (default: 128)")
elif not PRETRAINEDEMBEDDING:
    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
else:
    tf.flags.DEFINE_integer("embedding_dim", 1024, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("cotrain_every", 1, "Cotrain Every (default: 50)")
tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 25, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 250, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def preprocess_():
    # Data Preparation
    # ==================================================
    # Load data
    print("Loading data...")
    tf.reset_default_graph()

    if BINARY:
            preprocess.ITALIAN = False
            x_text, y = preprocess.load_data_and_bin_labels("./CrisisLexT26_preprocessed/")
            preprocess.ITALIAN = True
            x_text_italian, y_italian = preprocess.load_data_and_bin_labels("./italian_preprocessed/")
    else:
            preprocess.ITALIAN = False
            x_text, y = preprocess.load_data_and_labels("./CrisisLexT26_preprocessed/")
            preprocess.ITALIAN = True
            x_text_italian, y_italian = preprocess.load_data_and_labels("./italian_preprocessed/")

    x_english_to_italian = preprocess.load_from_file("../Data/EnglishData/CrisisLexT26_english_to_italian_output_related.txt")
    x_italian_to_english = preprocess.load_from_file("../Data/ItalianData/italian_to_english_output_related.txt")


    max_document_length = max([len(x.split(" ")) for x in x_text])
    max_document_length_italian = max([len(x.split(" ")) for x in x_text_italian])
    print("Max Document length:", max_document_length)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab_processor_italian = learn.preprocessing.VocabularyProcessor(max_document_length_italian)

    if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "ELMo" or main_pre_trained_embeddings.Embedding=="Bert"):
        x = x_text
        x_italian = x_text_italian
    else:
        x = np.array(list(vocab_processor.fit_transform(x_text)))
        x_italian = np.array(list(vocab_processor_italian.fit_transform(x_text_italian)))

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    shuffle_indices_italian = np.random.permutation(np.arange(len(y_italian)))

    if CNN:
        if PRETRAINEDEMBEDDING == True and (main_pre_trained_embeddings.Embedding == "ELMo" or main_pre_trained_embeddings.Embedding == "Bert" or main_pre_trained_embeddings.Embedding == 'GloVe' or main_pre_trained_embeddings.Embedding =="fastText"):
            x_shuffled = x
            y_shuffled = y
            x_shuffled_italian = x_italian
            y_shuffled_italian = y_italian
        else:
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
            x_shuffled_italian = x_italian[shuffle_indices_italian]
            y_shuffled_italian = y_italian[shuffle_indices_italian]

    if SVM or NB:
        x_shuffled = x_text
        y_shuffled = y

        x_shuffled_italian = x_text_italian
        y_shuffled_italian = y_italian

    # Split train/test set

    if PRETRAINEDEMBEDDING and main_pre_trained_embeddings.Embedding == "Bert":
        import bert_tokenization
        if ITALIAN:
            tokenizer = bert_tokenization.FullTokenizer(
                vocab_file="../Data/bert_checkpoint_multilingual/vocab.txt",
                do_lower_case=True)
        else:
            tokenizer = bert_tokenization.FullTokenizer(vocab_file="../Data/bert_checkpoint/vocab.txt",
                                                        do_lower_case=True)

        tokenized = [tokenizer.tokenize(j) for i, j in enumerate(x_shuffled)]
        x_t = []
        index = 0
        for i in tokenized:
            x_t.append([])
            for j in tokenized:
                x_t[-1].append(index)
                index += 1
        x_shuffled = x_t

    # x_ = [[np.array(i)] for i in range(len(x_train))]

    # SPLIT THE DATASET in 1) labeled training 2) unlabeled 3) validation 4) test

    percentage = 0.50
    dev_sample_index = -750
    dev_labeled_index = 3000
    dev_unlabeled_index = 7500

    x_train, x_unlabeled, x_dev = x_shuffled[:int(dev_labeled_index * percentage)], x_shuffled[dev_labeled_index:dev_unlabeled_index + dev_labeled_index], x_shuffled[dev_sample_index:]
    y_train, y_unlabeled, y_dev = y_shuffled[:int(dev_labeled_index * percentage)], y_shuffled[dev_labeled_index:dev_unlabeled_index + dev_labeled_index], y_shuffled[dev_sample_index:]

    italian_dev_sample_index = -250
    italian_dev_labeled_index = 1000
    italian_dev_unlabeled_index = 3000

    x_train_italian, x_unlabeled_italian, x_dev_italian = x_shuffled_italian[:int(italian_dev_labeled_index * percentage)], x_shuffled_italian[italian_dev_labeled_index:italian_dev_unlabeled_index + italian_dev_labeled_index], x_shuffled_italian[italian_dev_sample_index:]
    y_train_italian, y_unlabeled_italian, y_dev_italian = y_shuffled_italian[:int(italian_dev_labeled_index * percentage)], y_shuffled_italian[italian_dev_labeled_index:italian_dev_unlabeled_index + italian_dev_labeled_index], y_shuffled_italian[italian_dev_sample_index:]

    x_english_to_italian = x_english_to_italian[dev_labeled_index:dev_unlabeled_index + dev_labeled_index]
    x_italian_to_english = x_italian_to_english[italian_dev_labeled_index:italian_dev_unlabeled_index + italian_dev_labeled_index]
    del x, y, x_shuffled, y_shuffled, x_shuffled_italian, y_shuffled_italian

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Unlabeled/Dev split: {:d}/{:d}/{:d}".format(len(y_train),len(y_unlabeled), len(y_dev)))
    return x_text, x_train, x_unlabeled, x_dev, vocab_processor, y_train, y_unlabeled, y_dev, x_text_italian, x_train_italian, x_unlabeled_italian, x_dev_italian, vocab_processor_italian, y_train_italian, y_unlabeled_italian, y_dev_italian, x_english_to_italian, x_italian_to_english


def train(x_text, x_train, x_unlabeled, x_dev, vocab_processor, y_train, y_unlabeled, y_dev, x_text_italian, x_train_italian, x_unlabeled_italian, x_dev_italian, vocab_processor_italian, y_train_italian, y_unlabeled_italian, y_dev_italian, x_english_to_italian, x_italian_to_english):
    # Training
    # ==================================================

    if SVM or NB:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import SGDClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import BernoulliNB
        from NLTKPreprocessor import NLTKPreprocessor
        from NLTKPreprocessor import identity

        if SVM:
            text_clf = Pipeline([('vect', NLTKPreprocessor('English')), ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), ('clf-svm', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, random_state=42)), ])
            text_clf_italian = Pipeline([('vect', NLTKPreprocessor('Italian')), ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), ('clf-svm', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, random_state=42)), ])
        elif NB:
            text_clf = Pipeline([('vect', NLTKPreprocessor('English')), ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), ('clf-svm', BernoulliNB()), ])
            text_clf_italian = Pipeline([('vect', NLTKPreprocessor('Italian')), ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), ('clf-svm', BernoulliNB()), ])
        labels = np.argmax(y_train, axis=1) + 1
        labels_italian = np.argmax(y_train_italian, axis=1) + 1

        for i in range(FLAGS.num_epochs):
            _ = text_clf.fit(x_train, np.ravel(labels))
            _ = text_clf_italian.fit(x_train_italian, np.ravel(labels_italian))
            predicted_svm = text_clf.predict(x_dev)
            predicted_svm_italian = text_clf.predict(x_dev_italian)
            # SELF-TRAINING
            predicted_svm_unlabeled = text_clf.predict_proba(x_unlabeled)
            predicted_svm_unlabeled_italian = text_clf.predict_proba(x_unlabeled_italian)
            maxs = np.max(predicted_svm_unlabeled, axis=1)
            maxs_italian = np.max(predicted_svm_unlabeled_italian, axis=1)

            next_italian = []
            next = []
            next_y = []
            next_y_italian = []
            next_unlabeled = []
            next_unlabeled_italian = []

            next_english_to_italian = []
            next_italian_to_english = []

            for i in range(len(maxs)):
                if maxs[i] > confidence:
                    next.append(x_unlabeled[i])
                    next_y.append(np.argmax(predicted_svm_unlabeled[i]) + 1)
                    # CO-TRAINING
                    next_italian.append(x_english_to_italian[i])
                    next_y_italian.append(np.argmax(predicted_svm_unlabeled[i]) + 1)

                else:
                    next_unlabeled.append(x_unlabeled[i])
                    # CO-TRAINING
                    next_english_to_italian.append(x_english_to_italian[i])

            for i in range(len(maxs_italian)):
                if maxs_italian[i] > confidence:
                    next_italian.append(x_unlabeled_italian[i])
                    next_y_italian.append(np.argmax(predicted_svm_unlabeled_italian[i]) + 1)
                    # CO-TRAINING
                    next.append(x_italian_to_english[i])
                    next_y.append(np.argmax(predicted_svm_unlabeled_italian[i]) + 1)
                else:
                    next_unlabeled_italian.append(x_unlabeled_italian[i])
                    # CO-TRAINING
                    next_italian_to_english.append(x_italian_to_english[i])

            x_train += next
            x_train_italian += next_italian
            labels = np.append(labels, next_y)
            labels_italian = np.append(labels_italian, next_y_italian)
            x_unlabeled = next_unlabeled
            x_unlabeled_italian = next_unlabeled_italian
            # CO-TRAINING
            x_italian_to_english = next_italian_to_english
            x_english_to_italian = next_english_to_italian

            # END OF SELF-TRAINING
            labels_dev = np.argmax(y_dev, axis=1) + 1
            labels_dev_italian = np.argmax(y_dev_italian, axis=1) + 1

            matrix = confusion_matrix(labels_dev, predicted_svm)
            matrix_italian = confusion_matrix(labels_dev_italian, predicted_svm_italian)
            # predicted_svm == np.ravel(labels_dev))
            print("English evaluation: ")
            print(matrix)
            print("accuracy {}, precision {}, recall {}, f1_score {}".format(accuracy_score(labels_dev, predicted_svm),
                                                                             precision_score(labels_dev, predicted_svm),
                                                                             recall_score(labels_dev, predicted_svm),
                                                                             f1_score(labels_dev, predicted_svm)))
            print("Italian evaluation: ")
            print(matrix_italian)
            print("accuracy {}, precision {}, recall {}, f1_score {}".format(accuracy_score(labels_dev_italian, predicted_svm_italian),
                                                                     precision_score(labels_dev_italian, predicted_svm_italian),
                                                                     recall_score(labels_dev_italian, predicted_svm_italian),
                                                                     f1_score(labels_dev_italian, predicted_svm_italian)))
            print()

    if CNN:

        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == 'ELMo'):
                import bert_tokenization
                seq = max([len(x.split(" ")) for x in x_train])
                seq2 = max([len(x.split(" ")) for x in x_train_italian])
            elif PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == 'Bert'):
                seq = max([len(x) for x in x_train])
                seq2 = max([len(x) for x in x_train_italian])
            else:
                seq = x_train.shape[1]
                seq2 = x_train_italian.shape[1]

            cnn = TextCNN(
                sequence_length=seq,
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                text=x_text,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                binary=BINARY,
                italian=False)

            cnn_italian = TextCNN(
                sequence_length=seq2,
                num_classes=y_train_italian.shape[1],
                vocab_size=len(vocab_processor_italian.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                text=x_text_italian,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                binary=BINARY,
                italian=True)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            grads_and_vars_italian = optimizer.compute_gradients(cnn_italian.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            train_op_italian = optimizer.apply_gradients(grads_and_vars_italian, global_step=global_step)
            if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "fastText" or main_pre_trained_embeddings.Embedding == "GloVe" or main_pre_trained_embeddings.Embedding == "Bert"):
                embedding_init = cnn.W.assign(cnn.embedding_placeholder)
                embedding_init_italian = cnn_italian.W.assign(cnn_italian.embedding_placeholder)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            precision_summary = tf.summary.scalar("precision", cnn.precision)
            recall_summary = tf.summary.scalar("recall", cnn.recall)

            conf_summary = tf.summary.tensor_summary("confusion", cnn.confusion)

            loss_summary_italian = tf.summary.scalar("loss", cnn_italian.loss)
            acc_summary_italian = tf.summary.scalar("accuracy", cnn_italian.accuracy)
            precision_summary_italian = tf.summary.scalar("precision", cnn_italian.precision)
            recall_summary_italian = tf.summary.scalar("recall", cnn_italian.recall)

            conf_summary_italian = tf.summary.tensor_summary("confusion", cnn_italian.confusion)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged, conf_summary, loss_summary_italian, acc_summary_italian, conf_summary_italian])# , conf_summary
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "fastText" or main_pre_trained_embeddings.Embedding == "GloVe" or main_pre_trained_embeddings.Embedding == "Bert"):
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), embedding_init, embedding_init_italian], feed_dict={cnn.embedding_placeholder: cnn.weights}) #, x_init], feed_dict={cnn.embedding_placeholder: cnn.weights, cnn.x_placeholder: cnn.x})
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), embedding_init, embedding_init_italian], feed_dict={cnn_italian.embedding_placeholder: cnn_italian.weights}) #, x_init], feed_dict={cnn.embedding_placeholder: cnn.weights, cnn.x_placeholder: cnn.x})


            else: #if not PRETRAINEDEMBEDDING:
                sess.run(tf.global_variables_initializer())
            #else:
            #sess.run([tf.global_variables_initializer(), main_pre_trained_embeddings.embeddings], feed_dict={cnn.train_x: cnn.train_x})



            def train_step(cnn, x_batch, y_batch, x_unlabeled, y_unlabeled, x, y):
                """
                A single training step
                """
                # x_batch = tf.cast(x_batch, tf.float32)
                # a = result + x_batch
                a = []
                for i in x:
                    a.append(i)
                for i in x_batch:
                    a.append(i)
                b = []
                for i in y:
                    b.append(i)
                for i in y_batch:
                    b.append(i)

                if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "ELMo" or main_pre_trained_embeddings.Embedding=="Bert"):
                    feed_dict = {
                        cnn.unlabeled_training: x_unlabeled,
                        cnn.unlabeled_labels: y_unlabeled,
                        cnn.n: len(x_batch),
                        cnn.input_x: a,
                        cnn.x_text: x_batch,
                        cnn.input_y: b,
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    }
                else:

                    feed_dict = {
                        cnn.unlabeled_training: x_unlabeled,
                        cnn.y_unlabeled: y_unlabeled,
                        cnn.n: len(x_batch),
                        cnn.input_x: a,
                        cnn.input_y: b,
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    }

                if TRAIN and COTRAIN:
                    _, step, summaries, loss, accuracy, precision, recall, confusion, cross, next, next_y, maxs, scores, labels, y_predictions, result, accuracy_unlabeled, confusion_unlabeled, scores_unlabeled = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall, cnn.confusion, cnn.cross, cnn.next, cnn.next_y, cnn.maxs, cnn.scores_unlabeled, cnn.predictions, cnn.predict, cnn.results, cnn.accuracy_unlabeled, cnn.confusion_unlabeled, cnn.scores_unlabeled],
                        feed_dict)
                    # print(sess.run(cnn.results, {cnn.x_unlabeled: x_unlabeled}))
                    # print(cnn.scores_unlabeled)
                    # print(scores)
                    print("Number of tweets above confidence threshold: ", len(cross))
                    print(confusion_unlabeled)
                    result = result[1:]
                else:
                    y_predictions = []
                    result = []
                    next = x_unlabeled
                    next_y = y_unlabeled
                    _, step, summaries, loss, accuracy, precision, recall, confusion = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall, cnn.confusion],feed_dict)
                    #                 print(confusion)


                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, accuracy {:g}, precision {}, recall {}".format(time_str, step, loss, accuracy, precision, recall))
                train_summary_writer.add_summary(summaries, step)
                b=[]
                for i in y:
                    b.append(i)
                for i in y_predictions:
                    b.append(i)
                a=[]
                for i in x:
                    a.append(i)
                for i in result:
                    a.append(i)

                return next, next_y, a, b

            def dev_step(cnn, x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "ELMo" or main_pre_trained_embeddings.Embedding=="Bert"):
                    feed_dict = {
                        cnn.unlabeled_training: x_unlabeled,
                        cnn.n: len(x_batch),
                        cnn.x_text: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0,
                    }
                else:
                    feed_dict = {
                        cnn.unlabeled_training: x_unlabeled,
                        cnn.n: len(x_batch),
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0,
                    }
                step, summaries, loss, accuracy, precision, recall, confusion = sess.run( #, confusion
                    [global_step, dev_summary_op, cnn.loss_dev, cnn.accuracy_dev, cnn.precision_dev, cnn.recall_dev, cnn.confusion_dev],#, cnn.confusion],
                    feed_dict)
                print(confusion)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, accuracy {:g}, precision {}, recall {}".format(time_str, step, loss, accuracy, precision, recall))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches

            if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding=="ELMo"):
                x_ = [[np.array(i)] for i in range(len(x_train))]
                batches = preprocess.batch_iter(list(zip(x_, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                # TODO ITALIAN PART
            else:
                batches = preprocess.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                batches_italian = preprocess.batch_iter(list(zip(x_train_italian, y_train_italian)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...

            unlabeled_training = []
            for i in x_unlabeled:
                unlabeled_training.append([])
                for j in i:
                    unlabeled_training[-1].append(j)

            unlabeled_training_italian = []
            for i in x_unlabeled_italian:
                unlabeled_training_italian.append([])
                for j in i:
                    unlabeled_training_italian[-1].append(j)


            result = ()
            result_italian = ()
            y_prediction = ()
            y_prediction_italian = ()

            for batch, batch_italian in zip(batches, batches_italian):
                x_batch, y_batch = zip(*batch)
                x_batch_italian, y_batch_italian = zip(*batch_italian)

                current_step = tf.train.global_step(sess, global_step) + 1
                if current_step % FLAGS.cotrain_every == 0 and current_step > start_up:
                    COTRAIN = True
                else:
                    COTRAIN = False

                if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "GloVe" or main_pre_trained_embeddings.Embedding == "fastText"):
                    max_document_length = max([len(x.split(" ")) for x in x_text])
                    max_document_length_italian = max([len(x.split(" ")) for x in x_text_italian])
                    x = []
                    for counter, j in enumerate(x_batch):
                        x.append([])
                        x[counter] = []
                        for i in j.split():
                            index = cnn.word2idx.get(i, len(main_pre_trained_embeddings.word2idx) - 1)
                            # print(i,index)
                            x[counter].append(index)
                        while len(x[counter]) < max_document_length:
                            x[counter].append(len(main_pre_trained_embeddings.word2idx) - 1)

                    x = np.array(x)

                    x_italian = []
                    for counter, j in enumerate(x_text_italian):
                        x_italian.append([])
                        x_italian[counter] = []
                        for i in j.split():
                            index = cnn_italian.word2idx.get(i, len(main_pre_trained_embeddings.word2idx) - 1)
                            # print(i,index)
                            x_italian[counter].append(index)
                        while len(x_italian[counter]) < max_document_length_italian:
                            x_italian[counter].append(len(main_pre_trained_embeddings.word2idx) - 1)

                    x_batch = np.array(x)
                    x_batch_italian = np.array(x_italian)

                if PRETRAINEDEMBEDDING and main_pre_trained_embeddings.Embedding == "ELMo":
                    x_t = []
                    for i in x_batch:
                        x_t.append(x_train[i[0]])
                    for i in range(len(x_t)):
                        while len(x_t[i].split()) < seq:
                            x_t[i] += ' unk '

                    #  TODO: Italian Part

                    unlabeled_training, y_unlabeled, result, y_prediction = train_step(cnn, x_t, y_batch, unlabeled_training, y_unlabeled, result, y_prediction)
                    unlabeled_training, y_unlabeled, result, y_prediction = train_step(cnn_italian, x_t, y_batch, unlabeled_training, y_unlabeled, result, y_prediction)
                else:
                    unlabeled_training, y_unlabeled, result, y_prediction = train_step(cnn, x_batch, y_batch, unlabeled_training, y_unlabeled, result, y_prediction)
                    unlabeled_training_italian, y_unlabeled_italian, result_italian, y_prediction_italian = train_step(cnn_italian, x_batch_italian, y_batch_italian, unlabeled_training_italian, y_unlabeled_italian, result_italian, y_prediction_italian)

                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")

                    if PRETRAINEDEMBEDDING and main_pre_trained_embeddings.Embedding == "ELMo":
                        for i in range(len(x_dev)):
                            while len(x_dev[i].split()) < seq:
                                x_dev[i] += ' unk '
                    dev_step(cnn, x_dev, y_dev, writer=dev_summary_writer)
                    dev_step(cnn_italian, x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if current_step == FLAGS.num_epochs:
                    break

def main(argv=None):
    x_text, x_train, x_unlabeled, x_dev, vocab_processor, y_train, y_unlabeled, y_dev, x_text_italian, x_train_italian, x_unlabeled_italian, x_dev_italian, vocab_processor_italian, y_train_italian, y_unlabeled_italian, y_dev_italian, x_english_to_italian, x_italian_to_english = preprocess_()
    train(x_text, x_train, x_unlabeled, x_dev, vocab_processor, y_train, y_unlabeled, y_dev, x_text_italian, x_train_italian, x_unlabeled_italian, x_dev_italian, vocab_processor_italian, y_train_italian, y_unlabeled_italian, y_dev_italian, x_english_to_italian, x_italian_to_english)

if __name__ == '__main__':
    tf.app.run()

import random
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp

# SRAND

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
# ctx = mx.gpu(0)

from bert import *

bert_base, vocabulary = nlp.model.get_model('bert_24_1024_16', dataset_name='book_corpus_wiki_en_uncased', pretrained=True, use_pooler=True, use_decoder=False, use_classifier=False)
print(bert_base)

model = bert.BERTClassifier(bert_base, num_classes=2, dropout=0.1)
model.classifier.initialize(init=mx.init.Normal(0.02))
model.hybridize(static_alloc=True)

loss_function = gluon.loss.SoftmaxCELoss()
loss_function.hybridize(static_alloc=True)

metric = mx.metric.Accuracy()
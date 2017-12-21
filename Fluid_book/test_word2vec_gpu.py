from __future__ import print_function
import numpy as np
import math
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
from paddle.v2.fluid.optimizer import Adagrad
from paddle.v2.fluid.initializer import NormalInitializer
from paddle.v2.fluid.param_attr import ParamAttr
from paddle.v2.fluid.regularizer import L2Decay
import sys
import time

PASS_NUM = 5
EMBED_SIZE = 32
HIDDEN_SIZE = 256
N = 5
BATCH_SIZE = 32
IS_SPARSE = False
#IS_SPARSE = True
SEED=1

word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)


first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
forth_word = fluid.layers.data(name='forthw', shape=[1], dtype='int64')
next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')

global_l2_decay = L2Decay(8e-4)

emb_param_attr = ParamAttr(name="shared_w",
                    initializer=NormalInitializer(loc=0.0, scale=0.001, seed=SEED),
                    regularizer=global_l2_decay,
                    learning_rate=1)

embed_first = fluid.layers.embedding(
            input=first_word,
                size=[dict_size, EMBED_SIZE],
                    dtype='float32',
                        is_sparse=IS_SPARSE,
                            param_attr=emb_param_attr)
embed_second = fluid.layers.embedding(
            input=second_word,
                size=[dict_size, EMBED_SIZE],
                    dtype='float32',
                        is_sparse=IS_SPARSE,
                            param_attr=emb_param_attr)
embed_third = fluid.layers.embedding(
            input=third_word,
                size=[dict_size, EMBED_SIZE],
                    dtype='float32',
                        is_sparse=IS_SPARSE,
                            param_attr=emb_param_attr)
embed_forth = fluid.layers.embedding(
            input=forth_word,
                size=[dict_size, EMBED_SIZE],
                    dtype='float32',
                        is_sparse=IS_SPARSE,
                            param_attr=emb_param_attr)


concat_embed = fluid.layers.concat(
            input=[embed_first, embed_second, embed_third, embed_forth], axis=1)

hidden1_param_attr = ParamAttr(name="hidden1_w",
                                   initializer=NormalInitializer(scale=1. / math.sqrt(EMBED_SIZE * 8),
                                                seed=SEED),
                                    regularizer=global_l2_decay,
                                    learning_rate=1)

hidden1_bias_attr = ParamAttr(name="hidden1_b",
                                   initializer=NormalInitializer(scale=1., seed=SEED),
                                                              learning_rate=2)
hidden1 = fluid.layers.fc(input=concat_embed,
                         size=HIDDEN_SIZE,
                         act='sigmoid',
                         param_attr=hidden1_param_attr,
                         bias_attr=hidden1_bias_attr)

hidden1_drop = fluid.layers.dropout(x=hidden1, dropout_prob=0.5, seed = SEED)

predict_param_attr = ParamAttr(name="predict_w",
                    initializer=NormalInitializer(
                    loc=0.0,
                    scale=1. / math.sqrt(HIDDEN_SIZE),
                    seed=SEED),
                    regularizer=global_l2_decay)

predict_bias_attr = ParamAttr(name="predict_b",
                    initializer=NormalInitializer(
                    loc=0.0,
                    scale=1.,
                    seed=SEED),
                    learning_rate=2)

predict_word = fluid.layers.fc(input=hidden1_drop,
        size=dict_size,
        act='softmax',
        param_attr=predict_param_attr,
        bias_attr=predict_bias_attr)

cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
avg_cost = fluid.layers.mean(x=cost)
avg_cost = fluid.layers.scale(x=avg_cost, scale=float(BATCH_SIZE))
optimizer = Adagrad(learning_rate=3e-3)
optimizer.minimize(avg_cost)

accuracy = fluid.evaluator.Accuracy(input=predict_word, label=next_word)

inference_program = fluid.default_main_program().clone()

test_accuracy = fluid.evaluator.Accuracy(
        input=predict_word, label=next_word, main_program=inference_program)
test_target = [avg_cost] + test_accuracy.metrics + test_accuracy.states
inference_program = fluid.io.get_inference_program(
        test_target, main_program=inference_program)
train_reader = paddle.batch(
        paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)
test_reader = paddle.batch(paddle.dataset.imikolov.test(word_dict, N), BATCH_SIZE)

place = fluid.GPUPlace(0)
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(
        feed_list=[first_word, second_word, third_word, forth_word, next_word],
        place=place)
exe.run(fluid.default_startup_program())

for pass_id in range(PASS_NUM):
    batch_id = 0
    accuracy.reset(exe)
    print("begin")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    time_begin = time.clock()
    for data in train_reader():
        loss, acc = exe.run(fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost] + accuracy.metrics)
        pass_acc = accuracy.eval(exe)
        batch_id += 1
        if batch_id % 100 == 0:
            print("pass_id=" + str(pass_id) + " batch_id=" + str(batch_id)
                    + " cost=" +str(loss[0])+ " avg_cost=" +str(loss[0]/BATCH_SIZE)
                    + " acc=" + str(acc) + " pass_acc=" + str(pass_acc))
            sys.stdout.flush()
    time_end = time.clock()
    print("Time cost=%f"%(time_end-time_begin))
    test_accuracy.reset(exe)
    for data in test_reader():
        out, acc = exe.run(inference_program,
                    feed=feeder.feed(data),
                    fetch_list=[avg_cost] + test_accuracy.metrics)
    test_pass_acc = test_accuracy.eval(exe)
    print("pass_id=" + str(pass_id)
                    + " test_acc=" + str(test_pass_acc))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("end")

import math
import os
import time

import numpy
import paddle.v2 as paddle

with_gpu = os.getenv('WITH_GPU', '0') != '0'

embsize = 32
hiddensize = 256
N = 5
pass_acc = 0.
time_begin = None
time_end = None

def wordemb(inlayer):
    wordemb = paddle.layer.table_projection(
            input=inlayer,
            size=embsize,
            param_attr=paddle.attr.Param(
                name="_proj", initial_std=0.001, learning_rate=1, l2_rate=0))
    return wordemb

# save and load word dict and embedding table
def save_dict_and_embedding(word_dict, embeddings):
    with open("word_dict", "w") as f:
        for key in word_dict:
            f.write(key + " " + str(word_dict[key]) + "\n")
    with open("embedding_table", "w") as f:
        numpy.savetxt(f, embeddings, delimiter=',', newline='\n')

def load_dict_and_embedding():
    word_dict = dict()
    with open("word_dict", "r") as f:
        for line in f:
            key, value = line.strip().split(" ")
            word_dict[key] = int(value)

    embeddings = numpy.loadtxt("embedding_table", delimiter=",")
    return word_dict, embeddings

def main():
    paddle.init(use_gpu=False, trainer_count=1)
    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)
    # Every layer takes integer value of range [0, dict_size)
    firstword = paddle.layer.data(
             name="firstw", type=paddle.data_type.integer_value(dict_size))
    secondword = paddle.layer.data(
             name="secondw", type=paddle.data_type.integer_value(dict_size))
    thirdword = paddle.layer.data(
             name="thirdw", type=paddle.data_type.integer_value(dict_size))
    fourthword = paddle.layer.data(
             name="fourthw", type=paddle.data_type.integer_value(dict_size))
    nextword = paddle.layer.data(
            name="fifthw", type=paddle.data_type.integer_value(dict_size))

    Efirst = wordemb(firstword)
    Esecond = wordemb(secondword)
    Ethird = wordemb(thirdword)
    Efourth = wordemb(fourthword)

    contextemb = paddle.layer.concat(input=[Efirst, Esecond, Ethird, Efourth])
    hidden1 = paddle.layer.fc(
            input=contextemb,
            size=hiddensize,
            act=paddle.activation.Sigmoid(),
            layer_attr=paddle.attr.Extra(drop_rate=0.5),
            bias_attr=paddle.attr.Param(learning_rate=2),
            param_attr=paddle.attr.Param(
                initial_std=1. / math.sqrt(embsize * 8), learning_rate=1))

    predictword = paddle.layer.fc(
            input=hidden1,
            size=dict_size,
            bias_attr=paddle.attr.Param(learning_rate=2),
            act=paddle.activation.Softmax())

    cost = paddle.layer.classification_cost(input=predictword, label=nextword)

    parameters = paddle.parameters.create(cost)
    adagrad = paddle.optimizer.AdaGrad(
            learning_rate=3e-3,
            regularization=paddle.optimizer.L2Regularization(8e-4))
    trainer = paddle.trainer.SGD(cost, parameters, adagrad)

    def event_handler(event):
        global pass_acc
        global time_begin
        global time_end
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id == 0:
                print("begin")
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                time_begin = time.clock()
            if event.batch_id % 1 == 0:
                pass_acc += float(1-event.metrics["classification_error_evaluator"])
            if event.batch_id % 100 == 0 and event.batch_id != 0:
                print "Pass=%d, Batch=%d, Cost=%f, Acc=%f, Pass_acc=%f" % (
                        event.pass_id, event.batch_id, event.cost,
                        float(1-event.metrics["classification_error_evaluator"]),
                        pass_acc/(event.batch_id+1))

        if isinstance(event, paddle.event.EndPass):
            time_end = time.clock()
            print("Time cost=%f"%(time_end-time_begin))
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print("end")
            pass_acc = 0.
            result = trainer.test(
                    paddle.batch(paddle.dataset.imikolov.test(word_dict, N), 32))
            print "Pass=%d, Test Acc=%s" % (event.pass_id,
                    float(1-result.metrics["classification_error_evaluator"]))
    train_reader = paddle.batch(paddle.dataset.imikolov.train(word_dict, N), 32)

    trainer.train(
            paddle.batch(paddle.dataset.imikolov.train(word_dict, N), 32),
            num_passes=5,
            event_handler=event_handler)

    embeddings = parameters.get("_proj").reshape(len(word_dict), embsize)
    save_dict_and_embedding(word_dict, embeddings)

if __name__ == '__main__':
    main()

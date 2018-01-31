import paddle.v2 as paddle
import time
import sys

def convolution_net(dict_dim,
                    class_dim=2,
                    emb_dim=128,
                    hid_dim=128,
                    fc_dim=96,
                    is_infer=False):
    """
    cnn network definition
    :param dict_dim: size of word dictionary
    :type input_dim: int
    :params class_dim: number of instance class
    :type class_dim: int
    :params emb_dim: embedding vector dimension
    :type emb_dim: int
    :params hid_dim: number of same size convolution kernels
    :type hid_dim: int
    """

    # input layers
    data = paddle.layer.data("word",
                             paddle.data_type.integer_value_sequence(dict_dim))
    lbl = paddle.layer.data("label", paddle.data_type.integer_value(class_dim))

    # embedding layer
    emb = paddle.layer.embedding(input=data, size=emb_dim)

    # convolution layers with max pooling
    conv_3 = paddle.networks.sequence_conv_pool(
        input=emb, context_len=3, hidden_size=hid_dim)
    
    fc1 = paddle.layer.fc(input=conv_3,
                           size=fc_dim)
    # fc and output layer
    prob = paddle.layer.fc(input=fc1,
                           size=class_dim,
                           act=paddle.activation.Softmax())

    if is_infer:
        return prob
    else:
        cost = paddle.layer.classification_cost(input=prob, label=lbl)

        return cost, prob, lbl


def load_vocab(filename):
    vocab = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    return vocab
# load word dict with paddle inner function
word_dict = load_vocab(sys.argv[1])
word_dict["<unk>"] = len(word_dict)
print("dict_size : ",len(word_dict))
sys.stdout.flush()
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.imdb.train(word_dict), buf_size=51200),
    batch_size=4)
test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.imdb.test(word_dict), buf_size=51200),
    batch_size=4)

class_num = 2

num_passes = 30

dict_dim = len(word_dict)

paddle.init(use_gpu=False, trainer_count=1)

cost, prob, label = convolution_net(dict_dim, class_num)

parameters = paddle.parameters.create(cost)

# create optimizer
moment_optimizer = paddle.optimizer.Momentum(
    learning_rate=0.01)

# create trainer
trainer = paddle.trainer.SGD(
    cost=cost,
    parameters=parameters,
    update_equation=moment_optimizer)

# begin training network
feeding = {"word": 0, "label": 1}

begin_time = None
end_time = None
total_time = 0.

def _event_handler(event):
    global begin_time
    global end_time
    global total_time
    """
    Define end batch and end pass event handler
    """
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id == 0:
            begin_time = time.time()
            print('%d pass_begin:'% event.pass_id)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if event.batch_id % 1000 == 0:
            print("Pass %d, Batch %d, Cost %f, %s\n" % (
                event.pass_id, event.batch_id, event.cost, event.metrics))
        sys.stdout.flush()

    if isinstance(event, paddle.event.EndPass):
        print('%d pass_end:' % event.pass_id)
        end_time = time.time()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        total_time += (end_time-begin_time)
        print("pass_id: %d, time_cost: %f" % (event.pass_id, (end_time-begin_time)))
        sys.stdout.flush()
        print("total_time %f"% total_time)

        if test_reader is not None:
            result = trainer.test(reader=test_reader, feeding=feeding)
            print("Test at Pass %d, %s \n" % (event.pass_id, result.metrics))

trainer.train(
    reader=train_reader,
    event_handler=_event_handler,
    feeding=feeding,
    num_passes=num_passes)


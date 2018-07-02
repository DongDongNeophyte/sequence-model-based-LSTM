import tensorflow as tf
import numpy as np
from  sklearn import metrics

class Model(object):
    """docstring for Model."""
    def __init__(self,n_steps, n_inputs, n_class, cells_sizes, dropou, dropout_keep_prob, full_number,full_number2):
        self.input_x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, n_class], name="input_y")
        self.sizes = tf.placeholder(tf.int32, shape=(), name="sizes")
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name="seq_length")

        #[mea, var] = tf.nn.moments(self.input_x, axes=0)#
        #_input_x = tf.nn.batch_normalization(self.input_x, mea, var, 0, 0.1,0.0001, name="input")
        _input_x = self.input_x
        _state = 0
        for i, unit_num in enumerate(cells_sizes):
            with tf.name_scope("rnn-%s" % i):
                cell_fw = tf.nn.rnn_cell.LSTMCell(unit_num)
                cell_bw = tf.nn.rnn_cell.LSTMCell(unit_num)
                cell_fw_dropout = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropou)
                cell_bw_dropout = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=dropou)
                initial_state_fw = cell_fw_dropout.zero_state(self.sizes, dtype=tf.float32)
                initial_state_bw = cell_bw_dropout.zero_state(self.sizes, dtype=tf.float32)
                output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw_dropout, cell_bw_dropout, _input_x, sequence_length=self.seq_length, initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw, scope="rnn-"+str(i))
                _input_x = tf.concat(output, 2)
                _state = state
        #rnn_out =_state
        state = tf.concat(_state, 2)
        rnn_out = state[1]
        #rnn_out =  _input_x 
        [mea, var] = tf.nn.moments(rnn_out, axes=0)
        self.rnn_out1 = tf.nn.batch_normalization(rnn_out, mea, var, 0, 0.1,0.0001, name="rnn_out")
        #self.rnn_out1=rnn_out;
        rnn_out_shape = cells_sizes[-1] * 2
 
        with tf.name_scope("full_connect1"):
            W = tf.Variable(tf.random_uniform([rnn_out_shape, full_number]), [-1.0, 1.0], name="W")
            b = tf.Variable(tf.random_uniform([full_number]), [-1.0, 1.0], name="b")
            #W = tf.Variable(tf.truncated_normal([rnn_out_shape, full_number]), name="W")
            #W = tf.Variable(tf.zeros([rnn_out_shape, full_number]),  name="W")
            #b = tf.Variable(tf.constant(0.0,shape=[full_number]),name="b")
            #W1 = tf.get_variable("W1",shape=[rnn_out_shape, full_number],initializer=tf.contrib.layers.xavier_initializer())
            #b = tf.Variable(tf.constant(0.1,shape=[full_number]),name="b")
            full_out1 = tf.tanh(tf.nn.xw_plus_b(self.rnn_out1, W, b), name="full_out")
            #full_out1 =tf.nn.relu(tf.nn.xw_plus_b(self.rnn_out1, W, b), name="full_out")
            #full_out1 = tf.nn.sigmoid(tf.nn.xw_plus_b(self.rnn_out1, W, b), name="full_out")
            [mea, var] = tf.nn.moments(full_out1, axes=0)
            self.full_out = tf.nn.batch_normalization(full_out1, mea, var, 0, 0.1,0.0001, name="full_out")

        with tf.name_scope("full_connect2"):
            W = tf.Variable(tf.random_uniform([full_number, full_number2]), [-1.0, 1.0], name="W")
            b = tf.Variable(tf.random_uniform([full_number2]), [-1.0, 1.0], name="b")
            full_out22 = tf.tanh(tf.nn.xw_plus_b(self.full_out, W, b), name="full_out")
            #full_out22 = tf.nn.relu(tf.nn.xw_plus_b(self.full_out, W, b), name="full_out") 
            [mea, var] = tf.nn.moments(full_out22, axes=0)
            self.full_out2 = tf.nn.batch_normalization(full_out22, mea, var, 0, 0.1,0.0001, name="full_out")
        
        #with tf.name_scope("dropout"):
            #self.rnn_drop = tf.nn.dropout(self.full_out, keep_prob=dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.Variable(tf.random_uniform([full_number2, n_class]), [-1.0, 1.0], name="W")
            b = tf.Variable(tf.random_uniform([n_class]), [-1.0, 1.0], name="b")
            #W = tf.Variable(tf.truncated_normal([full_number, n_class]), name="W")
            #W = tf.Variable(tf.zeros([full_number, n_class]), name="W")
            #b = tf.Variable(tf.constant(0.0,shape=[n_class]),name="b")
            #W2 = tf.get_variable("W2",shape=[full_number, n_class],initializer=tf.contrib.layers.xavier_initializer())
            #b = tf.Variable(tf.constant(0.1,shape=[n_class]),name="b")
            self.scores =tf.nn.xw_plus_b(self.full_out2, W, b, name="scores")
            #self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.predicted_y=tf.nn.softmax(self.scores,name="predicted_y")


        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            #index=10*tf.argmax(self.input_y,1)
            #if index!=0:
            #if tf.argmax(self.input_y,1)==1:
               #losses=losses*10
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.predicted_y, 1),tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
           
        #with tf.name_scope("AUC"):
            #a = tf.cast(tf.argmax(self.scores, 1),tf.float32)
            #b = tf.cast(tf.argmax(self.input_y,1),tf.float32)
            #self.auc = tf.contrib.metrics.streaming_auc(a, b)
            #self.prediced_y=tf.nn.softmax(self.scores)
            #self.auc=metrics.roc_auc_score(self.input_y,self.prediced_y)

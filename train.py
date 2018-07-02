import os
import numpy as np
import math
import tensorflow as tf
import data_helper
import time
import datetime
from Model import Model
from sklearn import metrics

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

# paramter
tf.flags.DEFINE_integer("training_iters",598, "training iters")
tf.flags.DEFINE_integer("n_class", 2, "n_class")
tf.flags.DEFINE_integer("batch_size",1024, "batch size")
tf.flags.DEFINE_integer("n_inputs",80, "n_inputs")
tf.flags.DEFINE_integer("n_steps",50, "n_steps")
tf.flags.DEFINE_integer("full_number",32, "full connection layer")
tf.flags.DEFINE_integer("full_number2",8, "full connection layer2")
tf.flags.DEFINE_float("dropout1", 0.9, "Dropout keep probability")
tf.flags.DEFINE_float("dropout2", 0.9, "Droppout layer")
tf.flags.DEFINE_float("l2_reg", 0.0, "l2 regularization lambda")
tf.flags.DEFINE_integer("decay_steps",100, "decay steps")
tf.flags.DEFINE_integer("decay_steps1",100, "decay steps")
tf.flags.DEFINE_float("decay_rate", 0.1, "decay rate")
tf.flags.DEFINE_float("lr", 0.001, "learning rate1")
tf.flags.DEFINE_string("cells_sizes", '80,64', "numbers of cells of each layer")
tf.flags.DEFINE_integer("display_step", 50, "display_step")
tf.flags.DEFINE_integer("save_step", 100, "save_step")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
for attr,value in FLAGS.__flags.items():
    print("attr:%s\tvalue:%s" % (attr,str(value)))
# load data
print ("Loading data")
starttime = datetime.datetime.now()
sequences = data_helper.read_data(os.path.abspath(os.path.join(os.path.curdir, "data")), FLAGS.n_inputs)
endtime = datetime.datetime.now()
print (str((endtime-starttime).seconds), "seconds")
validation_sequence = sequences.validation.sequence.reshape((-1, FLAGS.n_steps, FLAGS.n_inputs))
test_sequence = sequences.test.sequence.reshape((-1, FLAGS.n_steps, FLAGS.n_inputs))
validation_iter = math.ceil(validation_sequence.shape[0]/FLAGS.batch_size)
print("validation_iter:",validation_iter )
test_iter = math.ceil(test_sequence.shape[0]/FLAGS.batch_size)

boundaries=[50,100,150,200,300,400,500]
learning_rates=[0.001,0.0001,0.00001,0.000001,0.0000001]
#learning_rates2=[0.0001,0.00008,0.00001,0.000008,0.000001]

# train
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        model = Model(n_steps=FLAGS.n_steps, n_inputs=FLAGS.n_inputs, n_class=FLAGS.n_class, cells_sizes=list(map(int,FLAGS.cells_sizes.split(","))), dropou=FLAGS.dropout1, dropout_keep_prob=FLAGS.dropout2, full_number=FLAGS.full_number, full_number2=FLAGS.full_number2)

        global_step = tf.Variable(0, trainable=False)
        #learning_rate1 = tf.train.exponential_decay(FLAGS.lr, global_step=global_step, decay_steps=FLAGS.decay_steps1, decay_rate=FLAGS.decay_rate)
        learning_rate = tf.train.exponential_decay(FLAGS.lr, global_step=global_step, decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate)
        #learning_rate1=tf.train.piecewise_constant(global_step,boundaries,learning_rates1)
        #learning_rate2=tf.train.piecewise_constant(global_step,boundaries,learning_rates2)
        #var1 = tf.trainable_variables()[0:8]
        #var2 = tf.trainable_variables()[8:]
        optimizer = tf.train.AdamOptimizer(learning_rate)
        #optimizer2 = tf.train.AdamOptimizer(learning_rate2)
        #train_op1 = optimizer1.minimize(model.loss, global_step=global_step,var_list=var1) 
        #train_op2 = optimizer2.minimize(model.loss, global_step=global_step,var_list=var2)
        #train_op = tf.group(train_op1, train_op2)
        ##optimizer=tf.train.AdadeltaOptimizer(rho=0.95,epsilon=1e-07)
        ##optimizer =tf.train.GradientDescentOptimizer(learning_rate)
        #optimizer = tf.train.MomentumOptimizer(learning_rate,0.8)
        train_op = optimizer.minimize(model.loss, global_step=global_step)
    
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
        print ("output dir:", out_dir)
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        print("checkdir:",checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)
        
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        print("train_summary_writer")
        # validation summaries
        validation_summary_op = tf.summary.merge([loss_summary, acc_summary])
        validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
        validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)
        print("validation_summary_writer")        
        print("before global initialization")
        sess.run(tf.global_variables_initializer())
        #sess.run(tf.initialize_local_variables())
        print("glibal initialization")
        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
          print("Variable: ",k)
          print("Shape: ",v.shape)
          print(v)
        saver = tf.train.Saver()
        current_step = tf.train.global_step(sess, global_step)
        starttime = datetime.datetime.now()
        print ("train")
        max_acc = 0.8
        while current_step < FLAGS.training_iters:
            batch_x, batch_l, batch_la, batch_y = sequences.train.next_batch(os.path.abspath(os.path.join(os.path.curdir, "data")),current_step,FLAGS.n_inputs)
            batch_x = batch_x.reshape((FLAGS.batch_size, FLAGS.n_steps, FLAGS.n_inputs))
            sess.run(train_op, feed_dict={model.input_x : batch_x, model.input_y : batch_y, model.sizes : FLAGS.batch_size, model.seq_length : batch_l})
            loss = sess.run(model.loss, feed_dict={model.input_x : batch_x, model.input_y : batch_y, model.sizes : FLAGS.batch_size, model.seq_length : batch_l})
            acc = sess.run(model.accuracy, feed_dict={model.input_x : batch_x, model.input_y : batch_y, model.sizes : FLAGS.batch_size, model.seq_length : batch_l})
            predicted_y=sess.run(model.predicted_y, feed_dict={model.input_x : batch_x, model.input_y : batch_y, model.sizes : FLAGS.batch_size, model.seq_length : batch_l})
            fpr, tpr, thresholds = metrics.roc_curve(batch_la.ravel(),predicted_y.ravel())
            auc_a=metrics.auc(fpr, tpr) 
            auc=metrics.roc_auc_score(batch_la,predicted_y, average='micro')
            b_f1 = metrics.f1_score(tf.argmax(batch_la,1),tf.argmax(predicted_y,1), average='binary')
            macro_f1 = metrics.f1_score(batch_la,predicted_y, average='macro')
            micro_f1 = metrics.f1_score(batch_la,predicted_y, average='micro')
            summaries = sess.run(train_summary_op, feed_dict={model.input_x : batch_x, model.input_y : batch_y, model.sizes : FLAGS.batch_size, model.seq_length : batch_l})
            train_summary_writer.add_summary(summaries, current_step)
            current_step = tf.train.global_step(sess, global_step)
            print(str(current_step) + " Loss= " + "{:.6f}".format(loss) + " acc: " + "{:.6f}".format(acc)+"AUC=")
            print(auc)
            print("AUC_other=")
            print(auc_a)
            print("macro_f1=")
            print(macro_f1)
            print("micro_f1=")
            print(micro_f1)
            if current_step % FLAGS.display_step == 0:
                print(str(current_step),)
                acc_validation = 0.
                for validation_ite in range(validation_iter):
                    validation_batch_x, validation_batch_l, validation_batch_la,validation_batch_y = sequences.validation.Next_batch(FLAGS.batch_size)
                    validation_batch_x = validation_batch_x.reshape((FLAGS.batch_size, FLAGS.n_steps, FLAGS.n_inputs))
                    acc = sess.run(model.accuracy, feed_dict={model.input_x : validation_batch_x, model.input_y : validation_batch_y, model.sizes : FLAGS.batch_size, model.seq_length : validation_batch_l})
                    loss = sess.run(model.loss, feed_dict={model.input_x : validation_batch_x, model.input_y : validation_batch_y, model.sizes : FLAGS.batch_size, model.seq_length : validation_batch_l})
                    predicted_y=sess.run(model.predicted_y, feed_dict={model.input_x : validation_batch_x, model.input_y : validation_batch_y, model.sizes : FLAGS.batch_size, model.seq_length : validation_batch_l})
                    auc=metrics.roc_auc_score(validation_batch_la,predicted_y, average='micro')
                    fpr, tpr, thresholds = metrics.roc_curve(validation_batch_la.ravel(),predicted_y.ravel())
                    auc_a=metrics.auc(fpr, tpr)
                    macro_f1 = metrics.f1_score(validation_batch_la,predicted_y, average='macro')
                    micro_f1 = metrics.f1_score(validation_batch_la,predicted_y, average='micro')
                    acc_validation += acc
                    summaries = sess.run(validation_summary_op, feed_dict={model.input_x : validation_batch_x, model.input_y : validation_batch_y, model.sizes : FLAGS.batch_size, model.seq_length : validation_batch_l})
                    validation_summary_writer.add_summary(summaries, current_step)
                    print("valoss=", "{:.6f}".format(loss), ", vaacc=", "{:.5f}".format(acc)+"AUC=")
                    print(auc)
                    print("AUC_other=")
                    print(auc_a)
                    print("macro_f1=")
                    print(macro_f1)
                    print("micro_f1=")
                    print(micro_f1)
                acc_validation /= validation_iter
                print("vaacc=", "{:.5f}".format(acc_validation))
                if acc_validation >= max_acc:
                    max_acc = acc_validation
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            if current_step % FLAGS.save_step == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        for test_ite in range(test_iter):
            test_batch_x, test_batch_l, test_batch_la,test_batch_y = sequences.test.Next_batch(FLAGS.batch_size)
            test_batch_x = test_batch_x.reshape((FLAGS.batch_size, FLAGS.n_steps, FLAGS.n_inputs))
            acc = sess.run(model.accuracy, feed_dict={model.input_x : test_batch_x, model.input_y : test_batch_y, model.sizes : FLAGS.batch_size, model.seq_length : test_batch_l})
            loss = sess.run(model.loss, feed_dict={model.input_x : test_batch_x, model.input_y : test_batch_y, model.sizes : FLAGS.batch_size, model.seq_length : test_batch_l})
            predicted_y=sess.run(model.predicted_y, feed_dict={model.input_x : test_batch_x, model.input_y : test_batch_y, model.sizes : FLAGS.batch_size, model.seq_length : test_batch_l})
            fpr, tpr, thresholds = metrics.roc_curve(test_batch_la.ravel(),predicted_y.ravel())
            auc_a=metrics.auc(fpr, tpr)
            auc=metrics.roc_auc_score(test_batch_la,predicted_y,average='micro')
            macro_f1 = metrics.f1_score(test_batch_la,predicted_y, average='macro')
            micro_f1 = metrics.f1_score(test_batch_la,predicted_y, average='micro')
            print("valoss=", "{:.6f}".format(loss), ", vaacc=", "{:.5f}".format(acc)+"AUC=")
            print(auc)
            print("AUC_other=")
            print(auc_a)
            print("macro_f1=")
            print(macro_f1)
            print("micro_f1=")
            print(micro_f1)
        endtime = datetime.datetime.now()
        print (str((endtime-starttime).seconds), "seconds")

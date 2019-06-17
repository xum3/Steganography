# -*- coding: utf-8 -*-
from model import Model
from gen_data import GenData

import os
import tensorflow as tf

LOGS_Path = "./logs/"
CHECKPOINTS_PATH = './checkpoints2/'


BATCH_SIZE = 8
LEARNING_RATE = .0001
BETA = .75

EXP_NAME = f"beta_{BETA}"


if __name__ == "__main__":
    model = Model()
    data = GenData('./optimization-ii-project-3/')
    files_list = data.files_list
    sess = tf.InteractiveSession(graph=tf.Graph(), config=tf.ConfigProto(log_device_placement=True))
    secret_tensor = tf.placeholder(shape=[None,224,224,3],dtype=tf.float32,name="input_prep")
    cover_tensor = tf.placeholder(shape=[None,224,224,3],dtype=tf.float32,name="input_hide")
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    train_op , summary_op, loss_op,secret_loss_op,cover_loss_op = model.prepare_training_graph(secret_tensor,cover_tensor,global_step_tensor)

    writer = tf.summary.FileWriter(os.path.join(LOGS_Path,EXP_NAME),sess.graph)

    test_op, test_loss_op,test_secret_loss_op,test_cover_loss_op = model.prepare_test_graph(secret_tensor,cover_tensor)

    covered_tensor = tf.placeholder(shape=[None,224,224,3],dtype=tf.float32,name="deploy_covered")
    deploy_hide_image_op , deploy_reveal_image_op = model.prepare_deployment_graph(secret_tensor,cover_tensor,covered_tensor)

    saver = tf.train.Saver(max_to_keep=1)
    sess.run(tf.global_variables_initializer())

    total_steps = len(files_list)//BATCH_SIZE + 1

    min_loss = 10
    for step in range(3000):
        covers,secrets = data.get_img_batch(batch_size=BATCH_SIZE)
        sess.run([train_op],feed_dict={"input_prep:0":secrets,"input_hide:0":covers})
        if step % 10 ==0 :
            summary,global_step, loss,secret_loss,cover_loss = sess.run([summary_op,global_step_tensor, loss_op,secret_loss_op,cover_loss_op],feed_dict={"input_prep:0":secrets,"input_hide:0":covers})
            writer.add_summary(summary,global_step)
            print("train global_step is {}, total loss is  {:.4f}, secret_loss is {:.4f}, cover_loss is {:.4f}".format(global_step, loss,secret_loss,cover_loss))
            
        if step % 100 ==0 :
            covers,secrets = data.get_img_batch(batch_size=1)
            summary,global_step, test_loss,test_secret_loss,test_cover_loss = sess.run([test_op,global_step_tensor, test_loss_op,test_secret_loss_op,test_cover_loss_op],feed_dict={"input_prep:0":secrets,"input_hide:0":covers})
            writer.add_summary(summary,global_step)
            print("test global_step is {}, total loss is  {:.4f}, secret_loss is {:.4f}, cover_loss is {:.4f}".format(global_step, test_loss,test_secret_loss,test_cover_loss))

            if step % 300 ==0 :
                save_path = saver.save(sess, os.path.join(CHECKPOINTS_PATH,EXP_NAME+"{:.4f}".format(test_loss)),global_step=global_step)

        if test_loss < min_loss:
            min_loss = test_loss
            save_path = saver.save(sess, os.path.join(CHECKPOINTS_PATH,EXP_NAME+"{:.4f}".format(test_loss)),global_step=global_step)
        
        

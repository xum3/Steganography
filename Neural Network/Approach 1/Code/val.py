# -*- coding: utf-8 -*-
from model import Model
from gen_data import GenData

import os
import tensorflow as tf
import numpy as np


model = Model()
data = GenData('./optimization-ii-project-3/')
files_list = data.files_list

secret_tensor = tf.placeholder(shape=[None,224,224,3],dtype=tf.float32,name="input_prep")
cover_tensor = tf.placeholder(shape=[None,224,224,3],dtype=tf.float32,name="input_hide")
global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

train_op , summary_op, loss_op,secret_loss_op,cover_loss_op = model.prepare_training_graph(secret_tensor,cover_tensor,global_step_tensor)
test_op, test_loss_op,test_secret_loss_op,test_cover_loss_op = model.prepare_test_graph(secret_tensor,cover_tensor)

covered_tensor = tf.placeholder(shape=[None,224,224,3],dtype=tf.float32,name="deploy_covered")
deploy_hide_image_op = model.encode(secret_tensor,cover_tensor)
deploy_reveal_image_op = model.decode(covered_tensor)

saver = tf.train.Saver()
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
saver.restore(sess, './checkpoints/beta_0.750.1396-2101')



from matplotlib import pyplot as plt
for step in range(3000):
    covers,secrets = data.get_img_batch(batch_size=1)
    deploy_hide_image = sess.run([deploy_hide_image_op],feed_dict={"input_prep:0":secrets,"input_hide:0":covers})
    deploy_reveal_image = sess.run([deploy_reveal_image_op],feed_dict={"deploy_covered:0":covers})

    covers_image = data.denormalize_batch(covers.squeeze())
    secrets_image = data.denormalize_batch(secrets.squeeze())
    hide_image = data.denormalize_batch(deploy_hide_image[0].squeeze())
    reveal_image = data.denormalize_batch(deploy_reveal_image[0].squeeze())
    plt.imshow( np.hstack( (covers_image,secrets_image,hide_image,reveal_image) ) )
    plt.show()
    exit()

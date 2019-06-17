# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

LOGS_Path = "./logs/"
CHECKPOINTS_PATH = './checkpoints/'


BATCH_SIZE = 8
LEARNING_RATE = .000005
BETA = .75

EXP_NAME = f"beta_{BETA}"

class Model():
    
    def get_prep_network_op(self, secret_tensor):
        with tf.variable_scope('prep_net'):
            
            with tf.variable_scope("3x3_conv_branch"):
                conv_3x3 = tf.layers.conv2d(inputs=secret_tensor,filters=50,kernel_size=3,padding='same',name="1",activation=tf.nn.relu)
                # conv_3x3 = tf.layers.conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="2",activation=tf.nn.relu)
                # conv_3x3 = tf.layers.conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="3",activation=tf.nn.relu)
                # conv_3x3 = tf.layers.conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="4",activation=tf.nn.relu)
                
            with tf.variable_scope("4x4_conv_branch"):
                conv_4x4 = tf.layers.conv2d(inputs=secret_tensor,filters=50,kernel_size=4,padding='same',name="1",activation=tf.nn.relu)
                # conv_4x4 = tf.layers.conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="2",activation=tf.nn.relu)           
                # conv_4x4 = tf.layers.conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="3",activation=tf.nn.relu)
                # conv_4x4 = tf.layers.conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="4",activation=tf.nn.relu)

            with tf.variable_scope("5x5_conv_branch"):
                conv_5x5 = tf.layers.conv2d(inputs=secret_tensor,filters=50,kernel_size=5,padding='same',name="1",activation=tf.nn.relu)
                # conv_5x5 = tf.layers.conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="2",activation=tf.nn.relu)           
                # conv_5x5 = tf.layers.conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="3",activation=tf.nn.relu)
                # conv_5x5 = tf.layers.conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="4",activation=tf.nn.relu)
                
            concat_1 = tf.concat([conv_3x3,conv_4x4,conv_5x5],axis=3,name='concat_1')
            
            conv_5x5 = tf.layers.conv2d(inputs=concat_1,filters=50,kernel_size=5,padding='same',name="final_5x5",activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(inputs=concat_1,filters=50,kernel_size=4,padding='same',name="final_4x4",activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(inputs=concat_1,filters=50,kernel_size=3,padding='same',name="final_3x3",activation=tf.nn.relu)
            
            concat_final = tf.concat([conv_5x5,conv_4x4,conv_3x3],axis=3,name='concat_final')

            return concat_final


    def get_hiding_network_op(self, cover_tensor,prep_output):
    
        with tf.variable_scope('hide_net'):
            concat_input = tf.concat([cover_tensor,prep_output],axis=3,name='images_features_concat')
            
            with tf.variable_scope("3x3_conv_branch"):
                conv_3x3 = tf.layers.conv2d(inputs=concat_input,filters=50,kernel_size=3,padding='same',name="1",activation=tf.nn.relu)
                conv_3x3 = tf.layers.conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="2",activation=tf.nn.relu)
                # conv_3x3 = tf.layers.conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="3",activation=tf.nn.relu)
                # conv_3x3 = tf.layers.conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="4",activation=tf.nn.relu)
                
            with tf.variable_scope("4x4_conv_branch"):
                conv_4x4 = tf.layers.conv2d(inputs=concat_input,filters=50,kernel_size=4,padding='same',name="1",activation=tf.nn.relu)
                conv_4x4 = tf.layers.conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="2",activation=tf.nn.relu)          
                # conv_4x4 = tf.layers.conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="3",activation=tf.nn.relu)
                # conv_4x4 = tf.layers.conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="4",activation=tf.nn.relu)

            with tf.variable_scope("5x5_conv_branch"):
                conv_5x5 = tf.layers.conv2d(inputs=concat_input,filters=50,kernel_size=5,padding='same',name="1",activation=tf.nn.relu)
                conv_5x5 = tf.layers.conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="2",activation=tf.nn.relu)          
                # conv_5x5 = tf.layers.conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="3",activation=tf.nn.relu)
                # conv_5x5 = tf.layers.conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="4",activation=tf.nn.relu)
                
            concat_1 = tf.concat([conv_3x3,conv_4x4,conv_5x5],axis=3,name='concat_1')
            
            conv_5x5 = tf.layers.conv2d(inputs=concat_1,filters=50,kernel_size=5,padding='same',name="final_5x5",activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(inputs=concat_1,filters=50,kernel_size=4,padding='same',name="final_4x4",activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(inputs=concat_1,filters=50,kernel_size=3,padding='same',name="final_3x3",activation=tf.nn.relu)
            
            concat_final = tf.concat([conv_5x5,conv_4x4,conv_3x3],axis=3,name='concat_final')
            output = tf.layers.conv2d(inputs=concat_final,filters=3,kernel_size=1,padding='same',name='output')
            
            return output

    
    def get_reveal_network_op(self, container_tensor):
    
        with tf.variable_scope('reveal_net'):
            
            with tf.variable_scope("3x3_conv_branch"):
                conv_3x3 = tf.layers.conv2d(inputs=container_tensor,filters=50,kernel_size=3,padding='same',name="1",activation=tf.nn.relu)
                conv_3x3 = tf.layers.conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="2",activation=tf.nn.relu)
                # conv_3x3 = tf.layers.conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="3",activation=tf.nn.relu)
                # conv_3x3 = tf.layers.conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="4",activation=tf.nn.relu)
                
            with tf.variable_scope("4x4_conv_branch"):
                conv_4x4 = tf.layers.conv2d(inputs=container_tensor,filters=50,kernel_size=4,padding='same',name="1",activation=tf.nn.relu)
                conv_4x4 = tf.layers.conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="2",activation=tf.nn.relu)          
                # conv_4x4 = tf.layers.conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="3",activation=tf.nn.relu)
                # conv_4x4 = tf.layers.conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="4",activation=tf.nn.relu)

            with tf.variable_scope("5x5_conv_branch"):
                conv_5x5 = tf.layers.conv2d(inputs=container_tensor,filters=50,kernel_size=5,padding='same',name="1",activation=tf.nn.relu)
                conv_5x5 = tf.layers.conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="2",activation=tf.nn.relu)           
                # conv_5x5 = tf.layers.conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="3",activation=tf.nn.relu)
                # conv_5x5 = tf.layers.conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="4",activation=tf.nn.relu)
                
            concat_1 = tf.concat([conv_3x3,conv_4x4,conv_5x5],axis=3,name='concat_1')
            
            conv_5x5 = tf.layers.conv2d(inputs=concat_1,filters=50,kernel_size=5,padding='same',name="final_5x5",activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(inputs=concat_1,filters=50,kernel_size=4,padding='same',name="final_4x4",activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(inputs=concat_1,filters=50,kernel_size=3,padding='same',name="final_3x3",activation=tf.nn.relu)
            
            concat_final = tf.concat([conv_5x5,conv_4x4,conv_3x3],axis=3,name='concat_final')
        
        output = tf.layers.conv2d(inputs=concat_final,filters=3,kernel_size=1,padding='same',name='output')

        return output

    def get_noise_layer_op(self, tensor, std=.1):
        with tf.variable_scope("noise_layer"):
            return tensor + tf.random_normal(shape=tf.shape(tensor), mean=0.0, stddev=std, dtype=tf.float32) 
        
    def get_loss_op(self, secret_true,secret_pred,cover_true,cover_pred,beta=.5):
        
        with tf.variable_scope("losses"):
            beta = tf.constant(beta,name="beta")
            secret_mse = tf.losses.mean_squared_error(secret_true,secret_pred)
            cover_mse = tf.losses.mean_squared_error(cover_true,cover_pred)
            final_loss = cover_mse + beta*secret_mse
            return final_loss , secret_mse , cover_mse 

    def get_tensor_to_img_op(self, tensor):
        with tf.variable_scope("",reuse=True):
            t = tensor*tf.convert_to_tensor([0.229, 0.224, 0.225]) + tf.convert_to_tensor([0.485, 0.456, 0.406])
            return tf.clip_by_value(t,0,1)

    def prepare_training_graph(self, secret_tensor,cover_tensor,global_step_tensor):
        
        prep_output_op = self.get_prep_network_op(secret_tensor)
        hiding_output_op = self.get_hiding_network_op(cover_tensor=cover_tensor,prep_output=prep_output_op)
        noise_add_op = self.get_noise_layer_op(hiding_output_op)
        reveal_output_op = self.get_reveal_network_op(noise_add_op)
        
        loss_op,secret_loss_op,cover_loss_op = self.get_loss_op(secret_tensor,reveal_output_op,cover_tensor,hiding_output_op,beta=BETA)

        minimize_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_op,global_step=global_step_tensor)
        
        tf.summary.scalar('loss', loss_op,family='train')
        tf.summary.scalar('reveal_net_loss', secret_loss_op,family='train')
        tf.summary.scalar('cover_net_loss', cover_loss_op,family='train')

        tf.summary.image('secret', self.get_tensor_to_img_op(secret_tensor), max_outputs=1, family='train')
        tf.summary.image('cover', self.get_tensor_to_img_op(cover_tensor), max_outputs=1, family='train')
        tf.summary.image('hidden', self.get_tensor_to_img_op(hiding_output_op), max_outputs=1, family='train')
        tf.summary.image('hidden_noisy', self.get_tensor_to_img_op(noise_add_op), max_outputs=1, family='train')
        tf.summary.image('revealed', self.get_tensor_to_img_op(reveal_output_op), max_outputs=1, family='train')

        merged_summary_op = tf.summary.merge_all()
        
        return minimize_op, merged_summary_op, loss_op,secret_loss_op,cover_loss_op

    def prepare_test_graph(self, secret_tensor,cover_tensor):
        with tf.variable_scope("",reuse=True):
        
            prep_output_op = self.get_prep_network_op(secret_tensor)
            hiding_output_op = self.get_hiding_network_op(cover_tensor=cover_tensor,prep_output=prep_output_op)
            reveal_output_op = self.get_reveal_network_op(hiding_output_op)
            
            loss_op,secret_loss_op,cover_loss_op = self.get_loss_op(secret_tensor,reveal_output_op,cover_tensor,hiding_output_op)

            tf.summary.scalar('loss', loss_op,family='test')
            tf.summary.scalar('reveal_net_loss', secret_loss_op,family='test')
            tf.summary.scalar('cover_net_loss', cover_loss_op,family='test')

            tf.summary.image('secret',self.get_tensor_to_img_op(secret_tensor),max_outputs=1,family='test')
            tf.summary.image('cover',self.get_tensor_to_img_op(cover_tensor),max_outputs=1,family='test')
            tf.summary.image('hidden',self.get_tensor_to_img_op(hiding_output_op),max_outputs=1,family='test')
            tf.summary.image('revealed',self.get_tensor_to_img_op(reveal_output_op),max_outputs=1,family='test')

            merged_summary_op = tf.summary.merge_all()

            return merged_summary_op, loss_op,secret_loss_op,cover_loss_op

    def prepare_deployment_graph(self, secret_tensor,cover_tensor,covered_tensor):
        with tf.variable_scope("",reuse=True):

            prep_output_op = self.get_prep_network_op(secret_tensor)
            hiding_output_op = self.get_hiding_network_op(cover_tensor=cover_tensor,prep_output=prep_output_op)

            reveal_output_op = self.get_reveal_network_op(covered_tensor)

            return hiding_output_op ,  reveal_output_op
    def encode(self, secret_tensor,cover_tensor):
        with tf.variable_scope("",reuse=True):

            prep_output_op = self.get_prep_network_op(secret_tensor)
            hiding_output_op = self.get_hiding_network_op(cover_tensor=cover_tensor,prep_output=prep_output_op)
            return hiding_output_op

    def decode(self, covered_tensor):
        with tf.variable_scope("",reuse=True):
            reveal_output_op = self.get_reveal_network_op(covered_tensor)

            return reveal_output_op
import math
import os
import tensorflow as tf
import numpy as np

from base_model import *
from utils.nn import *
from episodic_memory import *
from collections import namedtuple

class QuestionAnswerer(BaseModel):
    def build(self):
        """ Build the model. """
        self.build_cnn()
        self.build_rnn()

    def build_cnn(self):
        """ Build the CNN. """
        print("Building the CNN part...")
        if self.cnn_model=='vgg16':
            self.build_vgg16()
        elif self.cnn_model=='resnet50':
            self.build_resnet50()
        elif self.cnn_model=='resnet101':
            self.build_resnet101()
        elif self.cnn_model=='resnet_cifar10':
            self.build_resnet32_cifar10()
        else:
            self.build_resnet152()
        print("CNN part built.")

    def build_vgg16(self):
        """ Build the VGG16 net. """
        bn = self.params.batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_1_feats = convolution(imgs, 3, 3, 64, 1, 1, 'conv1_1')
        conv1_1_feats = batch_norm(conv1_1_feats, 'bn1_1', is_train, bn, 'relu')
        conv1_2_feats = convolution(conv1_1_feats, 3, 3, 64, 1, 1, 'conv1_2')
        conv1_2_feats = batch_norm(conv1_2_feats, 'bn1_2', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_2_feats, 2, 2, 2, 2, 'pool1')

        conv2_1_feats = convolution(pool1_feats, 3, 3, 128, 1, 1, 'conv2_1')
        conv2_1_feats = batch_norm(conv2_1_feats, 'bn2_1', is_train, bn, 'relu')
        conv2_2_feats = convolution(conv2_1_feats, 3, 3, 128, 1, 1, 'conv2_2')
        conv2_2_feats = batch_norm(conv2_2_feats, 'bn2_2', is_train, bn, 'relu')
        pool2_feats = max_pool(conv2_2_feats, 2, 2, 2, 2, 'pool2')

        conv3_1_feats = convolution(pool2_feats, 3, 3, 256, 1, 1, 'conv3_1')
        conv3_1_feats = batch_norm(conv3_1_feats, 'bn3_1', is_train, bn, 'relu')
        conv3_2_feats = convolution(conv3_1_feats, 3, 3, 256, 1, 1, 'conv3_2')
        conv3_2_feats = batch_norm(conv3_2_feats, 'bn3_2', is_train, bn, 'relu')
        conv3_3_feats = convolution(conv3_2_feats, 3, 3, 256, 1, 1, 'conv3_3')
        conv3_3_feats = batch_norm(conv3_3_feats, 'bn3_3', is_train, bn, 'relu')
        pool3_feats = max_pool(conv3_3_feats, 2, 2, 2, 2, 'pool3')

        conv4_1_feats = convolution(pool3_feats, 3, 3, 512, 1, 1, 'conv4_1')
        conv4_1_feats = batch_norm(conv4_1_feats, 'bn4_1', is_train, bn, 'relu')
        conv4_2_feats = convolution(conv4_1_feats, 3, 3, 512, 1, 1, 'conv4_2')
        conv4_2_feats = batch_norm(conv4_2_feats, 'bn4_2', is_train, bn, 'relu')
        conv4_3_feats = convolution(conv4_2_feats, 3, 3, 512, 1, 1, 'conv4_3')
        conv4_3_feats = batch_norm(conv4_3_feats, 'bn4_3', is_train, bn, 'relu')
        pool4_feats = max_pool(conv4_3_feats, 2, 2, 2, 2, 'pool4')

        conv5_1_feats = convolution(pool4_feats, 3, 3, 512, 1, 1, 'conv5_1')
        conv5_1_feats = batch_norm(conv5_1_feats, 'bn5_1', is_train, bn, 'relu')
        conv5_2_feats = convolution(conv5_1_feats, 3, 3, 512, 1, 1, 'conv5_2')
        conv5_2_feats = batch_norm(conv5_2_feats, 'bn5_2', is_train, bn, 'relu')
        conv5_3_feats = convolution(conv5_2_feats, 3, 3, 512, 1, 1, 'conv5_3')
        conv5_3_feats = batch_norm(conv5_3_feats, 'bn5_3', is_train, bn, 'relu')

        self.permutation = self.get_permutation(14, 14)
        conv5_3_feats.set_shape([self.batch_size, 14, 14, 512])
        conv5_3_feats_flat = self.flatten_feats(conv5_3_feats, 512)
        self.conv_feats = conv5_3_feats_flat
        self.conv_feat_shape = [196, 512]

        self.imgs = imgs
        self.is_train = is_train
        self.cnn_saver = tf.train.Saver()

    def basic_block(self, input_feats, name1, name2, is_train, bn, c, s=2):
        """ A basic block of ResNets. """
        branch1_feats = convolution_no_bias(input_feats, 1, 1, 4*c, s, s, name1+'_branch1')
        branch1_feats = batch_norm(branch1_feats, name2+'_branch1', is_train, bn, None)

        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, s, s, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, bn, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, bn, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, bn, None)

        output_feats = branch1_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def basic_block2(self, input_feats, name1, name2, is_train, bn, c):
        """ Another basic block of ResNets. """
        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, 1, 1, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, bn, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, bn, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, bn, None)

        output_feats = input_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def build_resnet50(self):
        """ Build the ResNet50 net. """
        bn = self.params.batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)     

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)
        res3b_feats = self.basic_block2(res3a_feats, 'res3b', 'bn3b', is_train, bn, 128)
        res3c_feats = self.basic_block2(res3b_feats, 'res3c', 'bn3c', is_train, bn, 128)
        res3d_feats = self.basic_block2(res3c_feats, 'res3d', 'bn3d', is_train, bn, 128)

        res4a_feats = self.basic_block(res3d_feats, 'res4a', 'bn4a', is_train, bn, 256)
        res4b_feats = self.basic_block2(res4a_feats, 'res4b', 'bn4b', is_train, bn, 256)
        res4c_feats = self.basic_block2(res4b_feats, 'res4c', 'bn4c', is_train, bn, 256)
        res4d_feats = self.basic_block2(res4c_feats, 'res4d', 'bn4d', is_train, bn, 256)
        res4e_feats = self.basic_block2(res4d_feats, 'res4e', 'bn4e', is_train, bn, 256)
        res4f_feats = self.basic_block2(res4e_feats, 'res4f', 'bn4f', is_train, bn, 256)

        res5a_feats = self.basic_block(res4f_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        self.permutation = self.get_permutation(7, 7)
        res5c_feats.set_shape([self.batch_size, 7, 7, 2048])
        res5c_feats_flat = self.flatten_feats(res5c_feats, 2048)
        self.conv_feats = res5c_feats_flat
        self.conv_feat_shape = [49, 2048]

        self.imgs = imgs
        self.is_train = is_train
        self.cnn_saver = tf.train.Saver()

######

    # Fundamental Layers
    def affine (self, x, in_dim, out_dim, name):
        with tf.variable_scope(name):
            w = tf.get_variable('DW',
                [in_dim, out_dim], 
                initializer=tf.uniform_unit_scaling_initializer(factor=1.0)
                                #tf.contrib.layers.xavier_initializer()
            )
            b = tf.get_variable('biases', 
                                [out_dim],
                               initializer=tf.constant_initializer(1, tf.float32)
                               )
            return tf.nn.xw_plus_b(x,w,b)
    
    def relu (self, x, leakiness=0.0):
        return tf.where(tf.less(x,0.0), leakiness*x, x, name='leaky_relu')
    
    
    def conv2d (self, x, filter_size, in_filters, out_filters, stride, name):
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
            'DW', [filter_size,filter_size, in_filters, out_filters],
            initializer=tf.random_normal_initializer(
                  stddev=np.sqrt(2.0/n))
            )
            return tf.nn.conv2d(x, kernel, [1, stride, stride,1], padding='SAME')
    
    def global_avg_pool(self, x, name):
        with tf.variable_scope(name):
            return tf.reduce_mean(x, [1,2])
    
    # Complex Cells
    def residual (self,x, in_filter, out_filter, stride, name, is_training,
                  relu_leakiness=0.0, keep_prob=1.0, activate_before_residual=False):
        with tf.variable_scope(name):
            if activate_before_residual:
                with tf.variable_scope("shared_residual_activation"):
                    x = tf.layers.batch_normalization(x,training=is_training)
                    x = self.relu(x, relu_leakiness)
                    orig_x=x
            else:
                with tf.variable_scope("residual_only_activation"):
                    orig_x = x
                    x = tf.layers.batch_normalization(x,training=is_training)
                    x = self.relu(x, relu_leakiness)
            with tf.variable_scope('sub_unit1'):
                x = self.conv2d(x,3, in_filter, out_filter, stride, "conv3x3")
            with tf.variable_scope('sub_unit2'):
                x = tf.layers.batch_normalization(x,training=is_training)
                x = self.relu(x, relu_leakiness)
                x = self.conv2d(x,3, out_filter, out_filter, 1, "conv3x3")
            if keep_prob  < 1.0:
                x = tf.layers.dropout(x, rate=(1-keep_prob) , training=is_training)
            with tf.variable_scope('add'):
                if in_filter != out_filter:
                    orig_x = tf.nn.avg_pool(orig_x, [1,stride,stride,1], [1,stride,stride,1], 'VALID')
                    orig_x = tf.pad(
                        orig_x, [[0, 0], [0, 0], [0, 0],
                                 [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
    
                x += orig_x
        return x
    
    def residual_unit(self, x, in_filter, out_filter, stride, name, is_training,
                  relu_leakiness=0.0, keep_prob=1.0, activate_before_residual=False, num_units=5):
        with tf.variable_scope(name):
            x = self.residual (x, in_filter, out_filter, stride, "residual_1", is_training,
                  relu_leakiness, self.hps.keep_prob, activate_before_residual)
         
           
        for i in range(1, num_units):
            with tf.variable_scope('%s_%d' % (name, i)):
                x  = self.residual (x, out_filter, out_filter, 1, "residual_1", is_training,
                  relu_leakiness, self.hps.keep_prob)
                
        
        return x

    def build_resnet32_cifar10(self):
        HParams= namedtuple('HParams',
                     'batch_size, num_classes,'
                     'num_residual_units, weight_decay_rate, '
                     'relu_leakiness, keep_prob')
        self.hps = HParams(batch_size=100,
                             num_classes=10,
                             num_residual_units=5,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             keep_prob=0.8,
                             )
        X = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape, name="x")
        is_training = tf.placeholder(tf.bool, name="is_training")
        strides=[1, 2, 2]
        filters=[16, 16, 32, 64]
        activate_before_residual = [True, False, False]
        num_classes=10
        
        # 3x3 Convolutional Layer with 16 filters and stride of 1
        with tf.variable_scope("init"):
            x = self.conv2d(X, 3, 3, filters[1], strides[0], "CONV_3X3_S1X16")
            
        x=self.residual_unit (x, filters[0], filters[1], strides[0], "unit_1", is_training,
                  self.hps.relu_leakiness, self.hps.keep_prob, activate_before_residual[0], self.hps.num_residual_units)
        x_int=self.residual_unit (x, filters[1], filters[2], strides[1], "unit_2_", is_training,
                  self.hps.relu_leakiness, self.hps.keep_prob, activate_before_residual[0], self.hps.num_residual_units)
        x=self.residual_unit (x_int, filters[2], filters[3], strides[2], "unit_3", is_training,
                  self.hps.relu_leakiness, self.hps.keep_prob, activate_before_residual[0], self.hps.num_residual_units)

        
        x_int = max_pool(x_int, 2, 2, 2, 2, 'pool_addo')
        self.permutation = self.get_permutation(7, 7)
        #x_int.set_shape([self.batch_size, 7, 7, 2048])
        x_int = tf.reshape(x_int, [self.batch_size, 7, 7, 2048])
        x_flat = self.flatten_feats(x_int, 2048)
        self.conv_feats = x_flat
        self.conv_feat_shape = [49, 2048]

        self.imgs = X
        self.is_train = is_training

        # Creating a saver instance just for CNN
        
        # Fist create a dictionary of param names: param to pass to the saver instance
        self.cnn_params = {}
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                self.cnn_params[var.op.name] = var
        self.cnn_saver = tf.train.Saver(self.cnn_params)

######
    

    def build_resnet101(self):
        """ Build the ResNet101 net. """
        bn = self.params.batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)       
        temp = res3a_feats
        for i in range(1, 4):
            temp = self.basic_block2(temp, 'res3b'+str(i), 'bn3b'+str(i), is_train, bn, 128)
        res3b3_feats = temp
 
        res4a_feats = self.basic_block(res3b3_feats, 'res4a', 'bn4a', is_train, bn, 256)
        temp = res4a_feats
        for i in range(1, 23):
            temp = self.basic_block2(temp, 'res4b'+str(i), 'bn4b'+str(i), is_train, bn, 256)
        res4b22_feats = temp

        res5a_feats = self.basic_block(res4b22_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        self.permutation = self.get_permutation(7, 7)
        res5c_feats.set_shape([self.batch_size, 7, 7, 2048])
        res5c_feats_flat = self.flatten_feats(res5c_feats, 2048)
        self.conv_feats = res5c_feats_flat
        self.conv_feat_shape = [49, 2048]

        self.imgs = imgs
        self.is_train = is_train
        self.cnn_saver = tf.train.Saver()

    def build_resnet152(self):
        """ Build the ResNet152 net. """
        bn = self.params.batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)       
        temp = res3a_feats
        for i in range(1, 8):
            temp = self.basic_block2(temp, 'res3b'+str(i), 'bn3b'+str(i), is_train, bn, 128)
        res3b7_feats = temp
 
        res4a_feats = self.basic_block(res3b7_feats, 'res4a', 'bn4a', is_train, bn, 256)
        temp = res4a_feats
        for i in range(1, 36):
            temp = self.basic_block2(temp, 'res4b'+str(i), 'bn4b'+str(i), is_train, bn, 256)
        res4b35_feats = temp

        res5a_feats = self.basic_block(res4b35_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        self.permutation = self.get_permutation(7, 7)
        res5c_feats.set_shape([self.batch_size, 7, 7, 2048])
        res5c_feats_flat = self.flatten_feats(res5c_feats, 2048)
        self.conv_feats = res5c_feats_flat
        self.conv_feat_shape = [49, 2048]

        self.img_files = img_files
        self.is_train = is_train
        self.cnn_saver = tf.train.Saver()

    def get_permutation(self, height, width):
        """ Get the permutation corresponding to a snake-like walk as decribed by the paper. Used to flatten the convolutional feats. """
        permutation = np.zeros(height*width, np.int32)
        for i in range(height):
            for j in range(width):
                permutation[i*width+j] = i*width+j if i%2==0 else (i+1)*width-j-1
        return permutation

    def flatten_feats(self, feats, channels):
        """ Flatten the feats. """
        temp1 = tf.reshape(feats, [self.batch_size, -1, channels])
        temp1 = tf.transpose(temp1, [1, 0, 2])
        temp2 = tf.gather(temp1, self.permutation)
        temp2 = tf.transpose(temp2, [1, 0, 2])
        return temp2

    def build_rnn(self):
        """ Build the RNN. """
        print("Building the RNN part...")
        params = self.params
        bn = params.batch_norm      
        is_train = self.is_train
        batch_size = self.batch_size                     

        dim_hidden = params.dim_hidden                     
        dim_embed = params.dim_embed                       
        max_ques_len = params.max_ques_len                 

        num_facts = self.conv_feat_shape[0]                                      
        dim_fact = self.conv_feat_shape[1]                                      
        num_words = self.word_table.num_words              

        self.word_weight = np.exp(-np.array(self.word_table.word_freq)*self.class_balancing_factor)

        if not self.train_cnn:
            facts = tf.placeholder(tf.float32, [batch_size, num_facts, dim_fact])   
        else:
            facts = self.conv_feats

        questions = tf.placeholder(tf.int32, [batch_size, max_ques_len])        
        question_lens = tf.placeholder(tf.int32, [batch_size])                   
        answers = tf.placeholder(tf.int32, [batch_size])                        
        answer_weights = tf.placeholder(tf.float32, [batch_size])                        
        
        gru = tf.contrib.rnn.GRUCell(dim_hidden)

        # Initialize the word embedding
        idx2vec = np.array([self.word_table.word2vec[self.word_table.idx2word[i]] for i in range(num_words)])
        with tf.variable_scope('embedding'):
            if params.fix_embed_weight:
                emb_w = tf.convert_to_tensor(idx2vec, tf.float32)                       
            else:
                emb_w = weight('emb_w', [num_words, dim_embed], init_val=idx2vec, group_id=1)

        # Encode the questions
        with tf.variable_scope('Question'):
            word_list = tf.unstack(questions, axis=1)                                             
            ques_embed = [tf.nn.embedding_lookup(emb_w, word) for word in word_list]             
            ques_embed = tf.transpose(tf.stack(ques_embed), [1, 0, 2])   

            all_states, final_state = tf.nn.dynamic_rnn(gru, ques_embed, dtype=tf.float32)       

            question_enc = []
            for k in range(batch_size):
                current_ques_enc = tf.slice(all_states, [k, question_lens[k]-1, 0], [1, 1, dim_hidden]) 
                question_enc.append(tf.squeeze(current_ques_enc))

            question_enc = tf.stack(question_enc)                                                 
           #ques_enc = final_state

        # Encode the facts
        with tf.name_scope('InputFusion'):

            with tf.variable_scope('Forward'):
                forward_states, _ = tf.nn.dynamic_rnn(gru, facts, dtype=tf.float32)           

            with tf.variable_scope('Backward'):
                reversed_facts = tf.reverse(facts, [1])                   
                backward_states, _ = tf.nn.dynamic_rnn(gru, reversed_facts, dtype=tf.float32) 
                backward_states = tf.reverse(backward_states, [1])              

            facts_enc = forward_states + backward_states                                      

        # Episodic Memory Update
        with tf.variable_scope('EpisodicMemory'):
            episode = EpisodicMemory(dim_hidden, num_facts, question_enc, facts_enc, params.attention, is_train, bn)
            memory = tf.identity(question_enc)                                                   
            
            # Tied memory weights
            if params.tie_memory_weight: 
                with tf.variable_scope('Layer') as scope:
                    for t in range(params.memory_step):
                        if params.memory_update == 'gru': 
                            memory = gru(episode.new_fact(memory), memory)[0]                     
                        else:
                            fact = episode.new_fact(memory)                                        
                            expanded_memory = tf.concat([memory, fact, question_enc], 1)           
                            memory = fully_connected(expanded_memory, dim_hidden, 'EM_fc', group_id=1)
                            memory = batch_norm(memory, 'EM_bn', is_train, bn, 'relu')  
                        scope.reuse_variables()

            # Untied memory weights
            else:
                for t in range(params.memory_step):
                    with tf.variable_scope('Layer%d' %t) as scope:
                        if params.memory_update == 'gru':
                            memory = gru(episode.new_fact(memory), memory)[0]                     
                        else:
                            fact = episode.new_fact(memory)                                        
                            expanded_memory = tf.concat([memory, fact, question_enc], 1)           
                            memory = fully_connected(expanded_memory, dim_hidden, 'EM_fc', group_id=1)
                            memory = batch_norm(memory, 'EM_bn', is_train, bn, 'relu')  

        memory = dropout(memory, 0.5, is_train)                                                  
        
        # Compute the result
        with tf.variable_scope('Result'):    
            expanded_memory = tf.concat([memory, question_enc], 1)    
            logits = fully_connected(expanded_memory, num_words, 'dec', group_id=1)
            results = tf.argmax(logits, 1)                                                        
            all_probs = tf.nn.softmax(logits)                                                    
            probs = tf.reduce_max(all_probs, 1)                                                      

        # Compute the loss
        with tf.variable_scope('Loss'):        
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=answers) 
            loss0 = cross_entropy * answer_weights
            loss0 = tf.reduce_sum(loss0) / tf.reduce_sum(answer_weights)
            if self.train_cnn:
                loss1 = params.weight_decay * (tf.add_n(tf.get_collection('l2_0')) + tf.add_n(tf.get_collection('l2_1')))  
            else:
                loss1 = params.weight_decay * tf.add_n(tf.get_collection('l2_1'))
            loss = loss0 + loss1

        # Build the solver
        if params.solver == 'adam':
            solver = tf.train.AdamOptimizer(params.learning_rate)
        elif params.solver == 'momentum':
            solver = tf.train.MomentumOptimizer(params.learning_rate, params.momentum)
        elif params.solver == 'rmsprop':
            solver = tf.train.RMSPropOptimizer(params.learning_rate, params.decay, params.momentum)
        else:
            solver = tf.train.GradientDescentOptimizer(params.learning_rate)

        tvars = tf.trainable_variables()
        gs, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 3.0)
        opt_op = solver.apply_gradients(zip(gs, tvars), global_step=self.global_step)

        self.facts = facts
        self.questions = questions
        self.question_lens = question_lens
        self.answers = answers
        self.answer_weights = answer_weights

        self.loss = loss
        self.loss0 = loss0
        self.loss1 = loss1
        self.opt_op = opt_op

        self.results = results
        self.probs = probs
        
        print("RNN part built.")        

    def get_feed_dict(self, batch, is_train, feats=None):
        """ Get the feed dictionary for the current batch. """
        if is_train:
            # training phase
            img_files, questions, question_lens, answers = batch
            imgs = self.img_loader.load_imgs(img_files)
            answer_weights = self.word_weight[answers]
            if self.train_cnn:
                return {self.imgs: imgs, self.questions: questions, self.question_lens: question_lens, self.answers: answers, self.answer_weights: answer_weights, self.is_train: is_train}
            else:
                return {self.facts: feats, self.questions: questions, self.question_lens: question_lens, self.answers: answers, self.answer_weights: answer_weights, self.is_train: is_train}

        else:
            # testing or validation phase
            img_files, questions, question_lens = batch
            imgs = self.img_loader.load_imgs(img_files)
            if self.train_cnn: 
                return {self.imgs: imgs, self.questions: questions, self.question_lens: question_lens, self.is_train: is_train} 
            else: 
                return {self.facts: feats, self.questions: questions, self.question_lens: question_lens, self.is_train: is_train}


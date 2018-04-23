import tensorflow as tf
import numpy as np
from tf_utils import *
import os

class RoomnetVanilla(object):
  def __init__(self):
    self.batch_size=20
    self.s_in=320
    self.s_out=40
    self.l_list=tf.constant([0,8,14,20,24,28,34,38,42,44,46, 48])

  def set_placeholder(self):
    self.im_in=tf.placeholder(tf.float32,(self.batch_size,self.s_in, self.s_in,3),name='im_in')
    self.gt_layout=tf.placeholder(tf.float32, (self.batch_size,self.s_out, self.s_out, 48), name='layout')
    self.gt_label=tf.placeholder(tf.float32, (self.batch_size, 11), name='class_label')
    self.is_training = tf.placeholder(tf.bool, shape=())
  def set_feed(self, image, lay, c_label, step, is_training=True):
    self.feed_dict={
      self.im_in:image,
      self.gt_layout:lay,
      self.gt_label:c_label,
      self.global_step:step,
      # self.epoch=epoch,
      self.is_training:is_training
    }
  def get_bn_decay(self, batch):
    bn_momentum = tf.train.exponential_decay(0.5, batch * 20, 200000, 0.5,staircase=True)
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    return bn_decay
  def get_lr(self, batch):
    lr=tf.train.exponential_decay(0.0005, batch, 20000, 0.5, staircase=True)
    return lr

  def build_network(self):
    bn_decay=self.bn_decay
    training =self.is_training

    x0=self.im_in#(320,320,3)
    x1_1=conv2d_bn_relu(x0, 64, 3,3,1,1,'x1_1', bn_decay, training)#(320,320,64)
    x1_2=conv2d_bn_relu(x1_1, 64, 3,3,1,1,'x1_2', bn_decay, training)
    x1_3=max_pool(x1_2, 'x1_3')
    
    x2_1=conv2d_bn_relu(x1_3, 128, 3,3,1,1,'x2_1', bn_decay, training)#(160,160,128)
    x2_2=conv2d_bn_relu(x2_1, 128, 3,3,1,1,'x2_2', bn_decay, training)
    x2_3=max_pool(x2_2, 'x2_3')

    x3_1=conv2d_bn_relu(x2_3, 256, 3,3,1,1,'x3_1', bn_decay, training)#(80,80,256)
    x3_2=conv2d_bn_relu(x3_1, 256, 3,3,1,1,'x3_2', bn_decay, training)
    x3_3=conv2d_bn_relu(x3_2, 256, 3,3,1,1,'x3_3', bn_decay, training)
    x3_4=max_pool(x3_3, 'x3_4')
    x3_5=dropout(x3_4, training, 'x3_5')
    
    x4_1=conv2d_bn_relu(x3_5, 512, 3,3,1,1,'x4_1', bn_decay, training)#(40,40,512)
    x4_2=conv2d_bn_relu(x4_1, 512, 3,3,1,1,'x4_2', bn_decay, training)
    x4_3=conv2d_bn_relu(x4_2, 512, 3,3,1,1,'x4_3', bn_decay, training)
    x4_4=max_pool(x4_3, 'x4_4')
    x4_5=dropout(x4_4, training, 'x4_5')

    x5_1=conv2d_bn_relu(x4_5, 512, 3,3,1,1,'x5_1', bn_decay, training)#(20,20,512)
    x5_2=conv2d_bn_relu(x5_1, 512, 3,3,1,1,'x5_2', bn_decay, training)
    x5_3=conv2d_bn_relu(x5_2, 512, 3,3,1,1,'x5_3', bn_decay, training)
    x5_4=max_pool(x5_3, 'x5_4')#latent feature
    x5_5=dropout(x5_4, training, 'x5_5')
    #(10,10,512)
    #start upsampling 
    #in unpooling we don't use the indices
    x6_1=deconv2d_bn_relu(x5_5, [self.batch_size,20,20,512], 3, 3, 2, 2, 'x6_1', bn_decay, training)#(20,20,512)
    x6_2=conv2d_bn_relu(x6_1, 512, 3,3,1,1,'x6_2', bn_decay, training)
    x6_3=conv2d_bn_relu(x6_2, 512, 3,3,1,1,'x6_3', bn_decay, training)
    x6_4=conv2d_bn_relu(x6_3, 512, 3,3,1,1,'x6_4', bn_decay, training)
    x6_5=dropout(x6_4, training, 'x6_5')
    
    x7_1=deconv2d_bn_relu(x6_5, [self.batch_size,40,40,512], 3, 3, 2, 2, 'x7_1', bn_decay, training)#(40,40,512)
    x7_2=conv2d_bn_relu(x7_1, 512, 3,3,1,1,'x7_2', bn_decay, training)
    x7_3=conv2d_bn_relu(x7_2, 512, 3,3,1,1,'x7_3', bn_decay, training)
    x7_4=conv2d_bn_relu(x7_3, 256, 3,3,1,1,'x7_4', bn_decay, training)

    x8_1=conv2d_bn_relu(x7_4, 64, 1,1,1,1,'x8_1', bn_decay, training)#(40,40,64)
    x8_2=conv2d_bn_relu(x8_1, 48, 1,1,1,1,'x8_2', bn_decay, training)#(40,40,48)
    x8_3=tf.nn.sigmoid(x8_2)
    self.out_layout=x8_3

    #get class label
    y0=tf.contrib.layers.flatten(x5_4)
    y1=linear(y0, 1024, 'y1')
    y2=linear(y1, 512, 'y2')
    y3=linear(y2, 11, 'y3')
#    y4=tf.nn.softmax(y3)
    self.out_label=y3

  def get_layloss(self,gt_layout, out_layout):
    lay_loss=0.0
    for i in range(self.batch_size):
      lay_gt=gt_layout[i]
      lay_out=out_layout[i]
      label=tf.cast(tf.argmax(self.gt_label[i]), tf.int32)
      begin=self.l_list[label]
      end=self.l_list[label+1]
      lay1=tf.slice(lay_gt, [0,0,begin], [40,40,end-begin])
      lay2=tf.slice(lay_out, [0,0,begin], [40,40,end-begin])
      lay_loss+=0.5*euclidean_loss(lay1,lay2)

      temp2=tf.where(lay_gt>0.36)
      l2_1=tf.gather_nd(lay_gt, temp2)
      l2_2=tf.gather_nd(lay_out, temp2)
      temp3=tf.where(lay_gt>0.7)
      l3_1=tf.gather_nd(lay_gt, temp3)
      l3_2=tf.gather_nd(lay_out, temp3)
      lay_loss+=euclidean_loss2(l2_1,l2_2)
      lay_loss+=euclidean_loss2(l3_1,l3_2)
    return lay_loss

  def set_loss(self):
    self.weight_loss=tf.add_n(tf.get_collection('w_losses'), name='w_loss')
#    self.class_loss=-tf.reduce_mean(self.gt_label*tf.log(self.out_label+1e-5))
    self.class_loss=cross_entroy_loss(self.out_label, self.gt_label)
    self.lay_loss=self.get_layloss(self.gt_layout, self.out_layout)
    tf.summary.scalar('class_loss', self.class_loss)
    tf.summary.scalar('lay_loss', self.lay_loss)
    self.loss=self.lay_loss+2*self.class_loss+0.1*self.weight_loss
    tf.summary.scalar('loss', self.loss)
  
  def run_optim(self,sess):
    return sess.run(self.optim, feed_dict=self.feed_dict)
  def run_loss(self, sess):
    return sess.run([self.loss, self.class_loss,self.lay_loss],feed_dict=self.feed_dict)
  def run_result(self,sess):
    label, layout=sess.run([self.out_label, self.out_layout],feed_dict=self.feed_dict)
    return label,layout
  def run_sum(self,sess):
    return sess.run(self.merged, feed_dict=self.feed_dict)
  def save_model(self, sess, folder, it):
    self.saver.save(sess, os.path.join(folder, "model"+str(it)), global_step=it)
  def restore_model(self, sess, folder):
    print 'restore all variables', folder
    return load_snapshot(self.saver, sess, folder)
  def print_loss_acc(self, sess):
    output=sess.run([self.loss,self.class_loss, self.lay_loss], feed_dict=self.feed_dict)
    print ('[Loss: %.4f] [c_loss: %.3f] [l_loss: %.3f]' % (output[0], output[1],output[2]))
  def run_step(self,sess):
    step_=sess.run(self.global_step)
    return epo, step_
  def step_assign(self,sess, i):
    step_p=self.global_step.assign(i)
    sess.run(step_p)

  def build_model(self):
    self.global_step=tf.Variable(0, trainable=False)
#    self.set_lr()
#    self.learning_rate=0.0005
    self.learning_rate=self.get_lr(self.global_step)
    self.bn_decay=self.get_bn_decay(self.global_step)
    self.set_placeholder()
    self.build_network()
    self.set_loss()
    self.merged=tf.summary.merge_all()
    self.t_vars=tf.trainable_variables() 
    self.optim=tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9).minimize(self.loss, var_list=self.t_vars)
    #self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=self.t_vars)
    self.saver = tf.train.Saver(self.t_vars, max_to_keep=3)


class RcnnNet(RoomnetVanilla):
  def __init__(self):
    super(RcnnNet, self).__init__()
  def build_network(self):
    bn_decay=self.bn_decay
    training =self.is_training

    x0=self.im_in#(320,320,3)
    x1_1=conv2d_bn_relu(x0, 64, 3,3,1,1,'x1_1', bn_decay, training)#(320,320,64)
    x1_2=conv2d_bn_relu(x1_1, 64, 3,3,1,1,'x1_2', bn_decay, training)
    x1_3=max_pool(x1_2, 'x1_3')
    
    x2_1=conv2d_bn_relu(x1_3, 128, 3,3,1,1,'x2_1', bn_decay, training)#(160,160,128)
    x2_2=conv2d_bn_relu(x2_1, 128, 3,3,1,1,'x2_2', bn_decay, training)
    x2_3=max_pool(x2_2, 'x2_3')

    x3_1=conv2d_bn_relu(x2_3, 256, 3,3,1,1,'x3_1', bn_decay, training)#(80,80,256)
    x3_2=conv2d_bn_relu(x3_1, 256, 3,3,1,1,'x3_2', bn_decay, training)
#    x3_3=conv2d_bn_relu(x3_2, 256, 3,3,1,1,'x3_3', bn_decay, training)
    x3_4=max_pool(x3_2, 'x3_4')
    x3_5=dropout(x3_4, training, 'x3_5')
    
    #####begin center blocks
    x4_1=conv2d(x3_5,'x4_1',relu=True)
    x4_2=conv2d(x4_1,'x4_2',relu=True)
#    x4_3=conv2d(x4_2,'x4_3',relu=True)
    x4_4=max_pool(x4_2, 'x4_4')
    x4_5=dropout(x4_4, training, 'x4_5')

    x5_1=conv2d(x4_5,'x5_1',relu=True)
    x5_2=conv2d(x5_1,'x5_2',relu=True)
    x5_3=conv2d(x5_2,'x5_3',relu=True)
    x5_4=max_pool(x5_3, 'x5_4')
    x5_5=dropout(x5_4, training, 'x5_5')

    x6_1=deconv2d_bn_relu(x5_5, [self.batch_size,20,20,512], 3, 3, 2, 2, 'x6_1', bn_decay, training)#(20,20,512)
    x6_2=conv2d(x6_1,'x6_2',relu=True)
    x6_3=conv2d(x6_2,'x6_3',relu=True)
    x6_4=conv2d(x6_3,'x6_4',relu=True)
    x6_5=dropout(x6_4, training, 'x6_5')
   
    x7_1=deconv2d_bn_relu(x6_5, [self.batch_size,40,40,512], 3, 3, 2, 2, 'x7_1', bn_decay, training)#(40,40,512)
#    x7_2=conv2d(x7_1,'x7_2',relu=True)
    x7_3=conv2d(x7_1,'x7_3',relu=True)
    x7_4=conv2d(x7_3,'x7_4',relu=True)

##### y

    y4_1=tf.nn.relu(conv2d(x3_5, 'x4_1', reuse=True)+conv2d(x4_1, 'h4_1'))
    y4_2=tf.nn.relu(conv2d(y4_1, 'x4_2', reuse=True)+conv2d(x4_2, 'h4_2'))
#    y4_3=tf.nn.relu(conv2d(y4_2, 'x4_3', reuse=True)+conv2d(x4_3, 'h4_3'))
    y4_4=max_pool(y4_2, 'y4_4')
    y4_5=dropout(y4_4, training, 'y4_5')

    y5_1=tf.nn.relu(conv2d(y4_5, 'x5_1', reuse=True)+conv2d(x5_1, 'h5_1'))
    y5_2=tf.nn.relu(conv2d(y5_1, 'x5_2', reuse=True)+conv2d(x5_2, 'h5_2'))
    y5_3=tf.nn.relu(conv2d(y5_2, 'x5_3', reuse=True)+conv2d(x5_3, 'h5_3'))
    y5_4=max_pool(y5_3, 'y5_4')
    y5_5=dropout(y5_4, training, 'y5_5')

    y6_1=deconv2d_bn_relu(y5_5, [self.batch_size,20,20,512], 3, 3, 2, 2, 'x6_1', bn_decay, training,reuse=True)
    y6_2=tf.nn.relu(conv2d(y6_1, 'x6_2', reuse=True)+conv2d(x6_2, 'h6_2'))
    y6_3=tf.nn.relu(conv2d(y6_2, 'x6_3', reuse=True)+conv2d(x6_3, 'h6_3'))
    y6_4=tf.nn.relu(conv2d(y6_3, 'x6_4', reuse=True)+conv2d(x6_4, 'h6_4'))
    y6_5=dropout(y6_4, training, 'y6_5')

    y7_1=deconv2d_bn_relu(y6_5, [self.batch_size,40,40,512], 3, 3, 2, 2, 'x7_1', bn_decay, training,reuse=True)
#    y7_2=tf.nn.relu(conv2d(y7_1, 'x7_2', reuse=True)+conv2d(x7_2, 'h7_2'))
    y7_3=tf.nn.relu(conv2d(y7_1, 'x7_3', reuse=True)+conv2d(x7_3, 'h7_3'))
    y7_4=tf.nn.relu(conv2d(y7_3, 'x7_4', reuse=True)+conv2d(x7_4, 'h7_4'))

    z4_1=tf.nn.relu(conv2d(x3_5, 'x4_1', reuse=True)+conv2d(y4_1, 'h4_1',reuse=True))
    z4_2=tf.nn.relu(conv2d(z4_1, 'x4_2', reuse=True)+conv2d(y4_2, 'h4_2',reuse=True))
#    z4_3=tf.nn.relu(conv2d(z4_2, 'x4_3', reuse=True)+conv2d(y4_3, 'h4_3',reuse=True))
    z4_4=max_pool(z4_2, 'z4_4')
    z4_5=dropout(z4_4, training, 'z4_5')

    z5_1=tf.nn.relu(conv2d(z4_5, 'x5_1', reuse=True)+conv2d(y5_1, 'h5_1',reuse=True))
    z5_2=tf.nn.relu(conv2d(z5_1, 'x5_2', reuse=True)+conv2d(y5_2, 'h5_2',reuse=True))
    z5_3=tf.nn.relu(conv2d(z5_2, 'x5_3', reuse=True)+conv2d(y5_3, 'h5_3',reuse=True))
    z5_4=max_pool(z5_3, 'z5_4')
    z5_5=dropout(z5_4, training, 'z5_5')

    z6_1=deconv2d_bn_relu(z5_5, [self.batch_size,20,20,512], 3, 3, 2, 2, 'x6_1', bn_decay, training,reuse=True)
    z6_2=tf.nn.relu(conv2d(z6_1, 'x6_2', reuse=True)+conv2d(y6_2, 'h6_2',reuse=True))
    z6_3=tf.nn.relu(conv2d(z6_2, 'x6_3', reuse=True)+conv2d(y6_3, 'h6_3',reuse=True))
    z6_4=tf.nn.relu(conv2d(z6_3, 'x6_4', reuse=True)+conv2d(y6_4, 'h6_4',reuse=True))
    z6_5=dropout(z6_4, training, 'z6_5')

    z7_1=deconv2d_bn_relu(z6_5, [self.batch_size,40,40,512], 3, 3, 2, 2, 'x7_1', bn_decay, training, reuse=True)
#    z7_2=tf.nn.relu(conv2d(z7_1, 'x7_2', reuse=True)+conv2d(y7_2, 'h7_2',reuse=True))
    z7_3=tf.nn.relu(conv2d(z7_1, 'x7_3', reuse=True)+conv2d(y7_3, 'h7_3',reuse=True))
    z7_4=tf.nn.relu(conv2d(z7_3, 'x7_4', reuse=True)+conv2d(y7_4, 'h7_4',reuse=True))
#### edn center blocks
            
    x8_1=conv2d_bn_relu(x7_4, 256, 1,1,1,1,'x8_1', bn_decay, training)#(40,40,256)
    x8_2=conv2d_bn_relu(x8_1, 64, 1,1,1,1,'x8_2', bn_decay, training)#(40,40,64)
    x8_3=conv2d_bn_relu(x8_2, 48, 1,1,1,1,'x8_3', bn_decay, training)#(40,40,48)
    x8_4=tf.nn.sigmoid(x8_3)
    self.out_layout_x=x8_4

    #get class label
    x_y0=tf.contrib.layers.flatten(x5_4)
    x_y1=linear(x_y0, 1024, 'y1')
    x_y2=linear(x_y1, 512, 'y2')
    x_y3=linear(x_y2, 11, 'y3')
    self.out_label_x=x_y3

    y8_1=conv2d_bn_relu(y7_4, 256, 1,1,1,1,'x8_1', bn_decay, training, reuse=True)#(40,40,256)
    y8_2=conv2d_bn_relu(y8_1, 64, 1,1,1,1,'x8_2', bn_decay, training, reuse=True)#(40,40,64)
    y8_3=conv2d_bn_relu(y8_2, 48, 1,1,1,1,'x8_3', bn_decay, training, reuse=True)#(40,40,48)
    y8_4=tf.nn.sigmoid(y8_3)
    self.out_layout_y=y8_4

    #get class label
    y_y0=tf.contrib.layers.flatten(y5_4)
    y_y1=linear(y_y0, 1024, 'y1', reuse=True)
    y_y2=linear(y_y1, 512, 'y2', reuse=True)
    y_y3=linear(y_y2, 11, 'y3', reuse=True)
    self.out_label_y=y_y3

    z8_1=conv2d_bn_relu(z7_4, 256, 1,1,1,1,'x8_1', bn_decay, training, reuse=True)#(40,40,256)
    z8_2=conv2d_bn_relu(z8_1, 64, 1,1,1,1,'x8_2', bn_decay, training, reuse=True)#(40,40,64)
    z8_3=conv2d_bn_relu(z8_2, 48, 1,1,1,1,'x8_3', bn_decay, training, reuse=True)#(40,40,48)
    z8_4=tf.nn.sigmoid(z8_3)
    self.out_layout=z8_4

    #get class label
    z_y0=tf.contrib.layers.flatten(z5_4)
    z_y1=linear(z_y0, 1024, 'y1', reuse=True)
    z_y2=linear(z_y1, 512, 'y2', reuse=True)
    z_y3=linear(z_y2, 11, 'y3', reuse=True)
    self.out_label=z_y3

  def set_loss(self):
    self.weight_loss=tf.add_n(tf.get_collection('w_losses'), name='w_loss')
    self.class_loss=cross_entroy_loss(self.out_label, self.gt_label)
    self.class_loss+=0.5*cross_entroy_loss(self.out_label_x, self.gt_label)
    self.class_loss+=0.5*cross_entroy_loss(self.out_label_y, self.gt_label)

    self.lay_loss=self.get_layloss(self.gt_layout, self.out_layout)
    self.lay_loss+=0.5*self.get_layloss(self.gt_layout, self.out_layout_x)
    self.lay_loss+=0.5*self.get_layloss(self.gt_layout, self.out_layout_y)
    
    tf.summary.scalar('class_loss', self.class_loss)
    tf.summary.scalar('lay_loss', self.lay_loss)
    self.loss=5*self.lay_loss+2*self.class_loss+0.1*self.weight_loss
    tf.summary.scalar('loss', self.loss)


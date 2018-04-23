import tensorflow as tf
import numpy as np
from tf_utils import *
from net import *
import os
from BatchFetcher import *
import cv2
import shutil
import time
import argparse
from get_res import get_im
batch_size=20
s_in=320
s_out=40
max_epoch=225
l_list=[0,8,14,20,24,28,34,38,42,44,46, 48]

datapath='/home/mcg/Data/LSUN/data'
datadir='/home/mcg/Data/LSUN/data/training_data'
val_datadir='/home/mcg/Data/LSUN/data/validation_data'

    
def train(args):
#  config_path()
  outpath=args.out_path
  log_dir=os.path.join(outpath, 'logs')
  model_dir=os.path.join(outpath, 'model')
  sample_dir=os.path.join(outpath, 'sample')
  dirs=[log_dir, model_dir,sample_dir]
  for dir_ in dirs:
    if not os.path.exists(dir_):
      os.makedirs(dir_)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  sess=tf.Session(config=config)
  device='/gpu:0'
  if args.gpu==1:
    device='/gpu:1'
  with tf.device(device):
    if args.net=='vanilla':
      net=RoomnetVanilla()
    if args.net=='rcnn':
      net=RcnnNet()
    if args.net=='classify':
      net=ClassifyNet()
    net.build_model()
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  if args.train==0:
    print 'train from scratch'
    start_step=0
    # start_epoch=0
  else:
    start_step=net.restore_model(sess, model_dir)
   
  train_writer = tf.summary.FileWriter(log_dir,sess.graph)
  start_time=time.time()
  fetchworker=BatchFetcher(datadir,True, True)
  fetchworker.start()
  fetchworker2=BatchFetcher(val_datadir,False, True)
  fetchworker2.start()
  step_per_epoch=fetchworker.get_max_step()
  fout=open(os.path.join(outpath, 'acc.txt'), 'a')
  if 1:
  #for epo in range(start_epoch,max_epoch+1):
    for i in range(start_step, 225*step_per_epoch):
      im_in,lay_gt, label_gt,names=fetchworker.fetch()
      net.set_feed(im_in, lay_gt, label_gt,i)
      net.run_optim(sess)
      net.step_assign(sess,i)
      global_step=i      
      # net.step_plus(sess)
      # _,global_step=net.run_step()
      if np.mod(global_step,10)==0:
        summ_str = net.run_sum(sess)
        train_writer.add_summary(summ_str, global_step)        
        im_in,lay_gt, label_gt,names=fetchworker2.fetch()
        net.set_feed(im_in, lay_gt, label_gt,i)
        pred_class, pred_lay=net.run_result(sess)
        c_out=np.argmax(pred_class, axis=1)
        c_gt=np.argmax(label_gt, axis=1)
        acc=np.mean(np.array(np.equal(c_out, c_gt), np.float32)) 
        print 'accuracy',acc
        fout.write('%s %s\n'%(i, acc))
      if np.mod(global_step, 500)==0:
        net.save_model(sess, model_dir, global_step)
      if np.mod(global_step,500)==0:
        im_in,lay_gt, label_gt,names=fetchworker2.fetch()
        net.set_feed(im_in, lay_gt, label_gt,i)
        pred_class, pred_lay=net.run_result(sess)
#        try:
#          save_results(im_in, lay_gt, label_gt, names, pred_lay, pred_class, sample_dir, global_step)
#        except:
        np.savez(os.path.join(sample_dir, '%s.npz'%(i)), im=im_in, gt_lay=lay_gt, gt_label=label_gt, names=names, pred_lay=pred_lay, pred_class=pred_class)
      print('[step: %d] [time: %s]'%(i, time.time()-start_time))
      net.print_loss_acc(sess)
  fetchworker.shutdown()
  fetchworker2.shutdown()

def test(args):
  outdir=os.path.join(args.out_path, 'test')
  model_dir=os.path.join(args.out_path, 'model')
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  sess=tf.Session(config=config)
  device='/gpu:0'
  if args.gpu==1:
    device='/gpu:1'
  with tf.device(device):
    if args.net=='vanilla':
      net=RoomnetVanilla()
    if args.net=='rcnn':
      net=RcnnNet()
    net.build_model()
  start_step=net.restore_model(sess, model_dir)
  print 'restored'
  fout=open(os.path.join(outdir, 'acc.txt'), 'w')
  start_time=time.time()
  fetchworker=BatchFetcher(val_datadir,False, False)
  fetchworker.start()
  total_step=fetchworker.get_max_step()
  print 'total steps', total_step
  for i in range(total_step):
    im_in,lay_gt, label_gt,names=fetchworker.fetch()
    net.set_feed(im_in, lay_gt, label_gt,i)
    pred_class, pred_lay=net.run_result(sess)
    c_out=np.argmax(pred_class, axis=1)
    c_gt=np.argmax(label_gt, axis=1)
    acc=np.mean(np.array(np.equal(c_out, c_gt), np.float32))
    fout.write('%s %s\n'%(i, acc))
    for j in range(batch_size):
      img = im_in[j]
      # print class_label, label2
      outim = get_im(img, pred_lay[j], c_out, str(j))
      outim2 = get_im(img, lay_gt[j], c_gt, str(j))
      outpath=os.path.join(outdir, str(i))
      if not os.path.exists(outpath):
        os.makedirs(outpath)
      cv2.imwrite(os.path.join(outpath, '%s_gt_%s.jpg' % (names[j], class_label)), outim2)
      cv2.imwrite(os.path.join(outpath, '%s_pred_%s.jpg' % (names[j], label2)), outim)
      cv2.imwrite(os.path.join(outpath, '%s.jpg' % (names[j])), img * 255)
    print('[step: %d] [time: %s] [acc: %s]'%(i, time.time()-start_time, acc))
    net.print_loss_acc(sess)
  fetchworker.shutdown()
 

if __name__=='__main__':
  parser=argparse.ArgumentParser()
  parser.add_argument('--train', type=int, default=-1, help='train 0 or continue 1 ')
  parser.add_argument('--test', type=int, default=-1, help='0 for test')
  parser.add_argument('--net', type=str, default='vanilla', help='net type')
  parser.add_argument('--out_path', type=str, default='output', help='output path')
  parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
  args = parser.parse_args()
  if not args.train==-1:
    train(args)
  if not args.test==-1:
    test(args)  

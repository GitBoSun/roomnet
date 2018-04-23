import sys
import numpy as np
import cv2
import random
import math
import os
import time
import socket
import threading
import Queue
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import scipy.io
import helper
import scipy.io as sio
import scipy.misc as smc
BATCH_SIZE=20
s_in=320
s_out=40
n_type=11
max_epoch=225
# 0:499, 1:137, 2:2, 3:17, 4:744, 5:1352, 6:54, 7:2, 8:1, 9:160, 10:32
#balance_list=[1, 3, 200, 20, 1, 1, 8, 200, 200, 3, 15]
balance_list=[1,1,1,1,1,1,1,1,1,1,1,1]
train_mat='/home/mcg/Data/LSUN/data/training.mat'
val_mat='/home/mcg/Data/LSUN/data/validation.mat'

class BatchFetcher(threading.Thread):
  def __init__(self, datapath,istrain, repeat):
    super(BatchFetcher, self).__init__()
#    self.queue=Queue.Queue(40)
    if istrain:
      self.queue=Queue.Queue(16)
    else:
      self.queue=Queue.Queue(2)
    self.batch_size=BATCH_SIZE
    self.stopped=False
    self.datadir=datapath
    self.istrain=istrain
    self.repeat=repeat
    self.cur=0
    self.epoch=0
    if self.istrain:
      self.names=self.get_train_names()  
      self.data=sio.loadmat(train_mat)['training'][0]
    else:
      self.names=self.get_val_names()
      self.data=sio.loadmat(val_mat)['validation'][0]
    self.max_step=self.get_max_step()

  def get_train_names(self):
    names=[]
    data=sio.loadmat(train_mat)['training'][0]
    for item in data:
      name=item[0][0]
      out=np.load(os.path.join(self.datadir, name+'.npz'))
      label=np.argmax(out['label'])
      for cnt in range(balance_list[label]):
        names.append(name)
        names.append(name+'2')
    names=sorted(names)
    print 'total train  number', len(names)
    names=np.random.permutation(names)
    return names
  def get_val_names(self):
    names=[]
    data=sio.loadmat(val_mat)['validation'][0]
    for item in data:
      name=item[0][0]
      names.append(name)
    names=sorted(names)
    print 'val number', len(names)
#    names=np.random.permutation(names)
    return names
  def get_max_step(self):
    return int(len(self.names)/self.batch_size)

  def work(self):
    if self.cur+self.batch_size>=len(self.names):
      if self.repeat:
        self.cur=0
        self.epoch+=1
        self.names=np.random.permutation(self.names)
      else:
        self.shutdown()
        return None
    batch_ims=np.zeros((self.batch_size, s_in, s_in, 3))
    batch_layout=np.zeros((self.batch_size, s_out, s_out, 48))
    batch_labels=np.zeros((self.batch_size, 11))
    batch_names=[]
    for i in range(self.batch_size):
      name=self.names[self.cur+i]
      batch_names.append(name)
      out=np.load(os.path.join(self.datadir, name+'.npz'))
      im=out['im']
      im=np.array(im, dtype=np.float32)/255.0
      lay=out['lay']
      label=out['label']
      batch_ims[i]=im
      batch_layout[i]=lay
      batch_labels[i]=label
    return [batch_ims, batch_layout, batch_labels, batch_names]
  def run(self):
    if self.cur+self.batch_size>=len(self.names) and self.repeat==False:
      self.shutdown()
    while self.epoch<max_epoch+1 and not self.stopped:
      self.queue.put(self.work())
      print 'push'
      self.cur+=self.batch_size

  def fetch(self):
#    if self.stopped:
#      return None
    return self.queue.get()
  def shutdown(self):
    self.stopped=True
    while not self.queue.empty():
      self.queue.get()
if __name__=='__main__':
  datadir='/home/mcg/Data/LSUN/data/training_data'
  fetchworker=BatchFetcher(datadir, True)
  fetchworker.start()
#  time.sleep(10)
  for i in range(5):
    a,b,c,d=fetchworker.fetch()
    print a.shape, b.shape, c,d
  fetchworker.shutdown()


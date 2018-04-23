import os
import numpy as np
import zlib
import cPickle as pickle
import cv2
import scipy.io as sio
import scipy.misc as smc

im_path='/home/mcg/Data/LSUN/data/images'
mat='/home/mcg/Data/LSUN/data/training.mat'
#test_mat='/home/mcg/Data/LSUN/data/validation.mat'
outpath='/home/mcg/Data/LSUN/data/training_data'
#outpath='/home/mcg/Data/LSUN/data/validation_data'
s=320
out_s=40
l_list=[0,8,14,20,24,28,34,38,42,44,46]
flip_idx={}
flip_idx[0]=[7,8,5,6,3,4,1,2]
flip_idx[1]=[4,5,6,1,2,3]
flip_idx[2]=[4,5,6,1,2,3]
flip_idx[3]=[1,4,3,2]
flip_idx[4]=[1,4,3,2]
flip_idx[5]=[1,3,2,4,6,5]
flip_idx[6]=[2,1,4,3]
flip_idx[7]=[3,4,1,2]
flip_idx[8]=[2,1]
flip_idx[9]=[2,1]
flip_idx[10]=[1,2]

if not os.path.exists(outpath):
  os.makedirs(outpath)

def guassian_2d(x_mean, y_mean, dev=5.0):
  x, y = np.meshgrid(np.arange(out_s), np.arange(out_s))
  #z=(1.0/(2.0*np.pi*dev*dev))*np.exp(-((x-x_mean)**2+ (y-y_mean)**2)/(2.0*dev**2))
  z=np.exp(-((x-x_mean)**2+ (y-y_mean)**2)/(2.0*dev**2))
  return z

data=sio.loadmat(mat)
#data=data['validation'][0]
data=data['training'][0]
j=0
for item in data:
  j=j+1
#  if j==5:
#    break 
  if j%100==0:
    print j
  name=item[0][0]
  ltype=item[2][0][0]
  pts=item[3]
#  print name
#  print pts
  h,w=item[4][0]
  im=cv2.imread(os.path.join(im_path, name+'.jpg'))
#  print h,w
  im=cv2.resize(im, (s,s), interpolation = cv2.INTER_CUBIC)

  class_label=np.zeros(11)
  class_label[ltype]=1.0
  layout=np.zeros((out_s, out_s, 48))
  for i, pt in enumerate(pts):
    x_mean=int(pt[0]*(40.0/w))
    y_mean=int(pt[1]*(40.0/h))
#    print x_mean, y_mean    
    if x_mean==40:
      x_mean=39
    if y_mean==40:
      y_mean=39
    # x_mean=39-x_mean
    layout[:,:,l_list[ltype]+flip_idx[ltype][i]-1]=guassian_2d(x_mean, y_mean)
  np.savez(os.path.join(outpath, '%s.npz'%(name)), im=im, lay=layout, label=class_label)
#data augmentation: flip
  im = cv2.flip(im, 1)
  for i, pt in enumerate(pts):
    x_mean = int(pt[0] * (40.0 / w))
    y_mean = int(pt[1] * (40.0 / h))
    #    print x_mean, y_mean
    if x_mean == 40:
      x_mean = 39
    if y_mean == 40:
      y_mean = 39
    x_mean=39-x_mean
    layout[:, :, l_list[ltype] + flip_idx[ltype][i] - 1] = guassian_2d(x_mean, y_mean)
  np.savez(os.path.join(outpath, '%s2.npz' % (name)), im=im, lay=layout, label=class_label)

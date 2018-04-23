import numpy as np
import cv2
import os

out=np.load('rcnn/test/2.npz')
im=out['im']
gt_lay=out['gt_lay']
gt_label=out['gt_label']
names=out['names']
pred_lay=out['pred_lay']
pred_class=out['pred_class']
# outpath='lr5_4_noban_38000'
outpath='rcnn/test/02'
if not os.path.exists(outpath):
    os.makedirs(outpath)
l_list=[0,8,14,20,24,28,34,38,42,44,46,48]
colors=np.array([[255,0,0],[0,0,255],[0,255,0],[255,255,0],[255,0,255], [0,255,255],[139,126,102], [118,238,198]])

def guassian_2d(x_mean, y_mean, dev=2.0):
  x, y = np.meshgrid(np.arange(40), np.arange(40))
  #z=(1.0/(2.0*np.pi*dev*dev))*np.exp(-((x-x_mean)**2+ (y-y_mean)**2)/(2.0*dev**2))
  z=np.exp(-((x-x_mean)**2+ (y-y_mean)**2)/(2.0*dev**2))
  return z

def get_im(ims, layout,label,j):
    lay=layout[:,:,l_list[label]:l_list[label+1]]
    num=lay.shape[2]
    outim=np.zeros((40,40,3))
    pts=[]
    for i in range(num):
        position=np.where(lay[:,:,i]==np.max(lay[:,:,i]))
        x1=position[0][0]
        x2=position[1][0]
        pts.append([x2,x1])
        im2=guassian_2d(x2,x1).reshape(40,40,1)
        outim+=im2*colors[i]
    outim=cv2.resize(outim, (320,320))
#    res=cv2.addWeighted(ims*255, 0.7, outim, 0.5, 0)
    pt=np.array(pts, np.int32)*8
    pt=tuple(map(tuple, pt))
    outim=np.array(outim, np.uint8)
    l=3
#    for i  in range(num):
#      cv2.circle(outim, pts[i], 5, colors[i], -1)
    if label==0:
      cv2.line(outim, pt[1], pt[0], (255,0,0), l)
      cv2.line(outim, pt[0], pt[6], (255,0,0), l)
      cv2.line(outim, pt[6], pt[7], (255,0,0), l)
      cv2.line(outim, pt[0], pt[2], (255,0,0),l)
      cv2.line(outim, pt[6], pt[4], (255,0,0), l)
      cv2.line(outim, pt[2], pt[3], (255,0,0), l)
      cv2.line(outim, pt[2], pt[4], (255,0,0), l)
      cv2.line(outim, pt[4], pt[5], (255,0,0), l)
    if label==1:
      cv2.line(outim, pt[0], pt[3], (255,0,0), l)
      cv2.line(outim, pt[1], pt[4], (255,0,0), l)
      cv2.line(outim, pt[0], pt[1], (255,0,0), l)
      cv2.line(outim, pt[3], pt[4], (255,0,0), l)
      cv2.line(outim, pt[0], pt[2], (255,0,0), l)
      cv2.line(outim, pt[3], pt[5], (255,0,0), l)
    
    if label==2:
      cv2.line(outim, pt[0], pt[1], (255,0,0), l)
      cv2.line(outim, pt[3], pt[4], (255,0,0), l)
      cv2.line(outim, pt[0], pt[3], (255,0,0), l)
      cv2.line(outim, pt[0], pt[2], (255,0,0),l)
      cv2.line(outim, pt[3], pt[5], (255,0,0), l)
    if label==3 or label==4:
      cv2.line(outim, pt[0], pt[1], (255,0,0), l)
      cv2.line(outim, pt[0], pt[2], (255,0,0), l)
      cv2.line(outim, pt[0], pt[3], (255,0,0), l)
    if label==5:
      cv2.line(outim, pt[3], pt[5], (255,0,0), l)
      cv2.line(outim, pt[3], pt[4], (255,0,0), l)
      cv2.line(outim, pt[0], pt[1], (255,0,0), l)
      cv2.line(outim, pt[0], pt[2], (255,0,0), l)
      cv2.line(outim, pt[0], pt[3], (255,0,0), l)

    if label==6 or label==7:
      cv2.line(outim, pt[0], pt[1], (255,0,0), l)
      cv2.line(outim, pt[2], pt[3], (255,0,0), l)
    if label==8 or label==9 or label==10:
      cv2.line(outim, pt[0], pt[1], (255,0,0), l)
#    outim=np.array(outim, np.float32)
    outim=cv2.resize(outim, (320,320))
    ims=np.array(ims*255, np.uint8)
    res=cv2.addWeighted(ims, 0.5, outim, 0.5, 0) 

    return res

for i in range(20):
    img=im[i]
    class_label=np.argmax(gt_label[i])
    label2=np.argmax(pred_class[i])
    print class_label, label2
    # print i,'pred'
    outim= get_im(img, pred_lay[i], label2, str(i))
    outim2=get_im(img, gt_lay[i], class_label, str(i))  
    cv2.imwrite(os.path.join(outpath, '%s_gt_%s.jpg'%(i, class_label)), outim2)
    cv2.imwrite(os.path.join(outpath, '%s_pred_%s.jpg'%(i, label2)), outim)
    cv2.imwrite(os.path.join(outpath, '%s.jpg'%(i)), img*255)





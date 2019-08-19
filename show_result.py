import os
import sys
import numpy as np
from core.config import cfg
import matplotlib.pyplot as plt
import pdb
p=pdb.set_trace
import core.utils as utils
from tqdm import tqdm
import shutil

np.set_printoptions(suppress=True)
root_dir='/home/lijiong/Github/tensorflow-yolov3/'
save_dir=root_dir+'show_result/'

if os.path.isdir(save_dir) == True:
    shutil.rmtree(save_dir)
os.mkdir(save_dir)


annot_path= cfg.TEST.ANNOT_PATH 
img_dir='/home/lijiong/Github/PyTorch-YOLOv3/data/images/'+'val2017/'

def ReadTxt(rootdir):
    lines = []
    with open(rootdir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines

lines=ReadTxt(annot_path)
pbar =tqdm(lines,ncols=50)
for i, line in enumerate(pbar):
#for i, line in tqdm(enumerate(annotation_file)):
    if os.path.exists(root_dir+"mAP/predicted/%d.txt"%i):
        annotation = line.strip().split()
        image_path = annotation[0]
        image=plt.imread(image_path)


        bbox_gt = np.array([list(map(float, box.split(','))) for box in annotation[1:]]).astype(int)
        bbox_gt=[np.append(box,box[4]) for box in bbox_gt]
        bbox_gt=[box+[0,0,0,0,2,0] for box in bbox_gt]
            #box=np.append(box,box[4])
        #print(np.array(bbox_gt))
        image = utils.draw_bbox(image, bbox_gt, show_label=True,use_color=(100, 100, 100))
        plt.imsave(save_dir+'%d.jpg'%i,image)
        #print("----------------------------------------------------------------")
        bbox_pr=[]
        f=open(root_dir+"mAP/predicted/%d.txt"%i)
        for line in f:
            tmp=line.split()[2:]
            pr_score=float(line.split()[1])
            tmp=np.array(tmp).astype(int)
            tmp=np.append(tmp,pr_score)
            tmp=np.append(tmp,0)        #here need to be update if use more classes \
            bbox_pr.append(tmp)
        #print(np.array(bbox_pr))
        image = utils.draw_bbox(image, bbox_pr, show_label=True,use_color=0)
        plt.imsave(save_dir+'%d.jpg'%i,image)
        f.close()
print("finished")
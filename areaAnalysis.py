#把txt文件解析为pixel mask以及图片标注
import glob
import os
import cv2
import numpy as np
import Polygon as plg
import matplotlib.pyplot as plt
from collections import defaultdict

#txt_list=glob.glob('weinman19_GT_txt/*.txt')
txt_list=glob.glob('original_size_OS_USGS/[1-9]*.txt')

maxAreaAll=0
minAreaAll=float('inf')

for txt_path in txt_list:

    base_name = os.path.basename(txt_path)

    print('base_name:',base_name)

    if not os.path.exists(txt_path):
        continue

    with open(txt_path, 'r') as f:
        data = f.readlines()

    #bbox_idx = 0

    polyList=[]

    for line in data:

        polyStr=line.split(',')

        poly=[]

        for i in range(0,len(polyStr)):
            if i%2==0:
                poly.append([float(polyStr[i]),float(polyStr[i+1])])

        polyList.append(poly)

    print('all: ',len(polyList))

    maxArea=0

    minArea=float('inf')

    area={'0-500':0,'500-1000':0,'1000-2000':0,'2000-3000':0,'3000-5000':0,'5000-7000':0,'7000-10000':0,'>10000':0}

    for i in range(0,len(polyList)):
        poly=plg.Polygon(polyList[i])
        if poly.area()<=500:
            area['0-500']+=1
        elif 500<poly.area()<=1000:
            area['500-1000']+=1
        elif 1000<poly.area()<=2000:
            area['1000-2000']+=1
        elif 2000<poly.area()<=3000:
            area['2000-3000']+=1
        elif 3000<poly.area()<=5000:
            area['3000-5000']+=1
        elif 5000<poly.area()<=7000:
            area['5000-7000']+=1
        elif 7000<poly.area()<10000:
            area['7000-10000']+=1
        else:
            area['>10000']+=1

        maxArea=max(poly.area(),maxArea)
        minArea=min(poly.area(),minArea)

    print(os.path.basename(txt_path),' :')
    print('maxArea: ',maxArea,' minArea: ',minArea)

    maxAreaAll=max(maxAreaAll,maxArea)
    minAreaAll=min(minAreaAll,minArea)

    plt.figure(figsize=(20,15))
    plt.title(os.path.basename(txt_path))
    x=list(area.keys())
    y=list(area.values())
    plt.bar(range(len(x)), y, tick_label=x)
    plt.show()

print('maxAreaAll: ',maxAreaAll,' minAreaAll: ',minAreaAll)



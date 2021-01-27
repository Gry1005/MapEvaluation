import glob
import os
import cv2
import numpy as np

txt_list=glob.glob('whole_OS_USGS_txt_GT/*.txt')

for txt_path in txt_list:

    base_name = os.path.basename(txt_path)

    print('base_name:', base_name)

    # txt_path = 'original_size_OS_USGS/' + base_name[0:len(base_name) - 4] + '.txt'
    txt_path2 = 'original_size_OS_USGS/' + base_name[0:len(base_name) - 4] + '.txt'

    txt2 = open(txt_path2)

    for line in txt2.readlines():
        with open(txt_path, "a") as mon:
            mon.write(line)

    txt2.close()





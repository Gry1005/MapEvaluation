import shapefile
import glob
import cv2
import os
import numpy as np
#from shapely.geometry import Polygon


import re

shp_dir = 'E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/original_more_USGS/*'
img_dir = 'E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/original_size_OS_USGS_jpg/'

gt_subdirs = glob.glob(shp_dir)

for i in range(0, len(gt_subdirs)):

    if os.path.basename(gt_subdirs[i]) == 'USGS-15-CA-hesperia-e1902-s1898-rp1912':
        continue

    subd = gt_subdirs[i]
    print(subd + '/*_add.shp')

    # load shp file
    shape_file = glob.glob(subd + '/*_add.shp')[0]  # one subdir only has one shape file
    sf = shapefile.Reader(shape_file)
    shapes = sf.shapes()
    #recds = sf.records()

    output_dir = 'original_more_OS_USGS_txt_GT/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    f = open(output_dir + os.path.basename(gt_subdirs[i])[:-4] + '.txt', 'w')
    for shape in shapes:
        cur_point = shape.points
        cur_point = np.array(cur_point)

        cur_point = cur_point * np.array([1, -1])  # reverse y coords
        # cur_point = cur_point + np.array([half_padding_w ,half_padding_h])

        num_rows = cur_point.shape[0] - 1
        for r_id in range(0, num_rows - 1):  # X*2
            f.write(str(cur_point[r_id][0]) + ',' + str(cur_point[r_id][1]) + ',')
        f.write(str(cur_point[num_rows - 1][0]) + ',' + str(cur_point[num_rows - 1][1]) + '\n')

        # append text
        '''
        if 'TXT' not in dir(recd)[0:5]:
            print dir(recd)
        else:
            print recd['TXT']
        break
        '''
    f.close()
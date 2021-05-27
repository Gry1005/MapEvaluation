#基于wolf2006的evaluation方法

import glob
import os
import cv2
import numpy as np
import Polygon as plg
import matplotlib.pyplot as plt
from collections import Counter
import csv


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()

def get_union(pD,pG):
    pInt = pD | pG
    if len(pInt) == 0:
        return 0
    return pInt.area()

#predict_txt_list=glob.glob('original_USGS_txt_04/USGS-15-CA-brawley-e1957-s1957-p1961.txt')
#predict_txt_list=glob.glob('original_USGS_txt_04/USGS-15-CA-capesanmartin-e1921-s1917.txt')

#predict_txt_list=glob.glob('D:/textDetect_evalSet/east_map_res_maplevel/*.txt')
#predict_txt_list=glob.glob('D:/textDetect_evalSet/psenet_map_res_maplevel/*.txt')
predict_txt_list=glob.glob('D:/textDetect_evalSet/synthText_model1.0_wholemap_OS/*.txt')
#predict_txt_list=glob.glob('D:/textDetect_evalSet/results_txt/synthtext_wce_w1_modelv4_vgg_textprob_centerline/usgs/txt/*.txt')

image_folder_path='E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/original_size_OS_USGS_jpg/'
#image_folder_path='E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/weinman19-maps/'

GT_folder_path='../original_size_OS_USGS/'

more_GT_folder_path='../original_more_OS_USGS_txt_GT/'

#GT_folder_path='weinman19_GT_txt/'

output_path='../testOutput/'
#output_path='weinman_eval/'

#用csv文件保存结果
csv_dir='../csvOutput/'
csvfile = open(csv_dir+'synthText_model1.0_wholemap_OS_tr0.5_tp0.5_k1.csv','w',newline='')
writeCSV=csv.writer(csvfile)
writeCSV.writerow(['mapName','recall','precision','f1'])


for txt_path in predict_txt_list:

    # 解析predict的txt

    base_name = os.path.basename(txt_path)

    if base_name=='USGS-15-CA-hesperia-e1902-s1898-rp1912.txt':
        continue

    print('base_name:', base_name)

    image_path = image_folder_path + base_name[0:len(base_name) - 4] + '.jpg'
    #image_path = image_folder_path + base_name[0:len(base_name) - 4] + '.tiff'

    image=cv2.imread(image_path)

    image_2 = image.copy()

    #image_1=image.copy()

    with open(txt_path, 'r') as f:
        data = f.readlines()

    predict_polyList = []

    for line in data:

        polyStr = line.split(',')

        poly = []

        polyRange=0

        #GoogleVision和weinman的output最后会有文字的结果
        for i in range(0, len(polyStr)):
        #for i in range(0, 7):
            if i % 2 == 0:
                poly.append([float(polyStr[i]), float(polyStr[i + 1])])

        predict_polyList.append(poly)

    threshold = 0

    #去除过小的box
    predict_polyList=list(filter(lambda x:plg.Polygon(x).area()>threshold, predict_polyList))

    print('prediction all poly: ', len(predict_polyList))

    for i in range(0, len(predict_polyList)):

        polyPoints = np.array([predict_polyList[i]], dtype=np.int32)

        cv2.polylines(image, polyPoints, True, (0, 0, 255), 1)

    #cv2.imshow('prediction result',image_1)

    #cv2.waitKey()


    # 解析p==GT的txt

    GT_txt_path=GT_folder_path+base_name

    GT_polyList=[]

    with open(GT_txt_path, 'r') as f:
        GT_data = f.readlines()

    for line in GT_data:

        polyStr = line.split(',')

        #处理###，正常情况下不需要!!!!
        #polyStr=polyStr[:-1]

        poly = []

        for i in range(0, len(polyStr)):
            if i % 2 == 0:
                poly.append([float(polyStr[i]), float(polyStr[i + 1])])

        GT_polyList.append(poly)


    #去除area<threshold的部分

    GT_polyList = list(filter(lambda x: plg.Polygon(x).area() > threshold, GT_polyList))

    print('GT all poly: ', len(GT_polyList))

    for i in range(0, len(GT_polyList)):

        polyPoints = np.array([GT_polyList[i]], dtype=np.int32)

        cv2.polylines(image, polyPoints, True, (255, 0, 0), 1)

    # 解析Do Not Care的GT txt
    more_txt_path = more_GT_folder_path + base_name

    if not os.path.exists(more_txt_path):
        continue

    more_GT_polyList = []

    with open(more_txt_path, 'r') as f:
        more_GT_data = f.readlines()

    for line in more_GT_data:

        polyStr = line.split(',')

        poly = []

        for i in range(0, len(polyStr)):
            if i % 2 == 0:
                poly.append([float(polyStr[i]), float(polyStr[i + 1])])

        more_GT_polyList.append(poly)

    #去除面积为0的部分
    more_GT_polyList = list(filter(lambda x: plg.Polygon(x).area() > threshold, more_GT_polyList))

    print('more GT all poly: ', len(more_GT_polyList))

    for i in range(0, len(more_GT_polyList)):
        polyPoints = np.array([more_GT_polyList[i]], dtype=np.int32)

        cv2.polylines(image, polyPoints, True, (0, 255, 0), 1)

    #cv2.imwrite(output_path+'GT_predict_'+base_name[0:len(base_name) - 4] + '.jpg',image)

    #算出GT box的合并
    all_GT_polyList=GT_polyList+more_GT_polyList

    #cv2.waitKey()

    metrics_recall=[[0 for _ in range(0,len(predict_polyList))] for _ in range(0,len(all_GT_polyList))]

    metrics_precision=[[0 for _ in range(0,len(predict_polyList))] for _ in range(0,len(all_GT_polyList))]

    for i in range(0,len(all_GT_polyList)):
        for j in range(0,len(predict_polyList)):

            poly1=plg.Polygon(all_GT_polyList[i])
            poly2=plg.Polygon(predict_polyList[j])

            #print('GT area:', poly1.area())
            #print('Predict area:', poly2.area())

            inter_area=get_intersection(poly1,poly2)

            #print('inter_area:',inter_area)

            if poly1.area==0 or poly2.area==0 or inter_area==0:
                metrics_precision[i][j]=0
                metrics_recall[i][j]=0
            else:
                metrics_precision[i][j]=inter_area / poly2.area()
                metrics_recall[i][j]=inter_area / poly1.area()

            #print('area precision:',metrics_precision[i][j])
            #print('area recall:', metrics_recall[i][j])


    '''
    #按照icdar2017的思路去计算每张地图的recall和precision

    flag_GT_icdar = [0 for _ in range(0, len(GT_polyList))]
    flag_predict_icdar = [0 for _ in range(0, len(predict_polyList))]

    #recall
    #有多少GT box被正确预测

    for i in range(0,len(GT_polyList)):
        for j in range(0,len(predict_polyList)):
            poly1=plg.Polygon(GT_polyList[i])
            poly2=plg.Polygon(predict_polyList[j])
            if get_intersection(poly1,poly2)/get_union(poly1,poly2)>=0.5:
                flag_GT_icdar[i]=1
                flag_predict_icdar[j]=1

    recall_2017=Counter(flag_GT_icdar)[1]/len(GT_polyList)

    #precision
    precision_2017=Counter(flag_predict_icdar)[1]/len(predict_polyList)

    print('recall_icdar2017:%.3f'%recall_2017)
    print('precision_icdar2017:%.3f'%precision_2017)
    '''


    #wolf2006, 计算polygonlevel的recall和precision

    #记录那些box没有被正确预测
    flag_GT=[0 for _ in range(0,len(GT_polyList))]
    flag_predict=[0 for _ in range(0,len(predict_polyList))]

    #可变的阈值

    #tr = 0.5
    #tp = 0.4

    #对于one-to-many情况下的惩罚系数
    k=1

    recallmx=[]
    precisionmx=[]

    for tr in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:

        recalllist=[]
        precisionlist=[]

        for tp in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:

            #把标记数组重置为0
            flag_GT = [0 for _ in range(0, len(GT_polyList))]
            flag_predict = [0 for _ in range(0, len(predict_polyList))]

            # 计算recall

            # 计算GT中有多少box被正确预测了
            #GT_count = 0

            for i in range(0, len(all_GT_polyList)):

                # 计算在候选的predict_box中，多少个可以被认为与当前的GT box ovelap
                overlap_predict_boxlist = []

                for j in range(0, len(predict_polyList)):
                    if metrics_precision[i][j] >= tp:
                        overlap_predict_boxlist.append(j)

                if len(overlap_predict_boxlist) == 0:
                    continue
                elif len(overlap_predict_boxlist) == 1:
                    # one-to-one
                    if metrics_recall[i][overlap_predict_boxlist[0]] >= tr:
                        #GT_count = GT_count + 1
                        if i<len(flag_GT):
                            flag_GT[i]=1
                        flag_predict[overlap_predict_boxlist[0]]=1
                else:
                    # one-to-many
                    rs = 0
                    for num in overlap_predict_boxlist:
                        rs = rs + metrics_recall[i][num]

                    if rs >= tr:
                        #GT_count = GT_count + k
                        if i<len(flag_GT):
                            flag_GT[i]=1
                        #predict boxes也被正确预测了
                        for num in overlap_predict_boxlist:
                            flag_predict[num]=1

            # 把GT box中area为0的去掉

            GT_all = 0

            for i in range(0, len(GT_polyList)):
                GTpoly = plg.Polygon(GT_polyList[i])
                if GTpoly.area() > 0:
                    GT_all += 1


            # 计算precision

            #predict_count = 0

            for j in range(0, len(predict_polyList)):

                # 与当前的predict box overlap的GT box有哪些
                overlap_GT_boxlist = []

                for i in range(0, len(all_GT_polyList)):
                    if metrics_recall[i][j] >= tr:
                        overlap_GT_boxlist.append(i)

                if len(overlap_GT_boxlist) == 0:
                    continue
                elif len(overlap_GT_boxlist) == 1:
                    # one-to-one
                    if metrics_precision[overlap_GT_boxlist[0]][j] >= tp:
                        #predict_count += 1
                        flag_predict[j]=1
                        if overlap_GT_boxlist[0]<len(flag_GT):
                            flag_GT[overlap_GT_boxlist[0]]=1
                else:
                    # one-to-many
                    ps = 0
                    for num in overlap_GT_boxlist:
                        ps = ps + metrics_precision[num][j]

                    if ps >= tp:
                        #predict_count = predict_count + k
                        flag_predict[j]=1

                        for num in overlap_GT_boxlist:
                            if num<len(flag_GT):
                                flag_GT[num]=1

            # 把predict box中area为0的去掉

            predict_all = 0

            for i in range(0, len(predict_polyList)):
                predictpoly = plg.Polygon(predict_polyList[i])
                if predictpoly.area() > 0:
                    predict_all += 1


            #计算最终结果
            overall_recall = Counter(flag_GT)[1] / GT_all
            overall_precision = Counter(flag_predict)[1] / predict_all

            overall_recall=round(overall_recall,3)
            overall_precision=round(overall_precision,3)

            recalllist.append(overall_recall)
            precisionlist.append(overall_precision)

            if tr==0.5 and tp==0.5:
                if overall_recall + overall_precision!=0:
                    f1_score = round(2 * (overall_recall * overall_precision / (overall_recall + overall_precision)), 3)
                else:
                    f1_score=0.0
                print('tr:',tr,' tp:',tp,' k:',k,' recall:',overall_recall,' precision:',overall_precision, ' f1:',f1_score)
                writeCSV.writerow([base_name,overall_recall,overall_precision,f1_score])

            #标记未正确识别的box

            if tr==0.5 and tp==0.5:
                for i in range(0, len(predict_polyList)):
                    if flag_predict[i]==0:
                        polyPoints = np.array([predict_polyList[i]], dtype=np.int32)
                        cv2.polylines(image_2, polyPoints, True, (0, 0, 255), 1)
                for i in range(0, len(GT_polyList)):
                    if flag_GT[i]==0:
                        polyPoints = np.array([GT_polyList[i]], dtype=np.int32)
                        cv2.polylines(image_2, polyPoints, True, (255, 0, 0), 1)

                #cv2.imwrite(output_path+'Fail_GT_predict_' + base_name[0:len(base_name) - 4] + '.jpg', image_2)



        recallmx.append(recalllist)
        precisionmx.append(precisionlist)



    col_labels = ['tp=0.1', '0.2', '0.3','0.4','0.5','0.6','0.7','0.8','0.9']

    row_labels = ['tr=0.1', '0.2', '0.3','0.4','0.5','0.6','0.7','0.8','0.9']

    table_vals_1 = recallmx

    table_vals_2 = precisionmx

    # 第一行第一列图形
    #ax1 = plt.subplot(1, 2, 1)
    # 第一行第二列图形
    #ax2 = plt.subplot(1, 2, 2)

    plt.figure(1)

    my_table_1 = plt.table(cellText=table_vals_1, colWidths=[0.111] * 10, rowLabels=row_labels, colLabels=col_labels,loc='best')

    #plt.sca(ax1)

    plt.axis('off')

    plt.title('recall')

    plt.plot()

    plt.show()

    plt.figure(2)

    my_table_2 = plt.table(cellText=table_vals_2, colWidths=[0.111] * 10, rowLabels=row_labels, colLabels=col_labels,loc='best')

    #plt.sca(ax2)

    plt.axis('off')

    plt.title('precision')

    plt.plot()

    plt.show()


csvfile.close()








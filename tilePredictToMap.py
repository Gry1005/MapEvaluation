import glob
import os
import cv2
import numpy as np

txt_list=glob.glob('D:/textDetect_evalSet/east_map_res/*.txt')

outputPath='D:/textDetect_evalSet/east_res_originalLevel/'

#得到所有地图的名字
mapList=set()

for txt_path in txt_list:
    mapList.add(os.path.basename(txt_path).split('_')[0])

print(mapList)

for mapName in mapList:

    print(outputPath+mapName+'.txt')
    outputFile=open(outputPath+mapName+'.txt','w')


    for txt_path in txt_list:
        if os.path.basename(txt_path).split('_')[0]==mapName:

            hStr=os.path.basename(txt_path).split('_')[1].split('w')[0][1:]
            wStr=os.path.basename(txt_path).split('_')[1].split('w')[1][:-4]

            print(os.path.basename(txt_path))
            print('hStr: ',hStr,' wStr: ',wStr)

            h=int(hStr)
            w=int(wStr)

            with open(txt_path, 'r') as f:
                data = f.readlines()

                # bbox_idx = 0

            polyList = []

            for line in data:

                polyStr = line.split(',')

                resStr = ""

                for i in range(0, len(polyStr)):
                    if i % 2 == 0:
                        if i+1!=len(polyStr)-2:
                            resStr=resStr+str(float(polyStr[i])+w*1000)+','+str(float(polyStr[i + 1])+h*1000)+','
                        elif i+1==len(polyStr)-2:
                            resStr = resStr + str(float(polyStr[i])+w*1000) + ',' + str(float(polyStr[i + 1])+h*1000)
                            break

                resStr=resStr+'\n'

                outputFile.write(resStr)

    outputFile.close()




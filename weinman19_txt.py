#解析weinman19的json文件，变为bounding box

import json
import glob
import os

json_list=glob.glob('E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/weinman19-maps/JSON/maps/*.json')

result_txt_path='weinman19_GT_txt/'

for json_path in json_list:

    base_name=os.path.basename(json_path)

    print(base_name)

    f = open(result_txt_path+base_name[:-5]+'.txt', 'w+')

    data= json.load(open(json_path,encoding='utf-8'))

    #print(data[0])
    #print(data[0]["items"][0]["points"][0])

    for i in range(0,len(data)):

        items=data[i]["items"]

        for j in range(0,len(items)):

            #每个pointList对应一行
            pointList=items[j]["points"]
            #print('pointList: ',pointList)

            content=""

            for k in range(0,len(pointList)):
                if k<len(pointList)-1:
                    content=content+str(pointList[k][0])+','+str(pointList[k][1])+','
                else:
                    content =content+ str(pointList[k][0]) + ',' + str(pointList[k][1])

            content=content+'\n'

            f.writelines(content)

    f.close()






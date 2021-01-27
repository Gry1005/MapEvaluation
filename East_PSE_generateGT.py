import glob
import os

#解决east和psenet，数据格式不同的问题

#input_txt_list=glob.glob('D:/textDetect_evalSet/text_detection_weiman/EAST/*.txt')
input_txt_list=glob.glob('D:/textDetect_evalSet/text_detection_weiman/submit_iPSENET/*.txt')

output_dir='D:/textDetect_evalSet/psenet_weinman19_maplevel/'

for txt_path in input_txt_list:

    base_name = os.path.basename(txt_path)

    print('base_name:', base_name)

    with open(txt_path, 'r') as f:
        data = f.readlines()

    outputFile=open(output_dir+base_name[4:],'w')

    for line in data:

        line=line.split(',')[0:-1]

        line=','.join(line)

        line=line+'\n'

        outputFile.writelines(line)

    outputFile.close()


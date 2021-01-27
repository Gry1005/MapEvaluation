from PIL import Image
import glob
import os

image_list=glob.glob('E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/weinman19-maps/*.tiff')

save_dir='E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/weinman19-maps-jpg/'

for image_path in image_list:

    im=Image.open(image_path)

    print(os.path.basename(image_path))

    im.save(save_dir+os.path.basename(image_path)[:-5]+'.jpg')
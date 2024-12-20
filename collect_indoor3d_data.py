import os
import sys
import re
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
# DATA_PATH = os.path.join(ROOT_DIR, 'data/Stanford3dDataset_v1.2_Aligned_Version')
DATA_PATH = os.path.join(ROOT_DIR,'生成npy','待转成xoz_npy','test')

import indoor3d_util

# anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths_facade_test.txt'))]
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]#data/Facade_raw_data/Area_1/facade_1/Annotations

# output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d') 
output_folder = os.path.join(ROOT_DIR,'生成npy', 'xoz_test')#Facade_as_S3DIS_train_NPY改为New_Facade_as_S3DIS_train_sample_NPY
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# revise_file = os.path.join(DATA_PATH, "Area_5/hallway_6/Annotations/ceiling_1.txt")
# with open(revise_file, "r") as f:
#     data = f.read()
#     data = data[:5545347] + ' ' + data[5545348:]
#     f.close()
# with open(revise_file, "w") as f:
#     f.write(data)
#     f.close()

for anno_path in anno_paths:
	print(anno_path)
	elements = anno_path.split('/')
	elements = re.split('[/\\\\]',anno_path)
	out_filename = elements[-2]+'_'+elements[-1]+'.npy' # Area_1_hallway_1.npy
	# out_filename = anno_path + '.npy'
	indoor3d_util.collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')

# for anno_path in anno_paths:
#   print(anno_path)+
#   try:
#     elements = re.split('[/\\\\]',anno_path)
#     # elements = anno_path.split('/')
#     out_filename = elements[-3]+'_'+elements[-2]+'.npy'
#     indoor3d_util.collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
#     # print(elements)
#     # print(out_filename)
#     # print(os.path.join(output_folder, out_filename))
#   except:
#     print(anno_path, 'ERROR!!')

import os
from fnmatch import fnmatch


f=open("Bottomwear_paper.txt","a+")

cloth_list=[]

cloth_name_list

#for path, subdirs, files in os.walk("/Users/mi0307/Downloads/Sizing-Warehouse_cropped/"):
for path, subdirs, files in os.walk("/Users/mi0307/Downloads/Sizing-Warehouse_cropped/"):
    for name in files:
        if fnmatch(name, '*.jpg'):
            #print os.path.join(path, name)
            cloth_list.append(os.path.join(path, name))
            f.write(str(os.path.join(path, name))+'\n')

print(cloth_list)
f.close()

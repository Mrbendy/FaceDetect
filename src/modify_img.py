import os
from PIL import Image
filepath = 'data/images/mdr/'
dir = os.listdir(filepath)
print(dir)
for image_name in dir:
    im = Image.open(filepath+image_name)
    im = im.resize((250,250),Image.ANTIALIAS)
    im.save(filepath+image_name)

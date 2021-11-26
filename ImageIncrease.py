from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
import os, glob, numpy as np

imgpath = './ap' # 처리 전 이미지들이 존재하는 폴더
savepath = './atopi' # 처리 후 이미지들 저장 폴더

filenames = [] # 파일 이름들이 저장되는 곳

files = glob.glob(imgpath+"/*.*")
for i, f in enumerate(files): # 폴더 속 모든 파일의 이름 가져오기
    img = Image.open(f)
    data = np.asarray(img)
    filenames.append(f)
    #print(str(filenames))

data_aug_gen = ImageDataGenerator(rotation_range=10,
                                  horizontal_flip=True,
                                  vertical_flip=True,)

for i in filenames: # 파일 전부 로드
    img = load_img(i)
    print(str(img))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    
    i = 0   
    for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir=savepath, save_format='jpg'): # 사진 부풀리기
        i += 1
        if i > 30: 
            break

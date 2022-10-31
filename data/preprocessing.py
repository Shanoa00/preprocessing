#Preprocessing all images in kuzushiji dataset
import numpy as np
import cv2
from matplotlib import pyplot as plt 
import os
from skimage.filters.rank import enhance_contrast
from skimage.morphology import disk, ball

plt.rcParams['figure.dpi'] = 250

# All dataset
path='/home/mauricio/Documents/Pytorch/kaggle-kuzushiji-2019/data/'
folder=  ['train_images1'] #['train_images1', 'test_images1'] 

#Just small part of dataset
#path= '/home/mauricio/Documents/Pytorch/Pre-processing_model/data/'
#folder= ['train_ori', 'test_ori'] 

def adapt_binarize(image_file, with_plot=False, gray_scale=True):
    image= image_file
    #gray= cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    if gray_scale is not True:
        image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #th, image_b = cv2.threshold(src=image, thresh=thresh_val, maxval=255, type=cv2.THRESH_BINARY)
    #image_b= cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 231, 51) #ADAPTIVE_THRESH_GAUSSIAN_C
    #image= cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    image_b= cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 231, 51) #ADAPTIVE_THRESH_GAUSSIAN_C
    if with_plot:
        cmap_val = None if not gray_scale else 'gray'
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
        
        ax1.axis("off")
        ax1.title.set_text('Original')
        
        ax2.axis("off")
        ax2.title.set_text("Binarized")
        
        ax1.imshow(image, cmap=cmap_val)
        ax2.imshow(image_b, cmap=cmap_val)
        #return True
    return image_b

def filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours and remove small noise
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #print(cnts)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20:
            cv2.drawContours(opening, [c], -1, 0, -1)

    # Invert and apply slight Gaussian blur
    return 255 - opening
    #result = cv2.GaussianBlur(result, (3,3), 0)

def binarize(image_file, thresh_val=127, tipo=cv2.THRESH_BINARY ,with_plot=False, gray_scale=True):
    image= image_file
    if gray_scale is not True:
        image= cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    th, image_b = cv2.threshold(src=image, thresh=thresh_val, maxval=255, type=tipo)
    #image_b= cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 221, 31)
    if with_plot:
        cmap_val = None if not gray_scale else 'gray'
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
        
        ax1.axis("off")
        ax1.title.set_text('Original')
        
        ax2.axis("off")
        ax2.title.set_text("threshold")
        
        ax1.imshow(image, cmap=cmap_val)
        ax2.imshow(image_b, cmap=cmap_val)
        #return True
    return image_b

def filtering(image, filter_size=2):
    # Morph open to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (filter_size,filter_size))
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours and remove small noise
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #print(cnts)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 40:
            cv2.drawContours(opening, [c], -1, 0, -1)

    # Invert and apply slight Gaussian blur
    return opening

def morphologic_filter(image, kernel):
    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enh = enhance_contrast(gray, disk(10))
    erosion = cv2.erode(enh,kernel,iterations = 1)
    bin22= binarize(erosion, thresh_val=115, tipo=cv2.THRESH_TOZERO) #THRESH_TOZERO #130
    #bin_dil= binarize(dilation, thresh_val=120, tipo=cv2.THRESH_BINARY+cv2.THRESH_OTSU) #THRESH_TOZERO #130
    bin33= adapt_binarize(bin22, gray_scale=True)
    return bin33

def morphologic2(image, kernel):
    normalizedImg = np.zeros(image.shape)
    norm= cv2.normalize(image, normalizedImg, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)
    gray= cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    #dst = cv2.fastNlMeansDenoisingColored(enh, None, 10, 15, 7, 8) #  10, 15, 7, 15-> 8
    #gray= cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)
    enh = enhance_contrast(gray, disk(10))
    #kernel = np.ones((4,4),np.uint8)
    #erosion = cv2.erode(enh,kernel,iterations = 1)
    bin22= binarize(enh, thresh_val=115, tipo=cv2.THRESH_TOZERO) #THRESH_TOZERO #130
    #bin_dil= binarize(dilation, thresh_val=120, tipo=cv2.THRESH_BINARY+cv2.THRESH_OTSU) #THRESH_TOZERO #130
    bin33= adapt_binarize(bin22, gray_scale=True)
    return bin33

    
def preprocess(folder, output):
    kernel = np.ones((4,4),np.uint8)
    color_threshold= 35
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            #get avg color
            avg_color_per_row = np.average(img, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            diff= avg_color[2]-avg_color[0]
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if diff >=  color_threshold:
                #Adaptive filter
                #en data as test_adaptive
                normalizedImg = np.zeros(img.shape)
                img= cv2.normalize(img, normalizedImg, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)
                dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
                erosion = cv2.erode(dst,kernel,iterations = 1)
                binarized= adapt_binarize(erosion, gray_scale=False)  
                binarized= cv2.dilate(binarized,kernel,iterations = 1)
                
            #     #threshold filter
            #     dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 15, 7, 8) #  10, 15, 7, 15-> 8
            #     erosion = cv2.erode(dst,kernel,iterations = 1)
            #     bin2= binarize(erosion, thresh_val=115, tipo=cv2.THRESH_TOZERO) #THRESH_TOZERO #130
            #     #bin_dil= binarize(dilation, thresh_val=120, tipo=cv2.THRESH_BINARY+cv2.THRESH_OTSU) #THRESH_TOZERO #130
            #     bin3= adapt_binarize(bin2, gray_scale=True)
            #     result= cv2.dilate(bin3, kernel, iterations = 1)
            #     binarized= filtering(result)
                
            #     cv2.imwrite(output+ str(filename), binarized)
            #     #print("color: ",filename)
            else:
                #binarized= binarize(img, thresh_val=127)
                binarized= morphologic2(img, kernel) 
            cv2.imwrite(output+ str(filename), binarized)
            #print("b/w ",filename)
            
            
            
for loc in folder:
    print(loc.split("_")[0]+'/')
    preprocess(path+loc, loc.split("_")[0]+'/')
    print("DONE!")
    
    
    #### 200022050-00014_2, 200022050-00004_2, 200021712-00062_1
    
    #todo: aplicar modo4 y grabar imagenes 
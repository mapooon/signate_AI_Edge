import cv2
import numpy  as np

class Clahe(object):
    def __init__(self,clipLimit=4,tileGridSize=(8,8),mode="rgb"):
        self.clipLimit=clipLimit
        self.tileGridSize=tileGridSize
        self.clahe=cv2.createCLAHE(clipLimit=self.clipLimit,tileGridSize=self.tileGridSize)
        self.mode=mode

    def __call__(self,img):
        if self.mode=="rgb":
            transformed=self.RGB_CLAHE(img)
        else:
            transformed=self.GrayScale_CLAHE(img)
        return transformed

    def RGB_CLAHE(self,img):
        lab=cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = self.clahe.apply(lab[:,:,0])
        output = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return output


    def GrayScale_CLAHE(self,img):
        #img=np.array(img).astype("uint8")
        img=self.clahe.apply(img)
        #img[:,:,1]=self.clahe.apply(img[:,:,1])
        #img=img.reshape((width,width,2))
        return img#Image.fromarray(img)

class Resize(object):
    def __init__(self,size):
        self.size=size[::-1]
    def __call__(self,img):
        return cv2.resize(img,self.size,interpolation=cv2.INTER_NEAREST)


class RandomCrop(object):
    def __init__(self,output_shape,up_cut=0,down_cut=0,right_cut=0,left_cut=0):
        self.height,self.width=tuple(output_shape)
        self.up_cut,self.down_cut,self.right_cut,self.left_cut=up_cut,down_cut,right_cut,left_cut
    
    # def __call__(self,img,target):
    #     img_height,img_width=tuple(img.shape[:2])
    #     height_point=np.random.randint(self.up_cut,img_height-self.height-self.down_cut)
    #     width_point=np.random.randint(self.left_cut,img_width-self.width,self.right_cut)
    #     return img[height_point:height_point+self.height,width_point:width_point+self.width],target[height_point:height_point+self.height,width_point:width_point+self.width]


    def __call__(self,img,target):
        img_height,img_width=tuple(img.shape[:2])
        height_point=np.random.randint(self.up_cut,img_height-self.height-self.down_cut)
        width_point=np.random.randint(self.left_cut,img_width-self.width-self.right_cut)
        #print(height_point,width_point)
        return img[height_point:height_point+self.height,width_point:width_point+self.width],target[height_point:height_point+self.height,width_point:width_point+self.width]


class RandomHorizontalFlip(object):
    def __init__(self,p):
        self.p=p

    def __call__(self,img,target):
        is_flip=np.random.uniform(0,1)>=self.p
        if is_flip:
            img=img[:,::-1]
            target=target[:,::-1]
        return img,target
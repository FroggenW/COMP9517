
import cv2
import numpy as np 

ImgDir = '.\\Group_Component\\sequence\\'
ImgBaseName = '{0:06d}.jpg'
ImgCnt = 795

def GetImageCnt():
    return ImgCnt
def ReadImageList(beg, end, imgdir=ImgDir, imgbasename=ImgBaseName):
    last = min(end, ImgCnt+1)
    N = last-beg 
    imglst = list(range(N))
    for i,id in enumerate(range(beg,last )):
        filename = imgdir+imgbasename.format(id)
        imglst[i] = np.asarray(cv2.imread(filename), dtype=np.float32)/255.0 + 0.001
        # imglst[i] = cv2.cvtColor( cv2.imread(filename), cv2.COLOR_BGR2RGB)
    return imglst
        
def ReadImageListByIndex( idx ):
    N = len(idx)

    imglst = list(range(N))
    for i,id in enumerate( idx):
        filename = ImgDir+ImgBaseName.format(id)
        imglst[i] = np.asarray(cv2.imread(filename), dtype=np.float32)/255.0 + 0.001
        # imglst[i] = cv2.cvtColor( cv2.imread(filename), cv2.COLOR_BGR2RGB)
    return imglst
    

def Image2Objarray(img):
    row, col,imgch = img.shape 
    out = np.ndarray((row, col), dtype=np.ndarray)
    for i in range(row):
        for j in range(col):
            out[i,j] = img[i,j,:]
    return out 
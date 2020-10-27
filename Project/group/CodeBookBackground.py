import cv2
import time
import numpy as np 
from DataIO import GetImageCnt, ReadImageList , ReadImageListByIndex, Image2Objarray

class  CodeWord:
    thres_color = 1e-4
    alpha = 0.5 # 0.4 - 0.7 
    beta = 1.2 # 1.1 - 1.5
    # Colordist = lambda x,v : (np.sum(x*v))**2
    def __init__(self,v, blow, bhigh, f, mnrl, first, last, v2 = None  ):
        self.v=v.copy()
        if v2 is None:
            self.v2 = np.sum(np.power(v,2))
        else:
            self.v2 = v2.copy()
        self.blow = blow
        self.bhigh = bhigh
        self.bmin = 0
        self.bmax = 0
        # self.bmin = bmax * CodeWord.alpha 
        # self.bmax = min( bmax* CodeWord.beta, bmin/CodeWord.alpha)
        self.f = f
        self.mnrl = mnrl 
        self.first = first 
        self.last = last 
        self.UpdateBtns()

    def UpdateBtns(self, Blow = 100, Bhigh=-100):
        self.blow = min(self.blow, Blow)
        self.bhigh = max(self.bhigh, Bhigh)
        self.bmin = self.blow * CodeWord.alpha 
        self.bmax = max( self.bhigh* CodeWord.beta, self.blow/CodeWord.alpha)
    def Find(self, x, x2, b , color_thres = -1):
        if color_thres < 0 :
            color_thres = CodeWord.thres_color
        if self.bmax <= b or self.bmin >= b :
            return False
        if self.Colordist( x, x2) > color_thres:
            return False
        else:
            return True
    def Update(self, x,  b, t ):
        self.v = (self.v*self.f + x)/(self.f+1)
        self.v2 = np.sum(np.power(self.v,2))
        self.UpdateBtns(b)
        self.f += 1
        # self.mnrl = max( self.mnrl, t-self.last) 
        self.mnrl = max( self.mnrl, t-self.last) 
        self.last = t 

    def UpdateMnrl(self, N):
        self.mnrl = max( self.mnrl,  N -1 - self.last + self.first)

        
        
    def Colordist(self, x, x2 ):
        # return (x2 - ((np.sum(x*self.v))**2)/self.v2)
        d = (x2 - ((np.sum(x*self.v))**2)/self.v2)
        return d 
    def Cvt2Array(self):
        a = np.zeros((10,))
        a[0:3]= self.v 
        a[3] = self.v2 
        a[4] = self.blow
        a[5] = self.bhigh
        a[6] = self.f 
        a[7] = self.mnrl
        a[8] = self.first
        a[9] = self.last
        return a

def BuildCodeWordFromArray( a ):
    v = a[:3]
    v2 = a[3]
    blow = a[4]
    bhigh = a[5]
    f = a[6]
    mnrl = a[7]
    first = a[8]
    last = a[9]
    cw = CodeWord(v, blow, bhigh, f, mnrl, first, last, v2 )
    return cw 

def LoadCodeBookFromFile(filename):
    CBf = np.load(filename, allow_pickle=True)
    row , col = CBf.shape 
    for i in range(row ):
        for j in range(col):
            cb = CBf[i,j]
            CBf[i,j] = [ BuildCodeWordFromArray(a) for a in cb  ]
    return CBf 

def ExtractBackGround(filename = 'CodeBook_40_0.0001.npy'):
    # CB = LoadCodeBookFromFile(CBfile)
    
    CBf = np.load(filename, allow_pickle=True)
    row, col = CBf.shape
    # bg = np.zeros(shape=(row, col, 3), dtype=np.uint8)
    bg = cv2.imread('1.jpg')
    # bg = baseimg.copy()
    for i in range(row):
        for j in range(col):
            if len(CBf[i,j]) == 0 :
                continue
            bg[i,j,:] = ((CBf[i,j][0][:3])*255).astype(np.uint8)
    outfilename = 'background2.jpg'
    cv2.imwrite(outfilename, bg)
    return bg

def ConstructCodeBook(beg=1, end=500, step = 2):
    if end < 0:
        end = GetImageCnt()
    # imglst = ReadImageList(beg, end)
    imglst = ReadImageListByIndex( np.arange(beg, end, step) )
    X2 =  [ np.sum(np.power(img,2), axis=2) for img in imglst]
    Brightness = [ np.sum(img, axis=2) for img in imglst]
    # Brightness = np.stack(Brightness, axis=2)
    imgrow, imgcol, imgch = imglst[0].shape 
    CB = np.ndarray((imgrow, imgcol), dtype=list)
    
    img = imglst[0]
    btns = Brightness[0]
    for i in range(imgrow):
        for j in range(imgcol):
            cb = CodeWord( img[i,j,:], btns[i,j],btns[i,j], 1, 0,0,0 )
            CB[i,j] = [cb]

    # probP = (164,122)
    for k, (img, x2, btns) in enumerate(zip(imglst[1:],X2[1:], Brightness[1:]),1):
        for i in range(imgrow):
            for j in range(imgcol):
                # if i == probP[0] and j == probP[1]:
                #     i = i
                cblist = CB[i,j]
                px=img[i,j,:]
                px2 = x2[i,j]
                pbtns=btns[i,j]
                findflag = False
                for cb in cblist:
                    if cb.Find(px, px2, pbtns):
                        cb.Update(px, pbtns, k)
                        findflag = True
                        break
                if findflag:
                    continue 
                else:
                    cb = CodeWord(px,pbtns, pbtns,1,k,k,k, px2)
                    cblist.append(cb)
    N = len(imglst)        
    ThresMnrl = N/2
    for i in range(imgrow):
        for j in range(imgcol):
            for cb in CB[i,j]:
                cb.UpdateMnrl(N)
            CB[i,j] = [ cb for cb in CB[i,j] if cb.mnrl <= ThresMnrl ]
    return CB , N
            

            
def  TestCodeBookConstruct():
    tic = time.perf_counter()
    start = 1
    last = 201
    step = 8
    CB,N = ConstructCodeBook(start,last, step)
    toc = time.perf_counter()-tic 
    print('time cost of CodeBook constructing with N=%d : %f s' % (N, toc))
    CBA = CB.copy()
    row , col = CB.shape 
    for i in range(row):
        for j in range(col):
            CBA[i,j] = [ cb.Cvt2Array() for cb in CBA[i,j] ]
    filename='CodeBook_%d_%f.npy' % (N , CodeWord.thres_color)
    np.save(filename, CBA)         


def  SiglePixelMovementDetect(cblist, px, px2, pbtns,color_thres ):
    pdect_img = 255
    for cb in cblist:
        if cb.Find(px, px2, pbtns, color_thres):
            pdect_img = 0            
            break
    return pdect_img


def  MovementDetect( imgIndices, CBfile, color_thres = CodeWord.thres_color*10):
    imglst = ReadImageListByIndex(imgIndices)
    N= len(imglst)
    X2 =  [ np.sum(np.power(img,2), axis=2) for img in imglst]
    Brightness = [ np.sum(img, axis=2) for img in imglst]
    imgrow, imgcol, imgch = imglst[0].shape 
    detect_imglst = [ np.ndarray((imgrow, imgcol), dtype = np.uint8) for i in range(N) ]
    CB = LoadCodeBookFromFile(CBfile)
    detector = np.frompyfunc(SiglePixelMovementDetect,5,1 )
    color_thres = np.ones(CB.shape)*color_thres
    for i , (img, x2, btns) in enumerate( zip(imglst,X2, Brightness) ):
        detect_imglst[i] = np.asarray( detector(CB,Image2Objarray(img), x2,btns ,color_thres ), dtype=np.uint8)
    return detect_imglst



def TestMovementDetect():
    # CBfile = 'CodeBook_1000_0.0001.npy'
    # CBfile = 'CodeBook_100_0.0001.npy'
    CBfile = 'CodeBook_40_0.0001.npy'
    beg = 1
    # end = GetImageCnt()
    end = beg + 20
    imgIndices = list(range(beg,end, 5))
    # imgIndices = [405, 525, 570, 775,1525]

    tic = time.perf_counter()
    detect_imglst = MovementDetect( imgIndices, CBfile)
    N = len(imgIndices)
    toc = time.perf_counter()-tic 
    print('average time cost of movement detection {0} s'.format(toc/N ))
    for k, img in zip(imgIndices, detect_imglst):
        filename = '.\\detected2\\%d.jpg' % k 
        cv2.imwrite(filename, img)

if __name__ == '__main__':
    TestCodeBookConstruct()
    # TestMovementDetect()
    ExtractBackGround('CodeBook_25_0.000100.npy')
    
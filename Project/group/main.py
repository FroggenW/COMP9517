import cv2
import numpy as np
import os
import copy

#   The  predefinition of the seven trackers
# OPENCV_OBJECT_TRACKERS = {
#     "csrt": cv2.TrackerCSRT_create,
#     "kcf": cv2.TrackerKCF_create,
#     "boosting": cv2.TrackerBoosting_create,
#     "mil": cv2.TrackerMIL_create,
#     "tld": cv2.TrackerTLD_create,
#     "medianflow": cv2.TrackerMedianFlow_create,
#     "mosse": cv2.TrackerMOSSE_create
# }

def py_cpu_nms(dets, thresh):
    y1 = dets[:, 1]
    x1 = dets[:, 0]
    y2 = y1 + dets[:, 3]
    x2 = x1 + dets[:, 2]

    scores = dets[:, 4]  # bbox打分
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]
    return keep

def nms_cnts(cnts, mask, min_area):
    bounds = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > min_area]
    if len(bounds) == 0:
        return []
    scores = [calculate(b, mask) for b in bounds]
    bounds = np.array(bounds)
    scores = np.expand_dims(np.array(scores), axis=-1)
    nms_threshold = 0.3
    keep = py_cpu_nms(np.hstack([bounds, scores]), nms_threshold)
    return bounds[keep]

def calculate(bound, mask):
    x, y, w, h = bound
    area = mask[y:y + h, x:x + w]
    pos = area > 0 + 0
    score = np.sum(pos) / (w * h)
    return score

#返回所选择矩形框的中心点
def center(box):
    (x, y, w, h) = box
    center_x = int(x + w / 2.0)
    center_y = int(y + h / 2.0)
    return (center_x, center_y)

class Rect:
    def Create(topleft_w_h):
        tl = [topleft_w_h[0], topleft_w_h[1]]
        br = [tl[0]+topleft_w_h[2], tl[1]+topleft_w_h[3]]
        return Rect(tl, br)
    def __init__(self, p1=[10000,1000000], p2=[-10000,-10000]):
        # self.tl = topleft.copy()
        # self.br = botright.copy()
        self.tl = [min(p1[0], p2[0]), min( p1[1], p2[1])]
        self.br = [max(p1[0], p2[0]), max( p1[1], p2[1])]
    def copy(self):
        return copy.copy(self)

    def center(self):
        return ( int( (self.tl[0]+self.br[0])/2), int((self.tl[1]+self.br[1])/2) )
    def  contains(self, p):
        if p[0] >= self.tl[0] and p[0] <= self.br[0] and \
           p[1] >= self.tl[1] and p[1] <= self.br[1] :
            return True 
        else:
            return False 
    def update(self, p):
        if self.contains(p):
            return 
        self.tl[0] = min(self.tl[0], p[0])
        self.tl[1] = min(self.tl[1], p[1])
        self.br[0] = max(self.br[0], p[0])
        self.br[1] = max(self.br[1], p[1])


def ReTriveImageFilenames( video_path):
    v_image = []
    for root, dirs, files in os.walk(video_path):
        files.sort()
        for name in files:
            v_image.append(os.path.join(video_path, name))
    return v_image

def  EculidDist(p1,p2):
    return  (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

class  CadiBox( Rect):
    def __init__(self, p1=[10000,1000000], p2=[-10000,-10000]):
        Rect.__init__(self, p1, p2)
        self.has_located = False 

class MovingObject:
    def __init__(self, bbox):
        self.bbox = bbox.copy() 
        self.ct = self.bbox.center()
        self.path = [self.ct]
        self.has_lost = False
        self.has_updated = False 
        self.cadi_boxes = []
        self.cadi_boxes_dist = []
        self.inGateBorder = False 
    def update(self):
        if len(self.cadi_boxes) == 0 :
            return False
        else:
            id = np.argmin(self.cadi_boxes_dist)
            bbox = self.cadi_boxes[id]
            bbox.has_located = True 
            self.ct = bbox.center() 
            self.path.append( self.ct )
            self.bbox = bbox            
            self.has_lost = False
            self.has_updated = True 
            return True 
    def prepareForUpdate(self):
        self.has_updated = False 
        self.has_lost = True 
        self.cadi_boxes.clear()
        self.cadi_boxes_dist.clear()
    def insertCadiBox(self, box, dist):
        self.cadi_boxes_dist.append(dist)
        self.cadi_boxes.append(box)
    def reset(self):
        self.path.clear()
        self.cadi_boxes.clear()
        self.cadi_boxes_dist.clear()
    def copy(self):
        return copy.copy(self)
class MovingGroup:
    def __init__(self, movObjs):
        self.movObjs = movObjs 
    def findNeighbor(self, pt, maxdist=120):
        md = maxdist**2
        objfind = None
        for obj in self.movObjs:
            d=EculidDist(obj.ct,pt)
            if d < md :
                md = d
                objfind = obj 
        return objfind, md
    def remove(self, obj):
        self.movObjs.remove(obj)
    def add(self, bbox):
        self.movObjs.append( MovingObject(bbox))
    def update(self, bounds, border,people_into_border, people_off_border):
        for obj in self.movObjs:
            obj.prepareForUpdate()
        boxes = [CadiBox(box.tl, box.br) for box in bounds]
        for box in boxes:
            boxct = box.center()
            objfind , dist= self.findNeighbor(boxct)
            if objfind is not None:
                objfind.insertCadiBox( box, dist)
            
        lost_objs = [ obj for obj in self.movObjs if not obj.update()]

        cnt_inborder = 0
        cnt_outborder = 0
        cnt_inborder_cur = 0 

        add_objs = [ MovingObject(box) for box in boxes if not box.has_located ]


        for obj in lost_objs:
            if obj.inGateBorder :
                cnt_outborder += 1
            self.movObjs.remove(obj)

        cnt_obj = len(bounds)
        state_objs = [False]*cnt_obj 
        for obj in self.movObjs:
            state = border.contains( obj.ct )
            if state == obj.inGateBorder:
                continue 
            if state:
                cnt_inborder += 1                
            else:
                cnt_outborder += 1
            obj.inGateBorder = state 


        for obj in add_objs:
            state = border.contains( obj.ct )
            obj.inGateBorder = state             
            self.movObjs.append(obj)
            if state:
                cnt_inborder += 1

        for obj in self.movObjs:
            if obj.inGateBorder:
                cnt_inborder_cur += 1
        people_into_border += cnt_inborder 
        people_off_border += cnt_outborder

        return  cnt_obj, cnt_inborder_cur, people_into_border, people_off_border


class ObjectDetector:
    def  __init__(self, bg_gray, binary_thres=40, medianBlurSize = 5, area_thres=200):
        self.bg = bg_gray 
        self.bin_thres = binary_thres
        self.medianBlurSize = medianBlurSize
        self.es_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.es_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.area_thres = area_thres
    def __call__(self, img):
        gray_diff = cv2.absdiff(img, self.bg)
        _, mask = cv2.threshold(gray_diff, self.bin_thres, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, self.medianBlurSize)
        mask = cv2.dilate(mask, self.es_dilate) 
        mask = cv2.erode(mask, self.es_erode) 

        cnts, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounds = nms_cnts(cnts, mask, self.area_thres)
        return [Rect.Create(bd) for bd in bounds]

class  PedestrainDetector:
    def __init__(self, background, title_window = 'Pedestrains Detection' ):
        self.gateBorder = Rect.Create((10,10,30,30))
        self.bg = background.copy()
        self.bg_gray = cv2.cvtColor( background, cv2.COLOR_BGR2GRAY)
        self.win_title = title_window
        self.objDetector = ObjectDetector(self.bg_gray)
        self.movGroup = MovingGroup([])
        self.cnt_in = 0
        self.cnt_out = 0

    def InitGateBorder(self, img):
        title_window = self.win_title
        def onMouseClick( event,  x,  y,  flags , pts):
            if event == cv2.EVENT_LBUTTONDOWN:
                pts[0] = (x,y)
                pts[2] = True
            elif event == cv2.EVENT_MOUSEMOVE:
                if not pts[2] :
                    return 
                imgtmp = img.copy()
                secondpt = (x,y)
                pts[1] = secondpt
                firstpt = pts[0]
                topleft = ( min(firstpt[0],secondpt[0]) ,  min(firstpt[1],secondpt[1]) )
                botright = (max(firstpt[0],secondpt[0]) ,  max(firstpt[1],secondpt[1]) )
                cv2.rectangle(imgtmp,topleft,botright,(255,0,0),2)
                cv2.imshow(title_window,imgtmp)
            elif event == cv2.EVENT_LBUTTONUP:
                pts[2] = False
        cv2.namedWindow(title_window)
        cv2.imshow(title_window,img)
        firstpt = [0,0]
        secondpt = [100,100]
        laststate = False 
        pts = [firstpt, secondpt, laststate]
        cv2.setMouseCallback(title_window, onMouseClick, pts)
        cv2.waitKey()
        self.gateBorder = Rect(pts[0], pts[1])

    def InitMovingObjects(self, img):
        bounds = self.objDetector(img)
        movObjs = [MovingObject(bd) for bd in bounds]
        self.movGroup = MovingGroup(movObjs)

    def drawTrace(self, img):
        for obj in self.movGroup.movObjs:
            if len(obj.path) < 2:
                continue 
            ct_beg = tuple(obj.path[0])
            cv2.circle(img, ct_beg, 2, (122,255,0), 2)
            for ct_end in obj.path[1:]:
                cv2.arrowedLine(img, ct_beg, tuple(ct_end), (0,255,122),2)
                ct_beg = tuple(ct_end)
    def update(self, img, img_gray, people_into_border, people_off_border):        
        bounds = self.objDetector(img_gray)
        cv2.rectangle(img, tuple(self.gateBorder.tl), tuple(self.gateBorder.br), (255, 0, 0), 2)
        for box in bounds:
            boxct = box.center()
            cv2.rectangle(img, tuple(box.tl), tuple(box.br), (0,255,0), 2)
            cv2.circle(img, boxct, 2, (0, 0, 255), 2)

        objcount, ppInBorder, people_into_border, people_off_border = self.movGroup.update(bounds,self.gateBorder,people_into_border, people_off_border)
        self.drawTrace(img)
        return objcount, ppInBorder, people_into_border, people_off_border

    def Run(self, video_path = './Group_Component/sequence/' , sample_step=1, savefile= False):
        v_image = ReTriveImageFilenames( video_path)
        TotalFileCount = len(v_image)
        video_sample = [ cv2.imread(v_image[i]) for i  in range(0, TotalFileCount, sample_step) ]
        gray_sample = [ cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in video_sample ]
        initial_frame = 1
        video_len = len(video_sample) - initial_frame
        peoples = np.zeros((video_len,), dtype=np.int32)
        moving_people = np.zeros((video_len,), dtype=np.int32)
        history_into_border = np.zeros((video_len,), dtype=np.int32)
        history_off_border = np.zeros((video_len,), dtype=np.int32)
        people_into_border = 0
        people_off_border = 0
        first_frame = video_sample[0]
        self.InitGateBorder(first_frame.copy())
        self.InitMovingObjects(gray_sample[0])

        for i in range(video_len):
            img_1 = video_sample[i+initial_frame]
            gray_1 = gray_sample[i+initial_frame]
            history_into_border[i] = people_into_border
            history_off_border[i] = people_off_border
            moving_people[i], peoples[i],people_into_border, people_off_border = self.update(img_1, gray_1,people_into_border, people_off_border)
            history_into_border[i] = people_into_border - history_into_border[i]
            history_off_border[i] = people_off_border -history_off_border[i]

        FontColor = (0,255,255)
        FontColorIn = (0,0,255)
        FontColorOut = (0,255,0)
        for i in range(video_len):
            cv2.putText(video_sample[i+initial_frame], "Moving People: {0}".format(moving_people[i]), (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, FontColor, 1)
            cv2.putText(video_sample[i+initial_frame], "People In Border: {0}".format(peoples[i]), (20,90), cv2.FONT_HERSHEY_SIMPLEX,
                        1,FontColor, 1)
            cv2.putText(video_sample[i+initial_frame], "into: {0}".format(history_into_border[i]), (20,140), cv2.FONT_HERSHEY_SIMPLEX,
                        1,FontColorIn, 1)
            cv2.putText(video_sample[i+initial_frame], "off: {0}".format(history_off_border[i]), (20,190), cv2.FONT_HERSHEY_SIMPLEX,
                        1,FontColorOut, 1)
            cv2.imshow(self.win_title, video_sample[i+initial_frame])
            keycode = cv2.waitKey(50)
            if keycode < 0:
                continue
            keycode = chr(keycode)
            if keycode == 'Q' or keycode == 'q':
                break
            elif keycode == ' ':
                cv2.waitKey(0)
        cv2.putText(video_sample[video_len], "Press any key to quit", (100,200), cv2.FONT_HERSHEY_SIMPLEX,
                        1,FontColor, 1)
        cv2.imshow(self.win_title, video_sample[video_len])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if savefile:
            imgdir = './ObjectDetected/'
            if not os.path.exists(imgdir):
                os.mkdir(imgdir)
            for id, img in enumerate(video_sample):
                filename = os.path.join(imgdir, '{0}.jpg'.format(id+1))
                cv2.imwrite(filename, img)


if __name__ == '__main__':
    bk = cv2.imread('background2.jpg')
    pd = PedestrainDetector(bk)
    pd.Run(savefile = False)
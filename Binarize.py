import numpy as np
import cv2
import skimage.filters as filters
from os import listdir
from os.path import isfile, join


class InteractiveBinarization():
    def __init__(self,path='./images/',out='./out/'):
        '''
        args:
            path: Path of the directory which has all the images
            out: Path of directory where your binarized images will be saved
        '''
        self.path = path
        self.images = [f for f in listdir(path) if isfile(join(path, f))]
        self.N = len(self.images)
        self.out = out

    def dummy(self,x=None)->None:
        '''
        Does not do anything. Used to pass to crateTrackbar as it needs a function
        '''
        pass

    
    def create_window(self,window_width:int=350,window_height:int=350)->None:
        '''
        Create a named window and set trackbar positions. Set all the values to defaults as needed.
        args:
            window_width: Width of the Window which has sliding bars
            window_height: Height of window for the sliding bars
        '''
        cv2.namedWindow('Tracking Window',cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow('Tracking Window', window_width, window_height)
        cv2.createTrackbar('gauss_k','Tracking Window',2,513,self.dummy) # gauss kernal size
        cv2.createTrackbar('x_sigma','Tracking Window',0,100,self.dummy) # gauss X sigma
        cv2.createTrackbar('y_sigma','Tracking Window',0,100,self.dummy) # gauss Y sigma
        cv2.createTrackbar('radius','Tracking Window',1,100,self.dummy) # sharpen radius
        cv2.createTrackbar('amount','Tracking Window',1,200,self.dummy) # sharpen amount number
        cv2.createTrackbar('angle','Tracking Window',0,360,self.dummy) # rotation angle
        # cv2.createTrackbar('morph_k','Tracking Window',2,30,self.dummy) # Morph kernal size
        # cv2.createTrackbar('method','Tracking Window',0,3,self.dummy) # Morphological methods. Just Testing

        
    def binarize(self,)->None:
        '''
        Method to binarize the Image based on the sliding values from the bars. It accepts Gauss Kernal, Sharpeen Amount, Sharpen Radius, Rotation Angle
        Press 'esc' or 'q' to quit, 's' to save the binarized image, 't' for printing the current bar values to image, 'p' for previous image and 'n' for next image
        '''
        methods = [cv2.MORPH_ERODE,cv2.MORPH_DILATE,cv2.MORPH_OPEN,cv2.MORPH_CLOSE] # morphological methods
        QUIT = False
        put_text = False
        read_image = True
        counter = 0

        while not QUIT:
            if read_image:
                self.create_window() # create a new trackbar window

                img_name = self.images[counter]
                img = cv2.imread(self.path+img_name)
                read_image = False

            self.g_k = cv2.getTrackbarPos('gauss_k','Tracking Window')
            if self.g_k % 2 == 0:
                self.g_k+=1

            self.g_x_sigma = cv2.getTrackbarPos('x_sigma','Tracking Window')
            self.g_y_sigma = cv2.getTrackbarPos('y_sigma','Tracking Window')
            self.amount = cv2.getTrackbarPos('amount','Tracking Window') # 1,2,3,4
            self.radius = cv2.getTrackbarPos('radius','Tracking Window') # same as above
            self.amount = round(self.amount/10,2) # 0.1.......... 9.99
            self.radius = round(self.radius/10,2) # same asa above
            self.angle = cv2.getTrackbarPos('angle','Tracking Window')
            # self.morph_k = cv2.getTrackbarPos('morph_k','Tracking Window')
            # self.method = cv2.getTrackbarPos('method','Tracking Window')

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            smooth = cv2.GaussianBlur(gray, (self.g_k,self.g_k), self.g_x_sigma,sigmaY=self.g_y_sigma)
            division = cv2.divide(gray, smooth, scale=255)


            sharp = filters.unsharp_mask(division, radius=self.radius, amount=self.amount, multichannel=False, preserve_range=False)
            sharp = (255*sharp).clip(0,255).astype(np.uint8)

            # morph_kernel = np.ones((self.morph_k,self.morph_k))
            # sharp = cv2.morphologyEx(sharp, methods[self.method], morph_kernel)

            thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_OTSU )[1]

            # rotate
            (h, w) = thresh.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, self.angle, 1)
            thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue=255)

            if put_text:
                text = f"g_k: {self.g_k} , x_sigma: {self.g_x_sigma} , y_sigma: {self.g_y_sigma} , amt: {self.amount} , rad: {self.radius} , angle: {self.angle}"
                cv2.putText(thresh,text,org=(30,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0,128,0),thickness=1)

            cv2.imshow('Binary', thresh)
            key = cv2.waitKey(1) # show for 1 miliseconds. Because the loop is infinite, it'll be infinitely showing the results
            if key==27 or key == ord('q'): # Press escape / q  to close all windows
                QUIT = True
                break

            elif key == ord('s'): # save binary image
                cv2.imwrite(self.out+'binary_'+img_name, thresh)

            elif key == ord('t'): # show or hide text on image
                put_text = not put_text

            elif key == ord('n'):
                if counter < self.N-1:
                    read_image = True
                    counter += 1
                    cv2.destroyWindow('Tracking Window')

            elif key == ord('p'):
                if counter > 0: 
                    read_image = True
                    counter -= 1
                    cv2.destroyWindow('Tracking Window')

        cv2.destroyAllWindows()


def has_blur(image_path:str,thresh:float)->bool:
    '''
    Use Laplacian Variance to find if an image has blur or not. It is very critical to find the threshold and is vert data specific
    args:
        image_path: path of the image
        thresh: threshold to find the find. If the Laplacian Variance has value Lower than that threshold, it has blur
    '''
    gray = cv2.imread(image_path,0)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < thresh:
        return True
    return False


def image_colorfulness(image_path:str,thresh:float,alpha:float=0.5,beta:float=0.3)->bool:
    '''
    Find the image colorfulness. Idea given at PyimageSearch. Uses a threshold to find the colorfulness. threshold is critical and data dependent
    We'll use it as a very basic spam classifier.
    NOTE: It'll fail if the incoming question image is not from a B&W book and has colorful texts and diagrams
    args:
        image_path: Path of the image
        thresh: Threshold of the colurfulness. We'll assume that images crossing this threshold are spams
        alpha: floating number in computing the value of  yb  as  (alpha * (R + G) - B)
        beta: number for computing the colorfulness as  (stdRoot + (beta * meanRoot))
    '''
    image = cv2.imread(image_path)
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G) # difference between Red and Green Channel
    yb = np.absolute(alpha * (R + G) - B) # yb = 0.5 * (R + G) - B
    (rbMean, rbStd) = (np.mean(rg), np.std(rg)) # mean and std
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    colorfulness = stdRoot + (beta * meanRoot)
    if colorfulness > thresh:
        return True
    return False

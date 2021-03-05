import numpy as np
import cv2
import skimage.filters as filters
from os import listdir
from os.path import isfile, join


class InteractiveBinarization():
    def __init__(self,path='./images/',out='./out/'):
        self.path = path
        self.images = [f for f in listdir(path) if isfile(join(path, f))]
        self.N = len(self.images)
        self.out = out

    def dummy(self,x=1):
        pass

    def binarize(self,window_width=350,window_height=350):
        
        cv2.namedWindow('Tracking Window',cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow('Tracking Window', window_width, window_height)
        
        cv2.createTrackbar('kernel','Tracking Window',3,513,self.dummy) # gauss kernal size
        cv2.createTrackbar('x_sigma','Tracking Window',0,100,self.dummy) # gauss X sigma
        cv2.createTrackbar('y_sigma','Tracking Window',0,100,self.dummy) # gauss Y sigma

        cv2.createTrackbar('amount1','Tracking Window',0,7,self.dummy) # sharpen amount number
        cv2.createTrackbar('amount2','Tracking Window',1,100,self.dummy) # sharpen amount decimal
        cv2.createTrackbar('radius1','Tracking Window',0,7,self.dummy) # sharpen radius
        cv2.createTrackbar('radius2','Tracking Window',1,100,self.dummy) #  sharpen radius decimal

        cv2.createTrackbar('angle','Tracking Window',0,360,self.dummy) # rotation angle

        QUIT = False
        put_text = False
        read_image = True
        counter = 0
        while not QUIT:

            if read_image:
                img_name = self.images[counter]
                img = cv2.imread(self.path+img_name)
                read_image = False

            g_k = cv2.getTrackbarPos('kernel','Tracking Window')
            if g_k % 2 == 0:
                g_k+=1

            g_x_sigma = cv2.getTrackbarPos('x_sigma','Tracking Window')
            g_y_sigma = cv2.getTrackbarPos('y_sigma','Tracking Window')

            s_a1 = cv2.getTrackbarPos('amount1','Tracking Window') # 1,2,3,4
            s_a2 = cv2.getTrackbarPos('amount2','Tracking Window') # .01, ..... 0.99
            s_r1 = cv2.getTrackbarPos('radius1','Tracking Window') # same as above
            s_r2 = cv2.getTrackbarPos('radius2','Tracking Window')
            s_a = round(s_a1 + s_a2/100,2) # 1.01.......... 7.99
            s_r = round(s_r1 + s_r2/100,2) # same asa above

            angle = cv2.getTrackbarPos('angle','Tracking Window')

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            smooth = cv2.GaussianBlur(gray, (g_k,g_k), g_x_sigma,sigmaY=g_y_sigma)
            division = cv2.divide(gray, smooth, scale=255)
            sharp = filters.unsharp_mask(division, radius=s_r, amount=s_a, multichannel=False, preserve_range=False)
            sharp = (255*sharp).clip(0,255).astype(np.uint8)

            kernel = np.ones((5,5),np.uint8)
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_OTSU )[1]

            # rotate
            (h, w) = thresh.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue=255)

            if put_text:
                text = f"g_k: {g_k} , g_x_sigma: {g_x_sigma} , g_y_sigma: {g_y_sigma} , s_a: {s_a} , s_r: {s_r} , angle: {angle}"
                cv2.putText(thresh,text,org=(30,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0,128,0),thickness=1)

            cv2.imshow('Image', thresh)

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

            elif key == ord('p'):
                if counter > 0: 
                    read_image = True
                    counter -= 1

        cv2.destroyAllWindows()
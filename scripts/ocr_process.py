__AUTHOR__  = "Mahesh Deshwal, Vedant Joshi"

'''
 Python script to perform preprocessing on image for
 extraction of text data by the tesseract OCR.

 ** While running on a local machine change the address of the tess data for language i.e. eng+hin
'''

import numpy as np
import matplotlib.pyplot as plt
import pytesseract as ocr
import cv2
import skimage
from skimage.filters import try_all_threshold
from skimage import filters
from skimage.filters import  threshold_isodata, threshold_li, threshold_mean, threshold_minimum, threshold_triangle, threshold_yen
from skimage import color, data, restoration
from scipy.signal import convolve2d
from skimage import exposure
from scipy.ndimage import interpolation as inter
from skimage.io import imread
from skimage.color import rgb2gray
from PIL import Image
import random
import imutils
import time



class TextExtract():

    def load_image(self,image_path,dpi=(300,300)):
        '''
         To load the image using PIL module for a specific DPI
         args:
            image_path : {string} folder path of the image
            dpi        : (tuple) to specify dpi deafault : (300,300)
        out :
            returns the image specified in the path
        '''

        im = imread(image_path)
        dpi_im = Image.fromarray(im)
        dpi_im.save(image_path,dpi=dpi)
        return im

    
    def convert_grayscale(self,image):
        '''
        Conversion of images to grayscale for further process pipeline
        args:
            image : {numpy array} of RGB format
        out :
            returns the grayscale image
        '''

        im = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        return im

        # smoothing/filters
        
    
    def apply_deblur(self,image,kind='gaussian',filter_size=(5,5)):
        '''
        Function used to apply specific filter based on the requirement in the process pipeline
        args :
            image         : {numpy array} of grayscale format
            kind          : {string} to specify filter to apply default : "median"
            filter_size   : (tuple) indicating size of filter default : 5x5 size
        out :
            returns the smoothed image
        '''

        if kind == 'median':
            im = cv2.medianBlur(image,filter_size)
            return im
        
        elif kind == 'gaussian':
            im = cv2.GaussianBlur(image, filter_size, 0)
            return im
        
        elif kind == 'blur':
            im = cv2.blur(image)
            return im
        
        elif kind == 'bilateral':
            im = cv2.bilateralFilter(image)
            return im
        
        elif kind == 'manual':
            kernel = np.ones(filter_size,np.uint8)
            im = cv2.filter2D(image, -1, kernel)
            return im

        
    def apply_thresholding(self,image,op='otsu'):
        '''
         Function to apply specific thresholding to get a binarized output
         args:
            image : {numpy array} in grayscale
            op    : {string} to specify type of thresholding default : otsu
        out:
            returns the binarized image
        '''
        if op == 'otsu':
            im = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU )[1] 
            return im

        elif op == 'isodata':
            thresh = threshold_isodata(image)
            im = image > thresh
            return im

        elif op == 'li':
            thresh = threshold_li(image)
            im = image > thresh
            return im

        elif op == 'mean':
            thresh = threshold_mean(image)
            im = image > thresh
            return im
        
        elif op == 'minimum':
            thresh = threshold_minimum(image)
            im = image > thresh
            return im
        
        elif op == 'triangle':
            thresh = threshold_triangle(image)
            im = image > thresh
            return im
        
        elif op == 'yen':
            thresh = threshold_yen(image)
            im = image > thresh
            return im

        
    def apply_weiner(self,image):
        '''
        Application of weiner filter
        args:
            image : {numpy array} image in grayscale
        out : 
            image applied with weiner filter
        '''

        psf = np.ones((5, 5)) / 25
        img = convolve2d(img, psf, 'same')
        img += 0.1 * img.std() * np.random.standard_normal(img.shape)
        deconvolved_img = restoration.wiener(img, psf, 1100)

        return deconvolved_img
    
 
    def apply_skew_correction(self, image, delta=1, limit=5):  
        '''
        Function that returns the image with corrected skew
        args :
            image : {numpy array} image in binarized form
            delta : {int} for sampling in the -limit,limit + delta range default : 1
            limit : {int} to specify range of angles to explore default : 5
        out :
         corrected angle and the rotated image
        '''

        def determine_score(arr, angle):
            '''
            Function that returns the score of histogram for the given angle at which we check
            args:
                arr   : {numpy image in binarized format}
                angle : {integer angle at which we calcuate the score}
            '''
            data = inter.rotate(arr, angle, reshape=False, order=0)
            histogram = np.sum(data, axis=1)
            score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
            return histogram, score

    
        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            histogram, score = determine_score(image, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        #rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # Better Border Removal
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue=255)

        return best_angle, rotated
    
            
    def perfect_binary(self, img_path,filter_size=(95,95),unsharp_radius=1.5,unsharp_amount=1.5):
        
        '''
        Make a perfect binary image with blackness removed
        args:
            img_path       : {str} path of the image
            filter_size    : (tuple) filter size
            unsharp_radius : {float} value to use in skimage.filters.unsharp_mask (radius) argument
            unsharp_amount : {float} value to use in skimage.filters.unsharp_mask (amount) argument
        out:
            smooth,division,sharp,thresh: (tuple) 4 different types of cv2 images array
        '''

        # image loading
        im = self.load_image(img_path)

        # grayscaling
        gray = self.convert_grayscale(im)

        # blur
        smooth = self.apply_deblur(gray, 'gaussian', filter_size)

        # divide gray by morphology image
        division = cv2.divide(gray, smooth, scale=255)

        # sharpen using unsharp masking
        sharp = filters.unsharp_mask(division,radius=unsharp_radius,amount=unsharp_amount,multichannel=False,
                                    preserve_range=False)
        sharp = (255*sharp).clip(0,255).astype(np.uint8)

        # threshold
        thresh = self.apply_thresholding(sharp)

        # return results
        return smooth,division,sharp,thresh
    

    def find_biggest_contour(self, image):
        '''
            Function used to finc the border of the image
            args:
                image : {numpy array} image in grayscale
            out :
                mask of the biggest contour
        '''

        # Copy to prevent modification
        image = image.copy()
        contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours are found then we return simply return nothing
        if(len(contours)==0):
            return -1,-1,-1

        # Isolate largest contour
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

        # Empty image mask with black background
        mask = np.zeros(image.shape, np.uint8)
        # Applying the largest contour on the empty image of zeros
        cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
        return mask
    
    
    def perform_ocr(self, image,lang_path):
        '''
         Function that applies the pytessaract OCR & returns the corresponding string
         args :
            image     : {numpy array} binarized image
            lang_path : {string} path adderess for the language to interpret for OCR
         out :
            string read by the OCR is returned 
        '''

        cli_config = '--oem 1 --psm 12 --tessdata-dir '+lang_path+' -l eng+hin'
        conv_string = ocr.image_to_string(image=image,config=cli_config,lang='eng+hin')
        return conv_string

    
    def perform_net_ocr(self,image_path):
        '''
         Performs a fixed pre-processing pipeline & application of OCR
         args:
            image_path : {string} file location
        '''

        # Gets the fixed thresholded image
        smooth,division,sharp,thresh = self.perfect_binary(image_path)

        # performs deskew
        angle, thresh = self.apply_skew_correction(thresh,1,60)


        m1 = self.find_biggest_contour(thresh) 

        # Black region after skew removal
        m2 = cv2.bitwise_not(m1, mask = None)
        image = thresh + m2

        print(self.perform_ocr(thresh,"/usr/local/Cellar/tesseract/4.1.1/share/tessdata"))
        

# Net function
# TextExtract().perform_net_ocr('/Users/Vedant_J/Desktop/InstaTemp/test_image/100.png')
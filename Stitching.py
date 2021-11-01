# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 10:13:55 2021

@author: KONIDE
"""


##----------------Stitcher------------------

import numpy as np
import cv2
import imutils


class Line :
    def __init__(self, data1, data2):
   
        self.line1 = data1
        self.line2 = data2
        #print(self.line1)
    def slope(self):
        (x1, y1), (x2, y2) = self.line1
        (x3, y3), (x4, y4) = self.line2
        
        if (y2-y1) == 0 :
            #print('Ys are equal, m1 = 0')
            m1 = 0
        else:
            m1 = (float(y2)-y1)/(float(x2)-x1)
        
        if (y4-y3) == 0 :
            #print('Ys are equal, m2 = 0')
            m2 = 0
        else:
            m2 = (float(y4)-y3)/(float(x4)-x3)
            
        return m1, m2
                    
    def yintercept(self, m1, m2):
        (x1, y1), (x2, y2) = self.line1
        (x3, y3), (x4, y4) = self.line2
        
        if m1 != 0 :
            b1 = y1 - m1*x1
        else :
            b1 = y1
            
        if m2 != 0 :
            b2 = y4 - m2*x4
            
        else: b2 = y4
        
        return b1, b2
    
    def findIntersect(self, m1,m2, b1, b2):
        
        if m1 != 0 | m2 != 0 :
            px = (b2-b1) / (m1-m2)
            py = (b2*m1 - b1*m2)/(m1-m2)
        elif m1 == 0 :
            px = (b1-b2)/m2
            py = b1
        elif m2 == 0 : 
            px = (b2-b1)/m1
            py = b2 
        else :  print('No points')
        
        return px, py
        

class IMP:

  
    def __init__(self, img):
        
        #import cv2
        #img = cv2.imread('c:/OpenCV/image-003.jpeg')     
        self.img = img
        
        #self.topHeight = 565
        #self.height, self.width = 1080, 1920
        
    def impTransformer(self):  
        
        import numpy as np
        import cv2 
        
        topHeight = 565
        height, width = self.img.shape[:2]
        #print('height', height,'width :', width)
        left = [(960, 380), (0, 650)]
        right = [(960, 380), (1920, 650)]
        up =  [(0, topHeight), (width+1000, topHeight)]
        down =  [(-10000,height), (width+100000, height)]
               
        leftup = Line(left, up)
        leftdown = Line(left, down)
        rightup = Line(right, up)
        rightdown = Line(right, down)
        m1, m2 = leftup.slope()
        b1, b2 = leftup.yintercept(m1,m2)
        p1x, p1y = leftup.findIntersect(m1,m2,b1,b2)
        
        #print('point1 : ', p1x, p1y)
        
       
        
        m1, m2 = leftdown.slope()
        b1, b2 = leftdown.yintercept(m1,m2)
        p2x, p2y = leftdown.findIntersect(m1,m2,b1,b2)
        #print('point2 : ', p2x, p2y)
        
       
        
        m1, m2 = rightup.slope()
        b1, b2 = rightup.yintercept(m1,m2)
        p3x, p3y = rightup.findIntersect(m1,m2,b1,b2)
        #print('point3 : ', p3x, p3y)
        
        m1, m2 = rightdown.slope()
        b1, b2 = rightdown.yintercept(m1,m2)
        p4x, p4y = leftup.findIntersect(m1,m2,b1,b2)
        #print('point4 : ', p4x, p4y)
         
        dst = np.array([[0,0], [0, 565], [1080,0], [1080,565]], dtype=np.float32)
        src = np.array([ [p1x,p1y], [p2x,p2y], [p3x,p3y], [p4x,p4y]], dtype=np.float32)
        mtrx = cv2.getPerspectiveTransform(src, dst)
        
        outimg = cv2.warpPerspective(self.img, mtrx, (1080,560))
        #cv2.imshow('out_image',outimg)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        return outimg


class Stitcher:
    
    def __init__(self):
		# determine if we are using OpenCV v3.X and initialize the
		# cached homography matrix
        self.isv3 = imutils.is_cv3()
        self.cachedH = None
        
    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
		# unpack the images
        (imageB, imageA) = images
        
		# if the cached homography matrix is None, then we need to
		# apply keypoint matching to construct it
        if self.cachedH is None:
			# detect keypoints and extract
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)
		# match features between the two images
            M = self.matchKeypoints(kpsA, kpsB,featuresA, featuresB, ratio, reprojThresh)
            
			# if the match is None, then there aren't enough matched
			# keypoints to create a panorama
            if M is None:
                return None
			# cache the homography matrix
            self.cachedH = M[1]
		# apply a perspective transform to stitch the images together
		# using the cached homography matrix
        result = cv2.warpPerspective(imageA, self.cachedH,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
		# return the stitched image
        return result
        
    def detectAndDescribe(self, image):
		# convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# check to see if we are using OpenCV 3.X
        if self.isv3:
			# detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
		# otherwise, we are using OpenCV 2.4.X
        else:
			# detect keypoints in the image
            detector = cv2.SIFT_create()
            kps = detector.detect(gray)
			# extract features from the image
            #extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = detector.compute(gray, kps)
            
		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
        kps = np.float32([kp.pt for kp in kps])
		# return a tuple of keypoints and features
        return (kps, features)
    
    
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2) 
        matches = []
		# loop over the raw matches
        for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio: matches.append((m[0].trainIdx, m[0].queryIdx))
    
	# computing a homography requires at least 4 matches
        if len(matches) > 4:
			# construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
			# compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
			# return the matches along with the homograpy matrix
			# and status of each matched point
        return (matches, H, status)
		# otherwise, no homograpy could be computed
 
    def makeImagesList(self):       
        cap = cv2.VideoCapture('C:/OpenCV/GH021047.MP4')
        
        #startFrame = 100 + skipSeconds*60        
        fps = round(cap.get(cv2.CAP_PROP_FPS)) # get frame numbers 
        
        #delay = int(5000/fps)
        
        if (fps == 0) :
            fps = 60
            
        #frameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT) # total frame numbers
        # frames = cap.get(cv2.CAP_PROP_POS_FRAMES) # current frame numbers
        
        #img = cv2.imread('c:/OpenCV/image-003.jpeg')
        
        i = 0        
        #stitcher = Stitcher()
        result = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret :
                break
            
            curr_frame = IMP(frame)   
            curr_outimg = curr_frame.impTransformer()
            curr_cropimg = curr_outimg[0:250, 0:1920]
            curr_cropimg = cv2.rotate(curr_cropimg, cv2.cv2.ROTATE_90_CLOCKWISE)
            result.append(curr_cropimg)
                
            i += 1
            
            #cv2.imwrite('road'+str(i)+'.jpg', curr_cropimg)
        
            if result is None:
                print('[info] homography could not computed')
                break
              
          #  cv2.imwrite('outimg'+ str(i)+'.jpg',outimg)    
        
        return result
    
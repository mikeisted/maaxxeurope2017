# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import numpy as np
import imutils
import time
import cv2

#global colour1Confidence
#global colour1Target
#colour1Confidence = [0,0,0,0,0]
#colour1Target = [0,0,0,0,0]
#colour1Good = False

global red1Confidence
global gradRed
red1Confidence = [0,0,0,0,0]
gradRed = [0,0,0,0,0]
red1Good = False

global red2Confidence
global intRed
red2Confidence = [0,0,0,0,0]
intRed = [0,0,0,0,0]
red2Good = False

global red1lConfidence
global gradlRed
red1lConfidence = [0,0,0,0,0]
gradlRed = [0,0,0,0,0]
red1lGood = False

global red2lConfidence
global intlRed
red2lConfidence = [0,0,0,0,0]
intlRed = [0,0,0,0,0]
red2lGood = False

global imagescale
imagescale = 35

global framecount
framecount = 0

global M
M = np.array ([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], dtype = "float32")

# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
perspx = 120 # Don't change this as it changes the height of the warped image.
pts = np.array([(0+perspx, 0), (319-perspx, 0),(319,239),(0, 239) ] , dtype = "float32")
         
# Order Points
# initialzie a list of coordinates that will be ordered
# such that the first entry in the list is the top-left,
# the second entry is the top-right, the third is the
# bottom-right, and the fourth is the bottom-left
rect = np.zeros((4, 2), dtype = "float32")
#rect = pts

# the top-left point will have the smallest sum, whereas
# the bottom-right point will have the largest sum
s = pts.sum(axis = 1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]

# now, compute the difference between the points, the
# top-right point will have the smallest difference,
# whereas the bottom-left will have the largest difference
diff = np.diff(pts, axis = 1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]

# Do 4 Point transform
(tl, tr, br, bl) = rect

# compute the width of the new image, which will be the
# maximum distance between bottom-right and bottom-left
# x-coordiates or the top-right and top-left x-coordinates
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

# compute the height of the new image, which will be the
# maximum distance between the top-right and bottom-right
# y-coordinates or the top-left and bottom-left y-coordinates
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

# now that we have the dimensions of the new image, construct
# the set of destination points to obtain a "birds eye view",
# (i.e. top-down view) of the image, again specifying points
# in the top-left, top-right, bottom-right, and bottom-left
# order

dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

# compute the perspective transform matrix and then apply it
M = cv2.getPerspectiveTransform(rect, dst)
print M

def preparemask (hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper);
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask;

def meldmask (mask_0, mask_1):
    mask = cv2.bitwise_or(mask_0, mask_1)
    return mask;


class PiVideoStream:

        global M
        
         
	def __init__(self, resolution=(320, 240), framerate=32):
               
		# initialize the camera and stream
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture,
			format="bgr", use_video_port=True)

		# initialize the frame and the variable used to indicate
		# if the thread should be stopped
		self.frame = None
		self.stopped = False


	def start(self):

                global M
                
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		print("Thread starting")
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		for f in self.stream:
			# grab the frame from the stream and clear the stream in
			# preparation for the next frame
			self.frame = f.array
			self.rawCapture.truncate(0)

			# if the thread indicator variable is set, stop the thread
			# and resource camera resources
			if self.stopped:
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()
				return

	def readfollow(self):

                global red1lConfidence
                global gradRed
                global red2Confidence
                global intRed
                global imagescale
                global M
                global framecount

                #framecount = framecount + 1
                #if framecount == 100:
                #    time.sleep(0.5)
                #    framecount  = 0

                # Set the image resolution.
                xres = 320
                yres = 240
                xColour1 = xRed = 0.0

                # Initialise confidence to indicate the line has not been located with the current frame
                newred1lConfidence = 0
                red1Good = False
                newRed2Confidence = 0
                red2Good = False

                # Initialise variables for line calculations
                xRed1 = xRed2 = 0
                yRed1 = yRed2 = 0
                intc = m = 0.0
                dx = dy = 0

                bearing = offset = 0
                            

                # return the frame most recently read
                frame = self.frame

                # apply the four point tranform to obtain a "birds eye view" of
                # the image

                
	        warped = cv2.warpPerspective(frame, M, (320+(8*imagescale), 268))
                height, width, channels = warped.shape

                # Set y coords of regions of interest.
                # The upper and lower bounds
                roidepth = 20    # vertical depth of regions of interest
                roiymin = 40    # minumum ranging y value for roi origin
                roiymintop = roiymin - roidepth
                roiymax = height - roidepth -1   # maximum ranging y value for bottom roi origin

                
                # Convert to hsv and define region of interest before further processing.
                fullhsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

                # green = 60
                # blue = 120;
                # yellow = 30;
                #Colour1 = 60

                # Set the sensitivity of the hue
                sensitivity = 20

                # Red is a special case as it sits either side of 0 on the HSV spectrum
                # So we create two masks, one greater than zero and one less than zero
                # Then combine the two.
                lower_red_0 = np.array([0, 100, 100]) 
                upper_red_0 = np.array([sensitivity, 255, 255])
                
                lower_red_1 = np.array([180 - sensitivity, 100, 100]) 
                upper_red_1 = np.array([180, 255, 255])


                # Initialise the bottom roi at the maximum limit
                y3 = roiymax
                y4 = y3 + roidepth

                while y3 > roiymin:

                    # This defines the lower band, looking closer in
                    roihsv2 = fullhsv[y3:y4, 0:(width-1)]

                    # Prepare the masks for the lower roi 
                    maskr_2 = preparemask (roihsv2, lower_red_0 , upper_red_0)
                    maskr_3 = preparemask (roihsv2, lower_red_1 , upper_red_1 )
                    maskr2 = meldmask ( maskr_2, maskr_3)

                    # find contours in the lower roi and initialize the center
                    cnts_red2 = cv2.findContours(maskr2.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
                    center2 = None

                    # Now to find the tracking line in the lower roi
                    # only proceed if at least one contour was found
                    if len(cnts_red2) > 0:
                        
                        # find the largest contour in the mask, then use
                        # it to compute the minimum enclosing circle and
                        # centroid
                        c_red2 = max(cnts_red2, key=cv2.contourArea)
                        ((x_red2, y_red2), radius_red2) = cv2.minEnclosingCircle(c_red2)
                        M_red2 = cv2.moments(c_red2)

                        # compute the center of the contour
                        cx_red2 = int(M_red2["m10"] / M_red2["m00"])
                        cy_red2 = int(M_red2["m01"] / M_red2["m00"])
                        

                        # cy_red is set in the region of interest, so need to adjust for origin in frame
                        cy_red2 = cy_red2 + y3
                        # center = ( cx_red, cy_red )

                        # only proceed if the radius meets a minimum size
                        if radius_red2 > 5:
                            newRed2Confidence = 100
                            # draw the circle and centroid on the frame
                            cv2.circle(warped, (cx_red2, cy_red2), int(radius_red2),
                            (0, 0, 255), 2)
                            # cv2.circle(frame, center, 5, (0, 0, 255), -1)

                            # calculate offset
                            xRed2 = cx_red2 - (width/2) # Contrived so pstve to right of centreline
                            yRed2 = height - cy_red2  # Adjust to make origin bottom centre of image

                            # The target has been found, so we can break out of the loop here
                            break

                    # But here the target has not been found, we need to move the ROI up
                    y3 = y3 - roidepth
                    y4 = y3 + roidepth

                # And here we have either hit the buffers or found the target.

                
                # So now try for the top roi, working down.                      
                # Initialise the top roi at the very top
                y1 = 0
                y2 = y1 + roidepth

                while y2 < y3: # Go as far as the lower roi but no more.

                    newred1lConfidence = 0

                    # This defines the upper roi, looking further away
                    roihsv1 = fullhsv[y1:y2, 0:(width-1)]                 

                    # Prepare the masks for the top roi 
                    maskr_0 = preparemask (roihsv1, lower_red_0 , upper_red_0)
                    maskr_1 = preparemask (roihsv1, lower_red_1 , upper_red_1 )
                    maskr1 = meldmask ( maskr_0, maskr_1)

                    # find contours in the upper roi and initialize the center
                    cnts_red1 = cv2.findContours(maskr1.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
                    center1 = None

                    # Now to find the tracking line in the upper roi
                    # only proceed if at least one contour was found
                    if len(cnts_red1) > 0:
                        
                        # find the largest contour in the mask, then use
                        # it to compute the minimum enclosing circle and
                        # centroid
                        c_red1 = max(cnts_red1, key=cv2.contourArea)
                        ((x_red1, y_red1), radius_red1) = cv2.minEnclosingCircle(c_red1)
                        M_red1 = cv2.moments(c_red1)

                        # compute the center of the contour
                        cx_red1 = int(M_red1["m10"] / M_red1["m00"])
                        cy_red1 = int(M_red1["m01"] / M_red1["m00"])
                        

                        # cy_red is set in the region of interest, so need to adjust for origin in frame
                        cy_red1 = cy_red1 + y1
                        # center = ( cx_red, cy_red )

                        # only proceed if the radius meets a minimum size
                        if radius_red1 > 5:
                            newred1lConfidence = 100
                            # draw the circle and centroid on the frame
                            cv2.circle(warped, (cx_red1, cy_red1), int(radius_red1),
                            (0, 0, 255), 2)
                            # cv2.circle(frame, center, 5, (0, 0, 255), -1)

                            # calculate offset
                            xRed1 = cx_red1-(width/2)   # Contrived so pstve to right of centreline
                            yRed1 = height - cy_red1  # Adjust to make origin bottom centre of image

                            # The target has been found, so we can break out of the loop here
                            break

                    # But here the target has not been found, we need to move the ROI down
                    y1 = y1 + roidepth
                    y2 = y1 + roidepth

                # And here we have either hit the buffers or found the target.


                if (newred1lConfidence == 100 and newRed2Confidence == 100):
                    # Calculate gradient and intercept for thes valid pair of points.
                    # The aspect ratio of warped image is wrong (should be long and thin!).
                    #Rather than stretch the image (processing time), we simply apply a multiple to the
                    #y axis.
                    aspect = 3.0
                
                    dy = yRed1 - yRed2
                    dx = xRed1 - xRed2
                    m = float(dx)/(aspect*float(dy))
                    intc = xRed2 - (m * float(yRed2))

                                         
                    # Add to the running average for each.
                    # Update gradient array and calculate running average to return as target gradient
                    # 
                    gradRed[4] = gradRed[3]
                    gradRed[3] = gradRed[2]
                    gradRed[2] = gradRed[1]
                    gradRed[1] = gradRed[0]
                    gradRed[0] = m
                    # Update the gradient for the bearing signal from the last 5 on a running average
                    m = (gradRed[0]+gradRed[1]+gradRed[2]+gradRed[3]+gradRed[4])/5


                    # Update intercept array and calculate running average to return as target intercept
                    intRed[4] = intRed[3]
                    intRed[3] = intRed[2]
                    intRed[2] = intRed[1]
                    intRed[1] = intRed[0]
                    intRed[0] = intc
                    # Update the x axis intercept for the offset signal from the last 5 on a running average
                    intc = (intRed[0]+intRed[1]+intRed[2]+intRed[3]+intRed[4])/5


                # The confidence running averages are updated whether the lock was successful or not
                # Update confidence array for lower roi
                red1lConfidence[4] = red1lConfidence[3]
                red1lConfidence[3] = red1lConfidence[2]
                red1lConfidence[2] = red1lConfidence[1]
                red1lConfidence[1] = red1lConfidence[0]
                red1lConfidence[0] = newred1lConfidence
                newred1lConfidence = (red1lConfidence[0]+red1lConfidence[1]+red1lConfidence[2]+red1lConfidence[3]+red1lConfidence[4])/5


                # Update confidence array for upper roi
                red2Confidence[4] = red2Confidence[3]
                red2Confidence[3] = red2Confidence[2]
                red2Confidence[2] = red2Confidence[1]
                red2Confidence[1] = red2Confidence[0]
                red2Confidence[0] = newRed2Confidence
                newRed2Confidence = (red2Confidence[0]+red2Confidence[1]+red2Confidence[2]+red2Confidence[3]+red2Confidence[4])/5


                # In following mode, we must have lock on both rois.  So red1Good and red2Good = True
                # Now to calculate signals to be returned, normalised between -1 and 1.
                if (newred1lConfidence > 60) and newRed2Confidence > 60:
                    red1Good = red2Good = True
                    offset = intc / (width/2) # This gives use the abiity to respond to values off the camera at y=0 !(so beyond 1)     
                    bearing = np.degrees(np.arctan(m)) # To move towards the target.    Bearing is in degrees.
                    imagescale = int(np.absolute(bearing)) # This is used to modulate pitch, so must always be positive
                    if imagescale > 35:
                        imagescale = 35
                else:
                    imagescale = 35 # We have lost lock, so this sets the image width to maximum .


                # Draw Region of interest
                #cv2.line(warped, (0, y1), (width, y1), (255,0,0))
                #cv2.line(warped, (0, y2), (width, y2), (255,0,0))
                #cv2.line(warped, (0, y3), (width, y3), (255,0,0))
                #cv2.line(warped, (0, y4), (width, y4), (255,0,0))

                # print "Following", " Bearing: ",bearing, red1Good, "  Offset: ", offset, red2Good

                # cv2.imshow('Frame',frame)
                cv2.imshow("Warped", warped)
                
                #    cv2.imwrite('/home/pi/images/image'+ (time.strftime("%H:%M:%S"))+'.jpg', frame)
                key = cv2.waitKey(1) & 0xFF
          
		return (bearing, red1Good, offset, red2Good)


	def readlost(self):

                global red1lConfidence
                global gradlRed
                global red2lConfidence
                global intlRed
                global bearing
                global M

                # Set the image resolution.
                xres = 320
                yres = 240
                xColour1 = xRed = 0.0

                # Initialise confidence to indicate the line has not been located with the current frame
                newred1lConfidence = 0
                redl1Good = False
                newred2lConfidence = 0
                redl2Good = False

                # Initialise variables for line calculations
                xRed1 = xRed2 = 0
                yRed1 = yRed2 = 0
                intc = m = 0.0
                dx = dy = 0
                bearing = offset = 0.0
             

                # return the frame most recently read
                frame = self.frame

                # apply the four point tranform to obtain a "birds eye view" of
                # the image.  We are mapping this onto an extended image to get the broadest horizon.
	        ewarped = cv2.warpPerspective(frame, M, (600, 268))
                height, width, channels = ewarped.shape
                # height = 240
                # width = 320

                # Set y coords of regions of interest.
                # The upper and lower bounds
                roidepth = 20    # vertical depth of regions of interest
                roiymin = 40    # minumum ranging y value for roi origin
                roiymintop = roiymin - roidepth
                roiymax = height - roidepth -1   # maximum ranging y value for bottom roi origin

                
                # Convert to hsv and define region of interest before further processing.
                fullhsv = cv2.cvtColor(ewarped, cv2.COLOR_BGR2HSV)

                # green = 60
                # blue = 120;
                # yellow = 30;
                #Colour1 = 60

                # Set the sensitivity of the hue
                sensitivity = 20

                # Red is a special case as it sits either side of 0 on the HSV spectrum
                # So we create two masks, one greater than zero and one less than zero
                # Then combine the two.
                lower_red_0 = np.array([0, 100, 100]) 
                upper_red_0 = np.array([sensitivity, 255, 255])
                
                lower_red_1 = np.array([180 - sensitivity, 100, 100]) 
                upper_red_1 = np.array([180, 255, 255])


                # Initialise the bottom roi at the maximum limit
                y3 = roiymax
                y4 = y3 + roidepth
                xRed2 = yRed2 = 0 # Anchor the first point at the origin.  We will seek the target with the upper roi.

                """
                while y3 > roiymin:

                    # This defines the lower band, looking closer in
                    roihsv2 = fullhsv[y3:y4, 0:(width-1)]

                    # Prepare the masks for the lower roi 
                    maskr_2 = preparemask (roihsv2, lower_red_0 , upper_red_0)
                    maskr_3 = preparemask (roihsv2, lower_red_1 , upper_red_1 )
                    maskr2 = meldmask ( maskr_2, maskr_3)

                    # find contours in the lower roi and initialize the center
                    cnts_red2 = cv2.findContours(maskr2.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
                    center2 = None

                    # Now to find the tracking line in the lower roi
                    # only proceed if at least one contour was found
                    if len(cnts_red2) > 0:
                        
                        # find the largest contour in the mask, then use
                        # it to compute the minimum enclosing circle and
                        # centroid
                        c_red2 = max(cnts_red2, key=cv2.contourArea)
                        ((x_red2, y_red2), radius_red2) = cv2.minEnclosingCircle(c_red2)
                        M_red2 = cv2.moments(c_red2)

                        # compute the center of the contour
                        cx_red2 = int(M_red2["m10"] / M_red2["m00"])
                        cy_red2 = int(M_red2["m01"] / M_red2["m00"])
                        

                        # cy_red is set in the region of interest, so need to adjust for origin in frame
                        cy_red2 = cy_red2 + y3
                        # center = ( cx_red, cy_red )

                        # only proceed if the radius meets a minimum size
                        if radius_red2 > 5:
                            newred2lConfidence = 100
                            # draw the circle and centroid on the frame
                            cv2.circle(ewarped, (cx_red2, cy_red2), int(radius_red2),
                            (0, 0, 255), 2)
                            # cv2.circle(frame, center, 5, (0, 0, 255), -1)

                            # calculate offset
                            xRed2 = cx_red2 - (width/2) # Contrived so pstve to right of centreline
                            yRed2 = height - cy_red2  # Adjust to make origin bottom centre of image

                            # The target has been found, so we can break out of the loop here
                            break

                    # But here the target has not been found, we need to move the ROI up
                    y3 = y3 - roidepth
                    y4 = y3 + roidepth

                # And here we have either hit the buffers or found the target.
                """
                
                # So now try for the top roi, working down.                      
                # Initialise the top roi at the very top
                y1 = 0
                y2 = y1 + roidepth

                while y2 < y3: # Go as far as the lower roi but no more.

                    newred1lConfidence = 0

                    # This defines the upper roi, looking further away
                    roihsv1 = fullhsv[y1:y2, 0:(width-1)]                 

                    # Prepare the masks for the top roi 
                    maskr_0 = preparemask (roihsv1, lower_red_0 , upper_red_0)
                    maskr_1 = preparemask (roihsv1, lower_red_1 , upper_red_1 )
                    maskr1 = meldmask ( maskr_0, maskr_1)

                    # find contours in the upper roi and initialize the center
                    cnts_red1 = cv2.findContours(maskr1.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
                    center1 = None

                    # Now to find the tracking line in the upper roi
                    # only proceed if at least one contour was found
                    if len(cnts_red1) > 0:
                        
                        # find the largest contour in the mask, then use
                        # it to compute the minimum enclosing circle and
                        # centroid
                        c_red1 = max(cnts_red1, key=cv2.contourArea)
                        ((x_red1, y_red1), radius_red1) = cv2.minEnclosingCircle(c_red1)
                        M_red1 = cv2.moments(c_red1)

                        # compute the center of the contour
                        cx_red1 = int(M_red1["m10"] / M_red1["m00"])
                        cy_red1 = int(M_red1["m01"] / M_red1["m00"])
                        

                        # cy_red is set in the region of interest, so need to adjust for origin in frame
                        cy_red1 = cy_red1 + y1
                        # center = ( cx_red, cy_red )

                        # only proceed if the radius meets a minimum size
                        if radius_red1 > 5:
                            newred1lConfidence = 100
                            # draw the circle and centroid on the frame
                            cv2.circle(ewarped, (cx_red1, cy_red1), int(radius_red1),
                            (0, 0, 255), 2)
                            # cv2.circle(frame, center, 5, (0, 0, 255), -1)

                            # calculate offset
                            xRed1 = cx_red1-(width/3)   # Contrived so pstve to right of centreline
                            yRed1 = height - cy_red1  # Adjust to make origin bottom centre of image

                            # The target has been found, so we can break out of the loop here
                            break

                    # But here the target has not been found, we need to move the ROI down
                    y1 = y1 + roidepth
                    y2 = y1 + roidepth

                # And here we have either hit the buffers or found the target.


                if newred1lConfidence == 100 : # So the bottom end is already set and anchored at the origin.
                    # Calculate gradient and intercept for thes valid pair of points.
                    # The aspect ratio of warped image is wrong (should be long and thin!).
                    # Rather than stretch the image (processing time), we simply apply a multiple to the
                    # y axis.
                    aspect = 3.0
                
                    dy = yRed1 - yRed2
                    dx = xRed1 - xRed2
                    m = float(dx)/(aspect*float(dy))
                    intc = 0 # This is always the case because the lower roi target is locked to the origin
                                         
                    # Add to the running average for each.
                    # Update gradient array and calculate running average to return as target gradient
                    # 
                    gradlRed[4] = gradlRed[3]
                    gradlRed[3] = gradlRed[2]
                    gradlRed[2] = gradlRed[1]
                    gradlRed[1] = gradlRed[0]
                    gradlRed[0] = m
                    # Update the gradient for the bearing signal from the last 5 on a running average
                    m = (gradlRed[0]+gradlRed[1]+gradlRed[2]+gradlRed[3]+gradlRed[4])/5

                    """
                    # Update intercept array and calculate running average to return as target intercept
                    intlRed[4] = intlRed[3]
                    intlRed[3] = intlRed[2]
                    intlRed[2] = intlRed[1]
                    intlRed[1] = intlRed[0]
                    intlRed[0] = intc
                    # Update the x axis intercept for the offset signal from the last 5 on a running average
                    intc = (intlRed[0]+intlRed[1]+intlRed[2]+intlRed[3]+intlRed[4])/5
                    """

                # The confidence running averages are updated whether the lock was successful or not
                # Update confidence array and calculate average to decide whether to set redl1Good
                red1lConfidence[4] = red1lConfidence[3]
                red1lConfidence[3] = red1lConfidence[2]
                red1lConfidence[2] = red1lConfidence[1]
                red1lConfidence[1] = red1lConfidence[0]
                red1lConfidence[0] = newred1lConfidence
                newred1lConfidence = (red1lConfidence[0]+red1lConfidence[1]+red1lConfidence[2]+red1lConfidence[3]+red1lConfidence[4])/5
                
            
                """
                # Update confidence array and calculate average
                red2lConfidence[4] = red2lConfidence[3]
                red2lConfidence[3] = red2lConfidence[2]
                red2lConfidence[2] = red2lConfidence[1]
                red2lConfidence[1] = red2lConfidence[0]
                red2lConfidence[0] = newred2lConfidence
                newred2lConfidence = (red2lConfidence[0]+red2lConfidence[1]+red2lConfidence[2]+red2lConfidence[3]+red2lConfidence[4])/5
                """

                # Now to calculate signals to be returned, normalised between -1 and 1.
                if newred1lConfidence > 60:
                    redl1Good = True
                    offset = intc # Recall intc locked to zero, so no offset here.
                    bearing = np.degrees(np.arctan(m)) # To move towards the target.  Bearing is in degrees.


                # Draw Region of interest
                cv2.line(ewarped, (0, y1), (width, y1), (255,0,0))
                cv2.line(ewarped, (0, y2), (width, y2), (255,0,0))
                cv2.line(ewarped, (0, y3), (width, y3), (255,0,0))
                cv2.line(ewarped, (0, y4), (width, y4), (255,0,0))

                # print "Lost", bearing, red1Good, offset, red2Good

                #cv2.imshow('Frame',frame)
                # cv2.imshow("Extended Warped", ewarped)
                
                #    cv2.imwrite('/home/pi/images/image'+ (time.strftime("%H:%M:%S"))+'.jpg', frame)
                key = cv2.waitKey(1) & 0xFF
          
		return (bearing, redl1Good, offset, redl2Good)

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True


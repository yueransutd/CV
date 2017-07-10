import imutils
import cv2
 
cap = cv2.VideoCapture(0)


# initialize the first frame
firstFrame = None

#get width and height of webcam
width = cap.get(3)  # float
height = cap.get(4) # float
#print width,height
#width=640.0, height=480.0


while (1):
	
    grabbed, frame = cap.read()
	
    if not grabbed:
	 	 break
 
	# resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    
    
 
    if firstFrame is None:
        firstFrame = gray
        continue
    
    
    
    #the absolute difference between the current frame and the first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    im2,cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(frame,cnts,-1,(255,0,0),3)
    
    res = cv2.bitwise_and(frame, frame, mask=thresh)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    eroded = cv2.erode(frame,kernel)
    
    # loop over the contours
    for c in cnts:
		# if the contour is too small, ignore it
        if cv2.contourArea(c) < width*height*0.03:
            continue
 
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
        
     
	
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("res",res)
    cv2.imshow("Eroded",eroded)
    key = cv2.waitKey(1) & 0xFF
 
	
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break
 
cap.release()
cv2.destroyAllWindows()




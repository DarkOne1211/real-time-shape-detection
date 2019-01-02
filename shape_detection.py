# REPLACE CONTOUR MAPPING WITH KMEANS CLUSTERING
# WARNING THIS IS NOT THE FINAL DESIGN. THIS USES CONTOUR MAPPING
import cv2 as cv
import numpy as np 
import imutils

def real_time_shape(show):
    # VIDEO CAPTURE
    cap_video = cv.VideoCapture(0)

    # RUNS FOREVER
    while(1):
        _,frame = cap_video.read()

        # CANNY EDGE DETECTION
        frameG = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
        edges = cv.Canny(frameG,200,200)
        thresh = cv.threshold(frameG,127,255,cv.THRESH_BINARY)[1]
        # CALLING SHAPE DETECTION FUNCTION
        shapes = shapeDetector(thresh,frame.copy())
        if (show):
            # DISPLAY ORIGINAL
            cv.imshow('Original Image',frame)

            # DISPLAY CANNY OUTPUT
            cv.imshow('Edges',edges)

            # DISPLAY THRESH OUTPUT
            cv.imshow('Threshold',thresh)

            # DISPLAY SHAPE OUTPUT
            cv.imshow('Shapes',shapes)
        cv.waitKey(5)

    cap_video.release()
    cv.destroyAllWindows()

def shapeDetector(image,origimage):
    
    # RESIZING THE IMAGE
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    
    # SETTING A THRESHOLD TO CONVERT IT TO BLACK AND WHITE

    # FINDING CONTOURS IN THE B/W IMAGE
    contours = cv.findContours(resized.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)[1]

    for cntour in contours:
        # CALCULATING THE CENTERgray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        shape = detect(cntour)
        M = cv.moments(cntour)
        if (M["m00"] == 0):
            cX = 0
            cY = 0
        else:
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
        cntour = cntour.astype("float")
        cntour *= ratio
        cntour = cntour.astype("int")
        cv.drawContours(origimage,[cntour],-1,(34,0,156),2)
        cv.putText(origimage, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX,
		0.5, (255, 255, 255), 2)
    return(origimage)

# NEEDS TO BE REPLACED BY K MEANS CLUSTERING INSTEAD OF CONTOUR MAPPING

def detect(c):
    shape = "unidentified"
    peri = cv.arcLength(c,True)
    approx = cv.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 3:
    	shape = "triangle"
 
    elif len(approx) == 4:
    	(_, _, w, h) = cv.boundingRect(approx)
    	ar = w / float(h)
    	shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    elif len(approx) == 5:
    	shape = "pentagon"

    else:
    	shape = "circle"

    return shape


if __name__ == "__main__":
    real_time_shape(1)
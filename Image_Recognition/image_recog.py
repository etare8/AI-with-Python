import cv2
import numpy as np
#import install_needs

# Load the image to analize
img_load = cv2.imread(r'.\fiber_analysis\img_mu1.png')

# Convert to grayscale and apply thresholding
# 
img_grey = cv2.cvtColor(img_load, cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours: take the input image and returns a list of all possible contours and their hierarchy
contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours and filter out non-ellipsoidal ones
selected_contours = []
for contour in contours:
    # Take each single contour and evaluate the area (number of squared pixels--->this must be gauged depending on the image pixels and pixel density)
    # A thorough description should include the evaluation of the input image real dimensions, evaluation of pixel density, and an estimation of the areas
    area = cv2.contourArea(contour)
    if area > 10000:
        #print(area) necessary to look at the contours and select the proper value to exclude small contours
        perimeter = cv2.arcLength(contour, closed=True) # lenght of a contour / closed=True means it is closed contour
        circularity = 4 * np.pi * area / (perimeter**2) # for the circle is = 1, must be gauged looking at images

        if circularity > 0.3:
            # Bounding rectangle is the smallest rectangle that can completely enclose the contour
            # We use this parameter to evaluate if the ratio of the object is at least within 1:2 and 2:1 (we can extend this to detect very "sharp" elements)
            x, y, width, height = cv2.boundingRect(contour) 
            # Ratio / Rapporto di immagine
            aspect_ratio = float(width)/height
            if aspect_ratio > 0.25 and aspect_ratio < 2.5:
                selected_contours.append(contour)

# Count the number of detected ellipsoidal elements
ellips_count = len(selected_contours)

print("\nNumber of Fibers:", ellips_count, "\n")

#----------------------------------------------------------------------------
# PLOTTING:

# Resize the image (maybe to large in terms of pixels)
img_resized = cv2.resize(img_load, (400,400))

# Scaling the contours that are associated to the real image and not to the resized one
# img.shape returns (0,1,2) meaning: 0) the height in pixels of the image
#                                    1) the width in pixels
#                                    2) the # of channels (in this case 3 since the img_load is in RGB or GBR)
scale_x = 400 / img_load.shape[1]
scale_y = 400 / img_load.shape[0]

# Scale the contours
scaled_contours = []
for circle in selected_contours:
    scaled_contour = np.zeros_like(circle)
    
    # Pay attention at the shape of a contour: [[[a,b]], [[a,b]],...,[[a,b]]], therefore [:,0,0] means that for all the elements [[a,b]] we select [a,b], specifically "a"
    scaled_contour[:, 0, 0] = circle[:, 0, 0] * scale_x
    scaled_contour[:, 0, 1] = circle[:, 0, 1] * scale_y
    scaled_contours.append(scaled_contour)

# Plotting Images:
# Contours detected and resized
cv2.drawContours(img_resized, scaled_contours, contourIdx=-1, color=(255, 0, 0), thickness=2) #BGR!!
cv2.imshow("Analized Image",img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
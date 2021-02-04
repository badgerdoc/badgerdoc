import cv2
import numpy as np


def get_table_contours(img):
    # thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # inverting the image
    img_bin = 255 - img_bin

    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1]//100
    # Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))


    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)


    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    #Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh,110,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bitxor = cv2.bitwise_xor(img,img_vh)
    bitnot = cv2.bitwise_not(bitxor)

    return cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def contours_to_boxes(contours, area_threshold):
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < area_threshold:
            continue
        # image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        boxes.append([x,y,w,h])
    return boxes


def draw_boxes(img, boxes, origin=(0, 0), color=(0,255,0), stroke=2):
    new_img = img.copy()
    for box in boxes:
        x, y, w, h = box
        x += origin[0]
        y += origin[1]
        cv2.rectangle(new_img,(x,y),(x+w,y+h),color,stroke)
    return new_img


def recognize_bordered_table(img, threshold):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if threshold > 1:
        raise ValueError('Threshold should be a value between 0 and 1')
    img_area = img.shape[0] * img.shape[1]
    contours, hierarchy = get_table_contours(img_gray)

    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    boxes = contours_to_boxes(contours, img_area * threshold)
    return boxes


if __name__ == '__main__':
    # read your file
    file = r'images_2/49.PNG'
    img = cv2.imread(file)

    boxes = recognize_bordered_table(img, 0.0001)
    cv2.imshow('i', draw_boxes(img, boxes))
    cv2.waitKey(0)


# #Creating a list of heights for all detected boxes
# heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
# #Get mean of heights
# mean = np.mean(heights)



# #Creating two lists to define row and column in which cell is located
# row=[]
# column=[]
# j=0
# #Sorting the boxes to their respective row and column
# for i in range(len(box)):
#     if(i==0):
#         column.append(box[i])
#         previous=box[i]
#     else:
#         if(box[i][1]<=previous[1]+mean/2):
#             column.append(box[i])
#             previous=box[i]
#             if(i==len(box)-1):
#                 row.append(column)
#         else:
#             row.append(column)
#             column=[]
#             previous = box[i]
#             column.append(box[i])
# print(column)
# print(row)



# #calculating maximum number of cells
# countcol = 0
# for i in range(len(row)):
#     countcol = len(row[i])
#     if countcol > countcol:
#         countcol = countcol


# #Retrieving the center of each column
# center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]
# center=np.array(center)
# center.sort()



# #Regarding the distance to the columns center, the boxes are arranged in respective order
# finalboxes = []
# for i in range(len(row)):
#     lis=[]
#     for k in range(countcol):
#         lis.append([])
#     for j in range(len(row[i])):
#         diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
#         minimum = min(diff)
#         indexing = list(diff).index(minimum)
#         lis[indexing].append(row[i][j])
#     finalboxes.append(lis)


# cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
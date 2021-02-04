import cv2


def remove_color(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img[150 < gray_img] = 255
    return gray_img


if __name__ == '__main__':
    file = 'images/1.png'
    img = cv2.imread(file)
    img = remove_color(img)
    cv2.imwrite('images/clean_1.png', img)
    cv2.imshow('Image', img)
    cv2.waitKey(0)

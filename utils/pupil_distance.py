import math

def get_pupil_distance(img_land):
    leftEye0 = (img_land[2 * 37] + img_land[2 * 38] + img_land[2 * 39] + img_land[2 * 40] + img_land[2 * 41] +
                img_land[2 * 36]) / 6.0
    leftEye1 = (img_land[2 * 37 + 1] + img_land[2 * 38 + 1] + img_land[2 * 39 + 1] + img_land[2 * 40 + 1] +
                img_land[2 * 41 + 1] + img_land[2 * 36 + 1]) / 6.0
    rightEye0 = (img_land[2 * 43] + img_land[2 * 44] + img_land[2 * 45] + img_land[2 * 46] + img_land[2 * 47] +
                 img_land[2 * 42]) / 6.0
    rightEye1 = (img_land[2 * 43 + 1] + img_land[2 * 44 + 1] + img_land[2 * 45 + 1] + img_land[2 * 46 + 1] +
                 img_land[2 * 47 + 1] + img_land[2 * 42 + 1]) / 6.0
    deltaX = float(rightEye0 - leftEye0)
    deltaY = float(rightEye1 - leftEye1)
    l = math.sqrt(deltaX * deltaX + deltaY * deltaY)
    #print("get_pupil_distance", l)
    return l

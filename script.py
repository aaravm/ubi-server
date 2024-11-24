import cv2
import numpy as np
import easyocr
import math

def map_angle_to_speed(angle):
    min_angle = -30
    max_angle = 210
    min_speed = 0
    max_speed = 240

    if angle < min_angle:
        angle = min_angle
    elif angle > max_angle:
        angle = max_angle

    speed = ((angle - min_angle) / (max_angle - min_angle)) * (max_speed - min_speed) + min_speed
    return speed

def map_angle_to_rpm(angle):
    min_angle = -30
    max_angle = 210
    min_rpm = 0
    max_rpm = 6

    if angle < min_angle:
        angle = min_angle
    elif angle > max_angle:
        angle = max_angle

    rpm = ((angle - min_angle) / (max_angle - min_angle)) * (max_rpm - min_rpm) + min_rpm
    return rpm

def posToAngle(x_center,y_center,x_end,y_end):
  if x_end == x_center:
    return 90
  elif x_end>x_center:
    if y_end>y_center:
      return 180 + math.degrees(math.atan((y_end-y_center)/(x_end-x_center)))
    else :
      return 180 - math.degrees(math.atan((y_center-y_end)/(x_end-x_center)))
  else :
    if y_end>y_center:
      return -math.degrees(math.atan((y_end-y_center)/(x_center-x_end)))
    else :
      return math.degrees(math.atan((y_center-y_end)/(x_center-x_end)))


# Initialize ORB detector
orb = cv2.ORB_create()
base_image_path = './base.jpg'
new_image_path = './new1.jpg'

base_image = cv2.imread(base_image_path, cv2.IMREAD_COLOR)
new_image = cv2.imread(new_image_path, cv2.IMREAD_COLOR)
# Detect keypoints and descriptors
keypoints1, descriptors1 = orb.detectAndCompute(base_image, None)
keypoints2, descriptors2 = orb.detectAndCompute(new_image, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(descriptors1, descriptors2)

matches = sorted(matches, key=lambda x: x.distance)

points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
points2 = np.array([keypoints2[m.trainIdx].pt for m in matches])

H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

height, width = base_image.shape[:2]

aligned_image = cv2.warpPerspective(new_image, H, (width, height))

rpm_meter = aligned_image[350:650,:340]

speed_meter = aligned_image[350:650,600:940]

right_indicator = aligned_image[280:375,600:700]

left_indicator = aligned_image[280:375,260:350]

info_board = aligned_image[350:550,350:600]

emergency = aligned_image[560:620,400:490]

seat_belt = aligned_image[560:620,480:530]

# Convert the image to HSV color space
img = rpm_meter

# Apply simple thresholding
ret, thresh1 = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)

# Display the result
hsv = cv2.cvtColor(thresh1, cv2.COLOR_BGR2HSV)

# Define the range of red color in HSV
lower_red = np.array([0,100,100])
upper_red = np.array([10,255,255])

# Threshold the HSV image to get only red colors
mask1 = cv2.inRange(hsv, lower_red, upper_red)

# For reddish purple colors
lower_red = np.array([170,100,100])
upper_red = np.array([180,255,255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)

# Combine both masks
mask = mask1 + mask2

# Bitwise-AND mask and original image
res = cv2.bitwise_and(thresh1, thresh1, mask=mask)

gray_1 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
edges_1 = cv2.Canny(gray_1, 100, 250, apertureSize=3)
# Use Hough Line Transform to detect lines in the image
lines = cv2.HoughLinesP(edges_1, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)
x_center,y_center = (150,150)
current_rpm = 0
if lines is not None:
    min_line_length = 20

    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_length = np.sqrt((x2 - x1) * (x2 - x1)  + (y2 - y1) * (y2 - y1))

        if((x1-x_center)**2+(y1-y_center)**2>(x2-x_center)**2+(y2-y_center)**2):
          x_end = x1
          y_end = y1
          x_cen = x2
          y_cen = y2
        else:
          x_end = x2
          y_end = y2
          x_cen = x1
          y_cen = y1

        if abs(x_cen - x_center)>20:
          continue

        if line_length >= min_line_length:
            cv2.line(res, (x1, y1), (x2, y2), (0, 255, 0), 2)
            dx = x2 - x1
            dy = y2 - y1
            # angle_radians = math.atan2(dy, dx)
            angle_degrees = posToAngle(x_cen,y_cen,x_end,y_end)
            current_rpm = map_angle_to_rpm(angle_degrees)*1000
    cv2.imshow("res",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
   print("Not found")

gray_ind_left = cv2.cvtColor(left_indicator, cv2.COLOR_BGR2GRAY)
mean_intensity = np.mean(gray_ind_left)
left_indicator_on = mean_intensity>40

gray_ind_right = cv2.cvtColor(right_indicator, cv2.COLOR_BGR2GRAY)
mean_intensity = np.mean(gray_ind_right)
right_indicator_on = mean_intensity>40

emergency_gray = cv2.cvtColor(emergency, cv2.COLOR_BGR2GRAY)
mean_intensity = np.mean(emergency_gray)
emergency_on = mean_intensity>40

seatbelt_gray = cv2.cvtColor(seat_belt, cv2.COLOR_BGR2GRAY)
mean_intensity = np.mean(seatbelt_gray)
seat_belt_on = mean_intensity>40

top_part = info_board[30:60,:]
bottom_part_left = info_board[150:,:150]
bottom_part_right = info_board[150:,150:]

reader = easyocr.Reader(['en'])
last_fuel = reader.readtext(top_part)
temp = reader.readtext(bottom_part_left)
dist = reader.readtext(bottom_part_right)

print(current_rpm)
print(left_indicator_on)
print(right_indicator_on)
print(temp)
print(last_fuel)
print(dist)



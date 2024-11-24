import cv2
import numpy as np
import easyocr
import math
from flask import Flask, request, jsonify

app = Flask(__name__)

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

def posToAngle(x_center, y_center, x_end, y_end):
    if x_end == x_center:
        return 90
    elif x_end > x_center:
        if y_end > y_center:
            return 180 + math.degrees(math.atan((y_end - y_center) / (x_end - x_center)))
        else:
            return 180 - math.degrees(math.atan((y_center - y_end) / (x_end - x_center)))
    else:
        if y_end > y_center:
            return -math.degrees(math.atan((y_end - y_center) / (x_center - x_end)))
        else:
            return math.degrees(math.atan((y_center - y_end) / (x_center - x_end)))

@app.route('/process-dashboard', methods=['POST'])
def process_dashboard():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    new_image_path = "./uploaded_image.jpg"
    base_image_path = './base.jpg'
    file.save(new_image_path)

    # Load the base and new images
    base_image = cv2.imread(base_image_path, cv2.IMREAD_COLOR)
    new_image = cv2.imread(new_image_path, cv2.IMREAD_COLOR)

    # Initialize ORB detector and match keypoints
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(base_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(new_image, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches])
    H, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    height, width = base_image.shape[:2]
    aligned_image = cv2.warpPerspective(new_image, H, (width, height))

    # Crop required regions
    rpm_meter = aligned_image[350:650, :340]
    speed_meter = aligned_image[350:650, 600:940]
    left_indicator = aligned_image[280:375, 260:350]
    right_indicator = aligned_image[280:375, 600:700]
    emergency = aligned_image[560:620, 400:490]
    seat_belt = aligned_image[560:620, 480:530]
    info_board = aligned_image[350:550, 350:600]

    # Process RPM and speed
    current_rpm = extract_rpm_value(rpm_meter)
    current_speed = extract_speed_value(speed_meter)

    # Process indicators and warnings
    left_indicator_on = detect_intensity(left_indicator)
    right_indicator_on = detect_intensity(right_indicator)
    emergency_on = detect_intensity(emergency)
    seat_belt_on = detect_intensity(seat_belt)

    # Process info board
    reader = easyocr.Reader(['en'])
    last_fuel = reader.readtext(info_board[30:60, :], detail=0)
    temp = reader.readtext(info_board[150:, :150], detail=0)
    dist = reader.readtext(info_board[150:, 150:], detail=0)

    # Return results
    results = {
        "current_rpm": str(current_rpm),
        "current_speed": str(current_speed),
        "left_indicator_on": str(left_indicator_on),
        "right_indicator_on": str(right_indicator_on),
        "emergency_on": str(emergency_on),
        "seat_belt_on": str(seat_belt_on),
        "last_fuel": ' '.join(last_fuel),
        "temp": ' '.join(temp),
        "dist": ' '.join(dist)
    }

    return jsonify(results)

def extract_rpm_value(meter_image):
    img = meter_image

    # Apply simple thresholding
    ret, thresh1 = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)

    # Display the result
    hsv = cv2.cvtColor(thresh1, cv2.COLOR_BGR2HSV)

    # Define the range of red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # For reddish purple colors
    lower_red = np.array([170, 100, 100])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine both masks
    mask = mask1 + mask2

    # Bitwise-AND mask and original image
    res1 = cv2.bitwise_and(thresh1, thresh1, mask=mask)
    gray_1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
    edges_1 = cv2.Canny(gray_1, 100, 250, apertureSize=3)

    # Use Hough Line Transform to detect lines in the image
    lines = cv2.HoughLinesP(edges_1, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)
    x_center, y_center = (150, 150)
    current_rpm = 0

    if lines is not None:
        min_line_length = 30

        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if (x1 - x_center) ** 2 + (y1 - y_center) ** 2 > (x2 - x_center) ** 2 + (y2 - y_center) ** 2:
                x_end = x1
                y_end = y1
                x_cen = x2
                y_cen = y2
            else:
                x_end = x2
                y_end = y2
                x_cen = x1
                y_cen = y1

            if abs(x_cen - x_center) > 20:
                continue

            if line_length >= min_line_length:
                cv2.line(res1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                angle_degrees = posToAngle(x_cen, y_cen, x_end, y_end)
                current_rpm = map_angle_to_rpm(angle_degrees)*1000
                
    return current_rpm

def extract_speed_value(meter_image):
    img = meter_image

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
    res2 = cv2.bitwise_and(thresh1, thresh1, mask=mask)

    gray_1 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    edges_1 = cv2.Canny(gray_1, 100, 250, apertureSize=3)
    # Use Hough Line Transform to detect lines in the image
    lines = cv2.HoughLinesP(edges_1, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)
    x_center,y_center = (150,150)
    current_speed = 0

    if lines is not None:
        min_line_length = 30

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
                cv2.line(res2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                dx = x2 - x1
                dy = y2 - y1
                # angle_radians = math.atan2(dy, dx)
                angle_degrees = posToAngle(x_cen,y_cen,x_end,y_end)
                current_speed = map_angle_to_speed(angle_degrees)

    return current_speed            

def detect_intensity(region):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) > 40

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

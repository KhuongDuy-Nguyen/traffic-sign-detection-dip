import cv2
import numpy as np
import os

def get_color_ranges():
    return {
        "Red": [
            (np.array([0, 100, 100]), np.array([5, 255, 255])),
            (np.array([170, 100, 100]), np.array([180, 255, 255])),
            (np.array([140, 50, 50]), np.array([170, 255, 255]))
        ],
        "Blue": [
            (np.array([105, 100, 100]), np.array([115, 255, 255]))
        ],
        "White": [
            (np.array([0, 0, 180]), np.array([180, 50, 255]))
        ],
        "Yellow": [
            (np.array([15, 80, 80]), np.array([40, 255, 255]))
        ]
    }

def check_triangle_colors(hsv, contour, color_ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255), -1)

    border_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(border_mask, [contour], -1, (255), 2)

    inner_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    eroded_contour = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    inner_area = cv2.bitwise_and(mask, eroded_contour)

    red_border_ratio = 0
    for lower, upper in color_ranges["Red"]:
        red_mask = cv2.inRange(hsv, lower, upper)
        red_border = cv2.bitwise_and(red_mask, border_mask)
        red_border_ratio += np.sum(red_border > 0) / (np.sum(border_mask > 0) + 1e-6)

    yellow_inner_ratio = 0
    for lower, upper in color_ranges["Yellow"]:
        yellow_mask = cv2.inRange(hsv, lower, upper)
        yellow_inner = cv2.bitwise_and(yellow_mask, inner_area)
        yellow_inner_ratio = np.sum(yellow_inner > 0) / (np.sum(inner_area > 0) + 1e-6)

    return red_border_ratio > 0.05 and yellow_inner_ratio > 0.5

def preprocess_mask(mask, kernel):
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def detect_triangle_shape(contour):
    area = cv2.contourArea(contour)
    if area < 100:
        return False

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

    if len(approx) != 3:
        return False

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if not (0.7 <= aspect_ratio <= 1.3):
        return False

    sides = []
    for i in range(3):
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % 3][0]
        side_length = np.linalg.norm(pt1 - pt2)
        sides.append(side_length)

    avg_side = sum(sides) / 3
    side_ratios = [side / avg_side for side in sides]
    if not all(0.7 <= ratio <= 1.3 for ratio in side_ratios):
        return False

    return True

def check_position(y, h, frame_height):
    relative_y = y / frame_height
    if relative_y > 0.6:
        return False
    relative_size = h / frame_height
    if relative_size < 0.05 or relative_size > 0.3:
        return False
    return True

def detect_signs(hsv, color_ranges, kernel):
    results = []
    frame_height = hsv.shape[0]

    for color, ranges in [("Red", color_ranges["Red"]), ("Blue", color_ranges["Blue"])]:
        mask = None
        for lower, upper in ranges:
            color_mask = cv2.inRange(hsv, lower, upper)
            mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)

        mask = preprocess_mask(mask, kernel)
        edges = cv2.Canny(mask, 100, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 180:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if not check_position(y, h, frame_height):
                continue

            ratio = w / h
            if 0.8 <= ratio <= 1.2:
                circularity = (4 * np.pi * area) / (cv2.arcLength(contour, True) ** 2)
                if 0.8 <= circularity <= 1.2:
                    results.append((f"{color} Circle", contour, (x, y, w, h)))

    combined_mask = None
    for color, ranges in [("Yellow", color_ranges["Yellow"]), ("Red", color_ranges["Red"])]:
        for lower, upper in ranges:
            color_mask = cv2.inRange(hsv, lower, upper)
            combined_mask = color_mask if combined_mask is None else cv2.bitwise_or(combined_mask, color_mask)

    combined_mask = preprocess_mask(combined_mask, kernel)
    edges = cv2.Canny(combined_mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not check_position(y, h, frame_height):
            continue

        if detect_triangle_shape(contour) and check_triangle_colors(hsv, contour, color_ranges):
            results.append(("Warning Sign", contour, (x, y, w, h)))

    return results

def process_video(video_path):
    output_dir = 'video_output'
    os.makedirs(output_dir, exist_ok=True)

    video_filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f'output_{video_filename}')

    cap = cv2.VideoCapture(video_path)
    color_ranges = get_color_ranges()
    kernel = np.ones((5, 5), np.uint8)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detected_signs = detect_signs(hsv, color_ranges, kernel)

        for label, contour, (x, y, w, h) in detected_signs:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(frame)

        cv2.imshow('Traffic Sign Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    video_dir = 'video'
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            print(f'Processing {video_file}...')
            process_video(video_path)

if __name__ == "__main__":
    main()

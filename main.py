import cv2
import numpy as np
import os
import unicodedata
import re

STANDARD_SIZE = (150, 150)

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

def calculate_pixel_count(hsv, mask, color_ranges):
    color_counts = {color: 0 for color in color_ranges}
    for color, ranges in color_ranges.items():
        color_mask = np.zeros_like(mask)
        for lower, upper in ranges:
            color_mask |= cv2.inRange(hsv, lower, upper)
        color_area = cv2.bitwise_and(mask, color_mask)
        color_counts[color] = np.sum(color_area > 0)
    return color_counts

def calculate_color_ratios(color_counts, total_area):
    color_ratios = {color: count / total_area for color, count in color_counts.items()}
    return color_ratios

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

def analyze_sign_color_and_shape(hsv, contour, color_ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255), -1)

    # Tính toán số lượng pixel của từng màu trong vùng biển báo
    color_counts = calculate_pixel_count(hsv, mask, color_ranges)
    total_area = np.sum(mask > 0)

    # Tính tỷ lệ màu trong vùng biển báo
    color_ratios = calculate_color_ratios(color_counts, total_area)

    # Phân tích hình dạng biển báo
    if detect_triangle_shape(contour):
        shape = "Triangle"
    else:
        shape = "Other Shape"

    return color_counts, color_ratios, shape

def preprocess_mask(mask, kernel):
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

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
                    # Phân tích màu sắc ngay cả khi phát hiện hình tròn
                    color_counts, color_ratios, shape = analyze_sign_color_and_shape(hsv, contour, color_ranges)
                    results.append((f"{color} Circle", contour, (x, y, w, h), color_counts, color_ratios, "Circle"))

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

        if detect_triangle_shape(contour):
            color_counts, color_ratios, shape = analyze_sign_color_and_shape(hsv, contour, color_ranges)
            results.append(("Warning Sign", contour, (x, y, w, h), color_counts, color_ratios, shape))

    return results

def load_templates(template_dir):
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith(('.jpg', '.png')):
            label = os.path.splitext(filename)[0]
            img = cv2.imread(os.path.join(template_dir, filename))  # Read in color
            # img = cv2.resize(img, STANDARD_SIZE)
            templates[label] = img
    return templates

def normalize_label(label):
    label = label.replace('Đ', 'D').replace('đ', 'd')
    label = unicodedata.normalize('NFD', label)
    label = ''.join(c for c in label if unicodedata.category(c) != 'Mn')
    label = re.sub(r"[^a-zA-Z0-9\s]", "", label)
    label = label.strip().title()
    return label

def load_templates_orb(template_dir, size=STANDARD_SIZE):
    orb = cv2.ORB_create()
    templates = {}

    for filename in os.listdir(template_dir):
        if filename.lower().endswith(('.jpg', '.png')):
            label = os.path.splitext(filename)[0]
            img = cv2.imread(os.path.join(template_dir, filename), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = resize_with_padding(img, size)
            kp, des = orb.detectAndCompute(img_resized, None)
            templates[label] = (img_resized, kp, des)
    return templates

def resize_with_padding(img, size=STANDARD_SIZE):
    h, w = img.shape[:2]
    scale = min(size[0] / h, size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))

    top = (size[1] - nh) // 2
    left = (size[0] - nw) // 2

    if len(img.shape) == 2:
        # Ảnh grayscale
        result = np.full((size[1], size[0]), 128, dtype=np.uint8)
    else:
        # Ảnh màu
        result = np.full((size[1], size[0], 3), 128, dtype=np.uint8)

    result[top:top + nh, left:left + nw] = resized
    return result

def enhance_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, STANDARD_SIZE)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    
    return sharpened

def match_sign_with_template(roi_bgr, templates_bgr, templates_orb, score_thresh=0.6):
    best_match = ("Unknown", 0.0)  # (label, score)

    for label, tmpl in templates_bgr.items():
        if roi_bgr.shape != tmpl.shape:
            continue

        result = cv2.matchTemplate(roi_bgr, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)

        if score > best_match[1]:
            best_match = (label, score)

    if best_match[1] >= score_thresh:
        print(f"[DEBUG][BGR] Matching {best_match[0]} → score: {best_match[1]:.2f}")
        return best_match[0]

    # --- Nếu BGR fail, fallback sang ORB ---
    orb = cv2.ORB_create()
    roi_gray = enhance_gray(roi_bgr)
    kp2, des2 = orb.detectAndCompute(roi_gray, None)

    if des2 is None or len(kp2) < 3:
        return "Unknown"

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match_orb = ("Unknown", float("inf"))

    for label, (tmpl, kp1, des1) in templates_orb.items():
        if des1 is None or len(kp1) < 3:
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) == 0:
            continue

        score = sum(m.distance for m in matches[:10]) / len(matches[:10])
        if score < best_match_orb[1]:
            best_match_orb = (label, score)

    if best_match_orb[1] < 60:
        print(f"[DEBUG][ORB] Matching {best_match_orb[0]} → score: {best_match_orb[1]:.2f}")
        return best_match_orb[0]
    else :
        return "Unknown"

def process_video(video_path, isDebug, isShowScreen):
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

    templates_bgr = load_templates("templates")
    templates_orb = load_templates_orb("templates")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detected_signs = detect_signs(hsv, color_ranges, kernel)

        for label, contour, (x, y, w, h), color_counts, color_ratios, shape in detected_signs:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f"Detected {label} at ({x}, {y}) with shape: {shape}")
            print(f"Color Counts: {color_counts}")
            print(f"Color Ratios: {color_ratios}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, STANDARD_SIZE)

            if (isDebug == True):
                cv2.imwrite(f"debug/roi_{x}_{y}.jpg", roi)

        
            matched_label = match_sign_with_template(roi, templates_bgr, templates_orb, 0.40)
            print("Match: ", matched_label)

            cv2.putText(frame, matched_label, (x + w + 10, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        out.write(frame)

        if (isShowScreen == True):
            cv2.imshow('Traffic Sign Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def generateTemplate(image_dirs, label_dirs, classes_dirs):
    output_dir = "templates"
    os.makedirs(output_dir, exist_ok=True)

    for image_dir, label_dir, classes_dir in zip(image_dirs, label_dirs, classes_dirs):
        print(f"Start generate for {classes_dir} - {image_dir} - {label_dir}")

        class_names = open(classes_dir).read().splitlines()
        for label_file in os.listdir(label_dir):
            if not label_file.endswith(".txt"):
                continue

            filename_base = label_file.replace(".txt", "")
            image_file_jpg = os.path.join(image_dir, filename_base + ".jpg")
            image_file_png = os.path.join(image_dir, filename_base + ".png")

            if os.path.exists(image_file_jpg):
                image_file = image_file_jpg
            elif os.path.exists(image_file_png):
                image_file = image_file_png
            else:
                continue

            img = cv2.imread(image_file)
            h, w = img.shape[:2]

            with open(os.path.join(label_dir, label_file), "r") as f:
                for idx, line in enumerate(f):
                    class_id, cx, cy, bw, bh = map(float, line.strip().split())
                    x = int((cx - bw / 2) * w)
                    y = int((cy - bh / 2) * h)
                    bw = int(bw * w)
                    bh = int(bh * h)

                    cropped = img[y:y+bh, x:x+bw]
                    # if cropped.size == 0:
                    #     continue

                    cropped = cv2.resize(cropped, STANDARD_SIZE)
                    
                    class_name = normalize_label(class_names[int(class_id)])
                    output_path = os.path.join(output_dir, f"{class_name}_{idx}.jpg")
                    cv2.imwrite(output_path, cropped)         
        
        print(f"Generate template for {classes_dir} - DONE")

def main():
    video_dir = 'video'

    image_dirs = ['data/images', 'data/VTS/images/train']
    label_dirs = ['data/labels', 'data/VTS/labels/train']
    classes_dirs = ['data/classes_en.txt', 'data/VTS/classes.txt']

    generateTemplate(image_dirs, label_dirs, classes_dirs)

    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            print(f'Processing {video_file}...')
            process_video(video_path, False, False)
            print(f'Success {video_file}...')

if __name__ == "__main__":
    main()

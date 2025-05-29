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


def process_video(video_path, templates):
    # Xử lý video để nhận diện biển báo giao thông
    video = cv2.VideoCapture(video_path)
    
    # Định nghĩa khoảng màu cho các biển báo
    lower_red_1 = np.array([0, 100, 100])
    upper_red_1 = np.array([5, 255, 255])
    lower_red_2 = np.array([170, 100, 100])
    upper_red_2 = np.array([180, 255, 255])
    lower_red_3 = np.array([140, 50, 50])
    upper_red_3 = np.array([170, 255, 255])
    
    lower_blue = np.array([105, 100, 100])
    upper_blue = np.array([115, 255, 255])
    
    lower_yellow = np.array([18, 100, 100])
    upper_yellow = np.array([25, 255, 255])
    
    kernel = np.ones((5, 5), np.uint8)
    
    if not video.isOpened():
        print("Video can't open")
        exit()
    
    output_video = video_writer(video, "output_video.avi")
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame2 = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        # f = frame[:frame.shape[0] // 2, :]
        
        frame = frame2.copy()
        img_hsv = convert_color(frame, cv2.COLOR_BGR2HSV)
        
        # Phát hiện biển báo màu đỏ
        mask_red_1 = mask_color(img_hsv, lower_red_1, upper_red_1)
        mask_red_2 = mask_color(img_hsv, lower_red_2, upper_red_2)
        mask_red_3 = mask_color(img_hsv, lower_red_3, upper_red_3)
        mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
        mask_red = cv2.bitwise_or(mask_red, mask_red_3)
        
        # Phát hiện biển báo màu xanh
        mask_blue = mask_color(img_hsv, lower_blue, upper_blue)
        
        # Phát hiện biển báo màu vàng
        mask_yellow = mask_color(img_hsv, lower_yellow, upper_yellow)
        edges_yellow = edge_detection(mask_yellow)
        
        masks = [mask_blue,mask_red]
        
        # Chuẩn hóa các templates
        templates = prepare_templates(templates)
        
        frame = draw_triangle(frame, edges_yellow, (0, 255, 0), 2)
        
        for mask in masks:
            mask = preproccess_mask(mask, kernel)
            edges = edge_detection(mask)
            frame = draw(frame, edges, (0, 255, 0), 2, templates)
            
        f = frame[:frame.shape[0] // 2, :]
        frame2[:frame2.shape[0] // 2, :] = f
        
        # Ghi text lên frame
        cv2.putText(frame, "52200173_52200178", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow("Video", frame2)
        
        output_video.write(frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Nhấn 'q' để thoát
            break
        elif key == ord(' '):  # Nhấn 'Space' để tạm dừng
            while True:
                key2 = cv2.waitKey(1) & 0xFF
                if key2 == ord(' '):  # Nhấn 'Space' lần nữa để tiếp tục
                    break
                elif key2 == ord('q'):  # Nhấn 'q' để thoát khi đang tạm dừng
                    video.release()
                    cv2.destroyAllWindows()
                    exit()
                    
        
    video.release()
    output_video.release()
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def preproccess_mask(mask, kernel):
     # Tiền xử lý mask: cân bằng histogram, giảm nhiễu, và xử lý hình thái học
    mask = cv2.equalizeHist(mask) # Cân bằng histogram để cải thiện độ tương phản
    mask = cv2.GaussianBlur(mask, (9,9), 0) # Giảm nhiễu bằng Gaussian Blur
    mask = cv2.dilate(mask, kernel, iterations=2) # Giãn (dilate) để làm nổi bật vùng chính
    mask = cv2.erode(mask, kernel, iterations=2) # Co (erode) để loại bỏ nhiễu
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Đóng khe hở nhỏ
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Loại bỏ nhiễu nhỏ
    return mask
def draw(frame, edges, color, thickness, templates):
    # Phát hiện biển báo giao thông từ ROI dựa trên các mẫu template
    
    # Tìm kiếm, phát hiện các contours trong frame
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lưu các vùng ROI sau khi lọc
    filtered_contours = []

    # Lọc các vùng ROI hợp lệ dựa trên diện tích và tỷ lệ
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        ratio = w / h
        # approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)

        if 180 < area < 1800 and 0.8 <= ratio <= 1.2:
            circularity = (4 * np.pi * area) / (cv2.arcLength(contour, True) ** 2)
            if 0.8 <= circularity <= 1.2:
                # Kiểm tra chồng lấn với các vùng đã chọn
                should_add = True
                for existing in filtered_contours:
                    if calculate_iou((x, y, w, h), existing) > 0.5:  # Ngưỡng IoU
                        should_add = False
                        break
                if should_add:
                    filtered_contours.append((x, y, w, h))

    # Nhận diện biển báo với các ROI không bị trùng lặp
    # Kiểm tra template matching cho từng ROI
    for x, y, w, h in filtered_contours:
        best_match = None # Tên của biển báo phù hợp nhất
        best_value = 0 # GIá trị tương đồng lớn nhất giữa roi và template

        roi = frame[y:y+h, x:x+w]

        for name, temp in templates.items():
            try:
                # Đảm bảo ROI có cùng kích thước với template
                roi_resized = cv2.resize(roi, (temp.shape[1], temp.shape[0]))

                # Chuyển cả hai về ảnh xám nếu chưa phải
                if len(roi_resized.shape) > 2:
                    roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                if len(temp.shape) > 2:
                    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

                # Đảm bảo cả hai ở dạng 8-bit
                roi_resized = prepare_roi(roi_resized)
                roi_resized = roi_resized.astype(np.uint8)
                temp = temp.astype(np.uint8)

                # Thực hiện template matching
                result = cv2.matchTemplate(roi_resized, temp, cv2.TM_CCOEFF_NORMED,cv2.TM_CCORR_NORMED)
                
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val > best_value:
                    best_value = max_val
                    best_match = name

            except Exception as e:
                print(f"Lỗi khi xử lý template {name}: {e}")
                continue

        # Nếu mức độ tương đồng cao, vẽ hình chữ nhật và ghi tên biển báo
        # cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        if best_match and best_value > 0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(frame, best_match, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
def calculate_iou(box1, box2):
     # Tính Intersection over Union (IoU) giữa hai hình chữ nhật
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Xác định tọa độ giao nhau
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Tính diện tích giao và diện tích hợp
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area

    # Trả về giá trị IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou
def prepare_templates(templates):
    prepared_templates = {}
    for name, template in templates.items():
        # Kiểm tra số chiều kênh của ảnh
        if len(template.shape) > 2:
            # Nếu là ảnh màu, chuyển sang ảnh xám
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            # Nếu đã là ảnh xám, giữ nguyên
            template_gray = template

        # Resize template nếu cần
        template_gray = cv2.resize(template_gray, (64,64))
        prepared_templates[name] = template_gray

    return prepared_templates

def prepare_roi(roi_resized):
    # Tiền xử lý vùng ROI: chuyển sang ảnh xám, cân bằng histogram, và giảm nhiễu
    # Chuyển về ảnh xám
    if len(roi_resized.shape) > 2:
        roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

    # Cân bằng histogram
    roi_resized = cv2.equalizeHist(roi_resized)
    
    # Giảm nhiễu
    roi_resized = cv2.GaussianBlur(roi_resized, (7,7), 0)
    
    return roi_resized

def draw_triangle(frame, edges, color, thickness):
    # Vẽ hình chữ nhật bao quanh các vùng tam giác được phát hiện
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Lọc các vùng nhỏ không đáng kể
            epsilon = 0.06 * cv2.arcLength(contour, True) # Tính ngưỡng đơn giản hóa
            approx = cv2.approxPolyDP(contour, epsilon, True)
            num_vertices = len(approx)
            if num_vertices == 3: # Xác định tam giác
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area # Kiểm tra độ lồi
                if 0.9 < solidity <= 1.0:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness) 
    return frame

def mask_color(frame, lower_color, upper_color):
    # Tạo mặt nạ (mask) bằng cách giữ lại các pixel trong khoảng màu chỉ định
    return cv2.inRange(frame, lower_color, upper_color)
def edge_detection(mask):
    # Phát hiện cạnh (edge detection) trên mask bằng thuật toán Canny
    return cv2.Canny(mask, 50, 150)

def convert_color(frame, color):
    # Chuyển đổi không gian màu của frame theo hệ màu chỉ định
    return cv2.cvtColor(frame, color)    
def video_writer(video, filename):
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    output_video = cv2.VideoWriter(filename, fourcc, fps, (frame_width // 2, frame_height // 2))
    return output_video

if __name__ == "__main__":
    
    templates = {
                "Cam do xe":"CamDoXe.jpg",
                "Bien bao camera phat nguoi": "BienBaoGiaoThongPhatNguoi.jpg",
                 "Cam di nguoc chieu":"CamDiNguocChieu.jpg",
                 "Cam dung va do xe":"CamDungVaDoXe.jpg",
                 "Cam re trai":"CamReTrai.jpg",
                 "Chi huong duong":"ChiHuongDuong.jpg",
                 "Chu y quan sat":"ChuYQuanSat.jpg",
                 "Co tre em":"CoTreEm.jpg",
                 "Di cham":"DiCham.jpg",
                 "Huong di moi lan xe theo vach ke duong":"HuongDiMoiLanXeTheoVachKeDuong.jpg",
                 "Huong di vong chuong ngai vat sang phai":"HuongDiVongChuongNgaiVatSangPhai.jpg",
                 "Thuong xuyen xay ra tai nan":"ThuongXuyenXayRaTaiNan.jpg"
                 }
    
    templates = {p:"./image/"+ f for p,f in templates.items()}
    templates = {p:cv2.imread(f, cv2.IMREAD_GRAYSCALE) for p,f in templates.items()}
    process_video("./video/video1.mp4", templates)
    # process_video("video2.mp4", templates)

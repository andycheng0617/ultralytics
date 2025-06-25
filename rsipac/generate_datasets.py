
import cv2
import os
import random
import csv
import shutil


def generate_yolo_dataset():
    """
    Generate YOLO format dataset from video and labels.
    """
    # Extract frames from video
    output_image_dir = "../datasets/2/images/train2025"
    output_label_dir = "../datasets/2/labels/train2025"
    
    if os.path.exists(output_image_dir):
        shutil.rmtree("../datasets/2/images/train2025", ignore_errors=False)
        shutil.rmtree("../datasets/2/labels/train2025", ignore_errors=False)
    
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    video_path = "../datasets/2/RSIPAC2025/Preliminary/train"
    for video_file in os.listdir(video_path):
        
        if video_file.endswith(".avi"):
            
            print(f"Processing video: {video_file}")
            
            ## video
            video_full_path = os.path.join(video_path, video_file)
            
            frame_index_to_shape = {}
            frame_index = 0
            cap = cv2.VideoCapture(video_full_path)
            while cap.isOpened():
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_index_to_shape[frame_index] = frame.shape
                
                frame_filename = os.path.join(output_image_dir, f"{video_file.replace('.avi', '')}-{frame_index}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_index += 1

            cap.release()
            
            ## csv
            frame_index_to_label = {}
            
            csv_path = os.path.join(video_path, video_file.replace('.avi', '') + '-gt.csv')
            with open(csv_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)  # Automatically maps rows to a dictionary using headers
                for row in reader:
                    # print(row)
                    
                    frame_id = int(row[0])
                    instance_id = int(row[1])
                    
                    minx = float(row[2])
                    miny = float(row[3])
                    bbox_width = float(row[4])
                    bbox_height = float(row[5])
                    
                    maxx = minx + bbox_width
                    maxy = miny + bbox_height
                    
                    image_height, image_width, _ = frame_index_to_shape[frame_id]
                    # print(f"Frame {frame_id} size: {image_width}x{image_height}")
                    
                    minx = max(minx, 0)
                    miny = max(miny, 0)
                    maxx = min(maxx, image_width)
                    maxy = min(maxy, image_height)
                    
                    bbox_width = maxx - minx
                    bbox_height = maxy - miny
                    
                    center_x = minx + bbox_width / 2
                    center_y = miny + bbox_height / 2
                    
                    # Convert to YOLO format
                    yolo_center_x = center_x / image_width
                    yolo_center_y = center_y / image_height
                    yolo_width = bbox_width / image_width
                    yolo_height = bbox_height / image_height

                    if yolo_center_x < 0 or yolo_center_x > 1 or yolo_center_y < 0 or yolo_center_y > 1 or \
                       yolo_width <= 0 or yolo_width > 1 or yolo_height <= 0 or yolo_height > 1:
                        print(f"Invalid YOLO coordinates for frame {frame_id}: ({yolo_center_x}, {yolo_center_y}, {yolo_width}, {yolo_height})")
                        continue
                    
                    
                    if frame_id not in frame_index_to_label:
                        frame_index_to_label[frame_id] = []
                    
                    frame_index_to_label[frame_id].append((0, yolo_center_x, yolo_center_y, yolo_width, yolo_height))
            
            
            for frame_id, labels in frame_index_to_label.items():
                label_filename = os.path.join(output_label_dir, f"{video_file.replace('.avi', '')}-{frame_id}.txt")
                
                with open(label_filename, 'w') as label_file:
                    for label in labels:
                        class_id, center_x, center_y, width, height = label
                        label_file.write(f"{class_id} {center_x} {center_y} {width} {height}\n")    




def visulize_rsipac_dataset():
    """
    Visualize RSIPAC dataset.
    """

    image_dir = "../datasets/2/images/train2025"

    video_path = "../datasets/2/RSIPAC2025/Preliminary/train"
    for video_file in os.listdir(video_path):
        
        if video_file.endswith(".avi"):
            
            ## csv
            frame_index_to_label = {}
            
            csv_path = os.path.join(video_path, video_file.replace('.avi', '') + '-gt.csv')
            with open(csv_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)  # Automatically maps rows to a dictionary using headers
                for row in reader:
                    # print(row)
                    
                    frame_id = int(row[0])
                    instance_id = int(row[1])
                    
                    minx = float(row[2])
                    miny = float(row[3])
                    bbox_width = float(row[4])
                    bbox_height = float(row[5])
                    
                    maxx = minx + bbox_width
                    maxy = miny + bbox_height
                
                    if frame_id not in frame_index_to_label:
                            frame_index_to_label[frame_id] = []
                        
                    frame_index_to_label[frame_id].append((0, minx, miny, maxx, maxy))
                
                
            for frame_id, labels in frame_index_to_label.items():
                
                image_file_name = f"{video_file.replace('.avi', '')}-{frame_id}.jpg"
                image_file_path = os.path.join(image_dir, image_file_name)
                
                image = cv2.imread(image_file_path)
                
                if image is None:
                    print(f"Failed to read image: {image_file_path}")
                    continue
                
                # Draw bounding boxes
                for label in labels:
                    class_id, x1, y1, x2, y2 = label
                    
                    # Draw rectangle
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        
                # Show image
                cv2.imshow("Image", image)
                
                k = cv2.waitKey(0)
                if k == 27:
                    break

            cv2.destroyAllWindows()
            
            break



def randomly_select_images_for_val():

    rsipac_data_path = "../datasets/2/images/train2025"

    images = os.listdir(rsipac_data_path)
    images_len = len(images)
    print(f"Total images: {images_len}")

    # 
    image_count = 0
    train_image_count_to_file_name = {}
    for image_file in os.listdir(rsipac_data_path):
        
        if not image_file.endswith(".jpg"):
                continue
        
        train_image_count_to_file_name[image_count] = image_file
        image_count += 1

    if os.path.exists("../datasets/2/images/val2025"):
        shutil.rmtree("../datasets/2/images/val2025", ignore_errors=False)
        shutil.rmtree("../datasets/2/labels/val2025", ignore_errors=False)
    
    os.mkdir("../datasets/2/images/val2025")
    os.mkdir("../datasets/2/labels/val2025")

    # random_images
    random_images = random.sample(range(0, images_len), 300)
    # print(random_images)
    for random_image_index in random_images:

        image_file_name = train_image_count_to_file_name[random_image_index]

        # jpg
        image_file_name_path = os.path.join(rsipac_data_path, image_file_name)
        shutil.copy(image_file_name_path, "../datasets/2/images/val2025/")

        # txt
        txt_file_name_path = os.path.join(rsipac_data_path.replace("images", "labels"), image_file_name.replace(".jpg", ".txt"))
        shutil.copy(txt_file_name_path, "../datasets/2/labels/val2025/")
        



def visualize_bbox_from_yolo():
    
    bboxes = []
    label_path = "../datasets/2/labels/train2025/1-2-0.txt"  # 替换为你的标签路径
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                bboxes.append((x_center, y_center, width, height))  # 假设格式为 xywh 归一化
    
    image_path = "../datasets/2/images/train2025/1-2-0.jpg"  # 替换为你的图片路径
    
    
    img = cv2.imread(image_path)
    for bbox in bboxes:
        x, y, w, h = bbox  # 假设格式为 xywh 归一化
        H, W = img.shape[:2]
        x1, y1 = int((x - w/2) * W), int((y - h/2) * H)
        x2, y2 = int((x + w/2) * W), int((y + h/2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imshow("Image with BBoxes", img)
    cv2.waitKey(0)




if __name__ == "__main__":
    

    # generate_yolo_dataset()
    
    # randomly_select_images_for_val()
    
    # visulize_rsipac_dataset()
    
    visualize_bbox_from_yolo()


    

























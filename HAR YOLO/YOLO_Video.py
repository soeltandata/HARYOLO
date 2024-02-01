# melakukan import YOLO dari ultralytics
from ultralytics import YOLO
# cv2 dan math untuk meproses gambar
import cv2
import math


def video_detection(path_x):
    # melakukan capture citra digital
    video_capture = path_x
    # mengambil lebar dan tinggi frame dari citra digital
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # mengarahkan pennyimpanan output video untuk di download oleh user
    VideoPath = "./static/output/output.avi"
    # melakukan cv2.VideoWriter untuk menyimpan dan menulis output video
    OutVideo = cv2.VideoWriter(VideoPath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    # memanggil file weights YOLOv8
    model = YOLO("YOLO-Weights/84 best weight.pt")
    # menyimpan klasa aktivitas untuk digunakan pada penulisan output label
    classNames = ['berdiri', 'berjalan', 'berlari', 'jatuh']
    # success untuk loop
    success = True
    # counter untuk write cv2.VideoWriter
    counter = 0

    while success:
        # membaca citra digital yang telah di capture
        success, img = cap.read()
        # looping counter
        counter = counter + 1
        # stream=True untuk mengaktifkan pemrosesan for-loop dapat mencegah masalah Out of Memory (OOM) karena mengurangi penyimpanan tensor perantara
        results = model(img, stream=True)

        # memproses dengan loop result dari model untuk setiap bounding box yang dibuat
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # mengambil nilai tensor x1,y1,x2,y2
                x1, y1, x2, y2 = box.xyxy[0]
                # mengubah nilai tensor x1,y1,x2,y2 menjadi integer untuk fungsi openCV
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # melakukan print koordinat box pada terminal
                print(x1, y1, x2, y2)
                # mengambil classname dari result untuk if else
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                # mengambil classname untuk menjadi label dan confidence score
                label = f'{class_name}'
                # melakukan if else untuk mewarnai bounding box pada setiap kelas
                if class_name == 'jatuh':
                    color = (0, 0, 255) # merah
                elif class_name == "berjalan":
                    color = (0, 255, 0) # hijau
                elif class_name == "berlari":
                    color = (255, 0, 0) # biru
                else:
                    color = (255, 255, 255) # putih
                # menggambarkan bounding box dan menuliskan label
                if conf > 0.7:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 2, [255,0,255], thickness=3, lineType=cv2.LINE_AA)
                # menyimpan hasil per 5 frame
                if (counter % 5 == 0):
                    OutVideo.write(img)
        yield img
cv2.destroyAllWindows()

def image_detection(path_y):
    # melakukan capture citra digital
    video_capture = path_y
    # mengambil lebar dan tinggi frame dari citra digital
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # mengarahkan pennyimpanan output gambar untuk di download oleh user
    ImagePath = "./static/output/output.jpg"
    # melakukan cv2.VideoWriter untuk menyimpan dan menulis output gambar
    OutImage = cv2.VideoWriter(ImagePath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    # memanggil file weights YOLOv8
    model = YOLO("YOLO-Weights/84 best weight.pt")
    # menyimpan klasa aktivitas untuk digunakan pada penulisan output label
    classNames = ['berdiri', 'berjalan', 'berlari', 'jatuh']
    # success untuk loop
    success = True

    while success:
        # membaca citra digital yang telah di capture
        success, img = cap.read()
        # stream=True untuk mengaktifkan pemrosesan for-loop dapat mencegah masalah Out of Memory (OOM) karena mengurangi penyimpanan tensor perantara
        results = model(img, stream=True)
        # memproses dengan loop result dari model untuk setiap bounding box yang dibuat
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # mengambil nilai tensor x1,y1,x2,y2
                x1, y1, x2, y2 = box.xyxy[0]
                # mengubah nilai tensor x1,y1,x2,y2 menjadi integer untuk fungsi openCV
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # melakukan print koordinat box pada terminal
                print(x1, y1, x2, y2)
                # mengambil classname dari result untuk if else
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                # mengambil classname untuk menjadi label dan confidence score
                label = f'{class_name}'
                # melakukan if else untuk mewarnai bounding box pada setiap kelas
                if class_name == 'jatuh':
                    color = (0, 0, 255)
                elif class_name == "berjalan":
                    color = (0, 255, 0)
                elif class_name == "berlari":
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 255)
                # menggambarkan bounding box dan menuliskan label
                if conf > 0.7:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 0, 255], thickness=3, lineType=cv2.LINE_AA)
            OutImage.write(img)
        yield img
cv2.destroyAllWindows()

def webcam_detection(path_z):
    # melakukan capture citra digital
    video_capture = path_z
    # mengambil lebar dan tinggi frame dari citra digital
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # memanggil file weights YOLOv8
    model = YOLO("YOLO-Weights/84 best weight.pt")
    # menyimpan klasa aktivitas untuk digunakan pada penulisan output label
    classNames = ['berdiri', 'berjalan', 'berlari', 'jatuh']
    # success untuk loop
    success = True

    while success:
        # membaca citra digital yang telah di capture
        success, img = cap.read()
        # stream=True untuk mengaktifkan pemrosesan for-loop dapat mencegah masalah Out of Memory (OOM) karena mengurangi penyimpanan tensor perantara
        results = model(img, stream=True)

        # memproses dengan loop result dari model untuk setiap bounding box yang dibuat
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # mengambil nilai tensor x1,y1,x2,y2
                x1, y1, x2, y2 = box.xyxy[0]
                # mengubah nilai tensor x1,y1,x2,y2 menjadi integer untuk fungsi openCV
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # melakukan print koordinat box pada terminal
                print(x1, y1, x2, y2)
                # mengambil classname dari result untuk if else
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                # mengambil classname untuk menjadi label dan confidence score
                label = f'{class_name}'
                # melakukan if else untuk mewarnai bounding box pada setiap kelas
                if class_name == 'jatuh':
                    color = (0, 0, 255)  # merah
                elif class_name == "berjalan":
                    color = (0, 255, 0)  # hijau
                elif class_name == "berlari":
                    color = (255, 0, 0)  # biru
                else:
                    color = (255, 255, 255)  # putih
                # menggambarkan bounding box dan menuliskan label
                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 2, [255, 0, 255], thickness=3, lineType=cv2.LINE_AA)

        yield img
cv2.destroyAllWindows()

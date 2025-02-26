from ultralytics import YOLO
import cv2
import os

# โหลดโมเดล YOLO
model = YOLO("yolo11l.pt")

# ตั้งค่าโฟลเดอร์ภาพ
image_folder = "images/"

# รายชื่อสัตว์ที่อาจมีหลายคลาส (อาจต้องเช็คจากโมเดลจริง)
KNOWN_CLASSES = {
    16: "Cow",
    17: "Bull",
    18: "Calf",
    19: "Holstein",
    20: "Jersey",
    21: "Angus",
    22: "Sheep",
    23: "Goat",
    24: "Horse",
    25: "Pig",
}

# วนลูปอ่านภาพทั้งหมดใน images/
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    # ตรวจจับวัตถุ
    results = model(image)

    print(f"\n🔍 ตรวจสอบภาพ: {image_name}")

    # วนลูปดูผลลัพธ์ทั้งหมด
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])  # ดึง class id
            confidence = float(box.conf[0])  # ความมั่นใจ

            # หาชื่อคลาส ถ้าไม่มีให้แสดงเป็น "Unknown"
            class_name = KNOWN_CLASSES.get(class_id, "Unknown")

            print(f"➡️ Class ID: {class_id} | ชื่อคลาส: {class_name} | ความมั่นใจ: {confidence:.2f}")

print("\n✅ การตรวจสอบคลาสเสร็จสิ้น!")

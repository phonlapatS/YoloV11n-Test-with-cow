from ultralytics import YOLO
import cv2
import os

# โหลดโมเดล YOLO
model = YOLO("yolo11l.pt")

# ตั้งค่าโฟลเดอร์ภาพ
image_folder = "images/"
output_folder = "output/"

# สร้างโฟลเดอร์ output ถ้ายังไม่มี
os.makedirs(output_folder, exist_ok=True)

# กำหนด ID ของสายพันธุ์วัว 3 คลาสที่ต้องการ
COW_BREEDS = {
    19: "Holstein",   # ตรวจสอบให้แน่ใจว่า ID ตรงกับโมเดล
    20: "Jersey",
    21: "Angus"
}

# วนลูปอ่านภาพทั้งหมดใน images/
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)

    # โหลดภาพ
    image = cv2.imread(image_path)

    # ทำการตรวจจับวัตถุ
    results = model(image)

    # วาดกรอบเฉพาะวัวที่เป็นสายพันธุ์ที่กำหนด
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])  # ดึง class id
            confidence = float(box.conf[0])  # ความมั่นใจ

            print(f"🔍 ตรวจพบ class_id: {class_id}, confidence: {confidence:.2f}")

            if class_id in COW_BREEDS and confidence > 0.8:  # เช็คว่า ID ตรงและความมั่นใจสูงพอ
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # ค่ากรอบ
                breed_name = COW_BREEDS[class_id]  # ชื่อสายพันธุ์

                # วาดกรอบสี่เหลี่ยม
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{breed_name}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            else:
                print(f"❌ ไม่พบสายพันธุ์ที่ต้องการ หรือความมั่นใจต่ำ (ID: {class_id})")

    # บันทึกผลลัพธ์
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image)

print("✅ ตรวจจับวัวเสร็จแล้ว! รูปที่มีวัวสายพันธุ์ที่ต้องการถูกบันทึกใน output/")

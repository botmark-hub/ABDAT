import cv2

def list_camera_indices(max_indices=5):
    print("กำลังตรวจสอบกล้อง...")
    for i in range(max_indices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ พบกล้องที่ index {i}")
            cap.release()
        else:
            print(f"❌ ไม่พบกล้องที่ index {i}")

if __name__ == "__main__":
    list_camera_indices(5)  # ลองตรวจสอบ 0-4
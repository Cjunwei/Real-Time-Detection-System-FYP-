import cv2
import time
import os
from ultralytics import YOLO

# =========================
# 1. Settings
# =========================
DETECTION_MODEL_PATH = r"D:\Cjunwei\Document\UniMAP\Course\FYP\Dataset\Initial Dataset\Robotflow(Full11)\runs\detect\YOLOv11m\YOLOv11m-With Mosaic\weights\best.pt"
SEGMENTATION_MODEL_PATH = r"D:\Cjunwei\Document\UniMAP\Course\FYP\Dataset\Initial Dataset\Robotflow(scratch1)\runs\segment\No Mosaic\YOLOv11n_640\weights\best.pt"
CAMERA_INDEX = 1

SAVE_DIR = r"D:\Cjunwei\Document\UniMAP\Course\FYP\Dataset\Finalize\Captures"
CONFIDENCE = 0.62
COOLDOWN = 0.01

LIVE_PATH = r"D:\Cjunwei\Document\UniMAP\Course\FYP\Dataset\Finalize\live.jpg"
TEMP_PATH = r"D:\Cjunwei\Document\UniMAP\Course\FYP\Dataset\Finalize\live_temp.jpg"

HASH_THRESHOLD = 8   # 越小越严格（建议 5~10）

IMG_SIZE = 640

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# =========================
# 2. Create folder
# =========================
os.makedirs(SAVE_DIR, exist_ok=True)

# 自动编号
existing_nums = []
for f in os.listdir(SAVE_DIR):
    name, ext = os.path.splitext(f)
    if ext.lower() in [".jpg", ".png"] and name.isdigit():
        existing_nums.append(int(name))

counter = max(existing_nums, default=0) + 1

# =========================
# 3. Image Hash Function
# =========================
def image_hash(img):
    img = cv2.resize(img, (8, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg = gray.mean()
    return (gray > avg).astype(int)

def hash_diff(h1, h2):
    return (h1 != h2).sum()

last_hash = None

# =========================
# 4. Load model
# =========================
Detection_model = YOLO(DETECTION_MODEL_PATH)
Segmentation_model = YOLO(SEGMENTATION_MODEL_PATH)

# =========================
# 5. Webcam
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

last_capture_time = 0

print("Auto capture + dedup started")
print("Press 'q' to quit")

# =========================
# 6. Main loop
# =========================
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    results = Detection_model.predict(
        source=frame,
        conf=CONFIDENCE,
        save=False,
        show=False,
        verbose=False
    )

    results = Segmentation_model.predict(
        source=frame,
        conf=CONFIDENCE,
        save=False,
        show=False,
        verbose=False
    )

    r = results[0]
    annotated_frame = r.plot()

    # =========================
    # 有 detection 才处理
    # =========================
    if len(r.boxes) > 0:

        current_time = time.time()

        if current_time - last_capture_time > COOLDOWN:

            # 当前图 hash
            current_hash = image_hash(annotated_frame)

            is_duplicate = False

            if last_hash is not None:
                diff = hash_diff(current_hash, last_hash)

                # 判断是否重复
                if diff < HASH_THRESHOLD:
                    is_duplicate = True
                    print("Duplicate detected → skip")

            if not is_duplicate:
                filename = os.path.join(SAVE_DIR, f"{counter}.jpg")

                cv2.imwrite(filename, annotated_frame)
                print(f"Saved: {filename}")

                counter += 1
                last_hash = current_hash

            last_capture_time = current_time

    # 显示画面
    # cv2.imshow("YOLO Dedup Capture", annotated_frame)
    cv2.imwrite(TEMP_PATH, annotated_frame)

    try:
        import os
        os.replace(TEMP_PATH, LIVE_PATH)
    except:
        pass

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# =========================
# 7. Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()
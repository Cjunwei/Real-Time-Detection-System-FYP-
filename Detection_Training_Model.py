from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO("yolo11m.pt")

    model.train(
        data="data.yaml",
        epochs=500,
        imgsz=800,
        batch=8,
        patience=20,
        workers=2,
        device = 0,
        mosaic=0.7,
        close_mosaic=15,
        # mixup=0,
        # copy_paste=0,
        # erasing=0,
        # degrees=0,
        # translate=0,
        # scale=0,
        # shear=0,
        # perspective=0,
        # fliplr=0,
        # flipud=0,
        # hsv_h=0,
        # hsv_s=0,
        # hsv_v=0
    )

    # model.train(
    # data="data.yaml",
    # epochs=500,
    # imgsz=640,
    # batch=8,
    # device=0,
    # workers=3,

    # mosaic=0.3,
    # close_mosaic=30,
    # mixup=0.0,
    # copy_paste=0.0,

    # degrees=2,
    # translate=0.03,
    # scale=0.2,
    # shear=0.0,
    # perspective=0.0,

    # hsv_h=0.005,
    # hsv_s=0.2,
    # hsv_v=0.2,

    # patience=30,
    # # plots=True
# )

if __name__ == "__main__":
    freeze_support()
    main()
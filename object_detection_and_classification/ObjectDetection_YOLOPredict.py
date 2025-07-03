from ultralytics import YOLO

'''
Evaluate YOLO model for object detection on test split; predictions are saved locally
'''

def main():
    model = YOLO("../models/model_yolo/weights/best.pt")
    model.val(data="../datasets/dataset_yolo/data.yaml", split='test')
    model.predict(source="../datasets/dataset_yolo/test/images", save=True, conf=0.25)

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()

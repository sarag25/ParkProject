from ultralytics import YOLO
import torch

'''
Fine-tune a pretrained YOLO model for object detection
'''

def main():
    model = YOLO("yolov8n.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train(data="../datasets/dataset_yolo/data.yaml", epochs=40, imgsz=640, batch=8, device=device, workers=4, half=True, patience=10, val=True)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
from ultralytics import YOLO
import torch

if __name__=="__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:",device)
    model=YOLO("yolov8n.pt")
    model.train(data="config.yaml",epochs=50,batch=16,imgsz=640,device=device,workers=0)

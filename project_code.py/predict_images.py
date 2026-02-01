from ultralytics import YOLO
import cv2

model=YOLO("C:/PythonMLProject/runs/detect/train5/weights/best.pt")
image_path="C:/Users/DHARSHANA D/OneDrive/Pictures/Screenshots/Screenshot 2026-01-18 215517.png"
results=model.predict(source=image_path,conf=0.25,device=0,save=True)
annotated_image=results[0].plot()
cv2.imshow("Vehicle Detection",annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

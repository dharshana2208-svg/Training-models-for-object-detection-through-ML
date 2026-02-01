import cv2
model=YOLO("C:/PythonMLProject/runs/detect/train5/weights/best.pt")
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret: break
    results=model.predict(source=frame,stream=True,conf=0.25)
    for result in results:
        boxes=result.boxes
        for box in boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0])
            conf=float(box.conf[0])
            cls=int(box.cls[0])
            label=f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.imshow("YOLOv8 Vehicle Detection",frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break
cap.release()
cv2.destroyAllWindows()

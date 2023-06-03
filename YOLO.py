from ultralytics import YOLO
import cv2
import math

def webcam_detection(path_x):
    video_capture = path_x

    cap = cv2.VideoCapture(0)

    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))

    out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'),10,(frame_width, frame_height))

    model=YOLO('best.pt')
    classNames = ["Batik-Bali", "Batik-Betawi", "Batik-Celup", "Batik-Cendrawasih", "Batik-Dayak", "Batik-Geblek-Renteng", "Batik-Insang", "Batik-Kawung", "Batik-Lasem", "Batik-Megamendung", "Batik-Pala", "Batik-Parang", "Batik-Poleng", "Batik-Sekar-Jagad", "Batik-Tambal"]

    while True:
        success,img = cap.read()
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                #print(x1,y1,x2,y2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 3)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

        yield img
        #out.write(img)
        #cv2.imshow("Image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
        #    break


    #out.release()
cv2.destroyAllWindows()
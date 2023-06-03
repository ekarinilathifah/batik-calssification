from flask import Flask, render_template, request, jsonify, Response, session
import cv2
from YOLO import webcam_detection

app = Flask(__name__, static_url_path='/static')

app.config['SECRET KEY'] = 'monangchoz'

def generate_frames(path_x = ''):
    yolo_output = webcam_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer=cv2.imencode('.jpg', detection_)
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

@app.route("/")
def beranda():
    return render_template('webcam.html')

@app.route("/webapp")
def webapp():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
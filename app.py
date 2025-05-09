from flask import Flask, render_template, Response
import cv2
import time
import pygame

app = Flask(__name__)

# Initialize webcam
cam = cv2.VideoCapture(0)

# Initialize pygame for sound alert
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('C:/Users/Asus/Desktop/new2/alert.wav')

last_alert_time = 0

def generate_frames():
    global last_alert_time
    ret, frame1 = cam.read()

    while True:
        ret, frame2 = cam.read()
        if not ret:
            break

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 5000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if time.time() - last_alert_time > 1:
                alert_sound.play()
                last_alert_time = time.time()

        ret, buffer = cv2.imencode('.jpg', frame1)
        frame = buffer.tobytes()

        frame1 = frame2

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

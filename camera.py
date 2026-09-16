from pinkylib import Camera
from flask import Flask, Response
import cv2

app = Flask(__name__)

cam = Camera()
cam.start()


def generate_frames():
    while True:
        frame = cam.get_frame()

        if frame is None:
            continue

        # JPEG로 변환
        ret, buffer = cv2.imencode('.jpg', frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )


@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Pinky Pro Camera</title>
        </head>
        <body>
            <h1>Pinky Pro Camera</h1>
            <img src="/video" width="640">
        </body>
    </html>
    """


@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )
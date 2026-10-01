"""Flask 웹 UI: 영상 스트림 + 조작 버튼/키보드."""
import time

from flask import Flask, Response, render_template_string


HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Pinky Centerline + YOLO Route</title>
<style>
body{background:#111;color:white;font-family:Arial;text-align:center}
img{width:95%;max-width:1200px;border:2px solid #555}
button{font-size:24px;margin:5px;padding:10px 20px}
</style>
</head>
<body>
<h2>Pinky Centerline + YOLO STOP / STATION / GOAL</h2>
<img src="/video_feed">
<div>
<button onclick="cmd('w')">W</button>
<button onclick="cmd('s')">S</button>
<button onclick="cmd('a')">A</button>
<button onclick="cmd('d')">D</button>
<button onclick="cmd('p')">AUTO</button>
<button onclick="cmd('r')">RESET</button>
<button onclick="cmd('space')">STOP</button>
</div>
<script>
function cmd(k){ fetch('/cmd/'+k); }
document.addEventListener('keydown', function(e){
    if(e.repeat) return;
    let k=e.key.toLowerCase();
    if(e.code==='Space'){ e.preventDefault(); cmd('space'); }
    else if(['w','a','s','d','p','r'].includes(k)){ cmd(k); }
});
</script>
</body>
</html>
"""


def create_app(robot):
    app = Flask(__name__)

    def mjpeg():
        while True:
            data = robot.latest_jpeg()
            if data is not None:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n'
            time.sleep(0.04)

    @app.route("/")
    def index():
        return render_template_string(HTML)

    @app.route("/video_feed")
    def video_feed():
        return Response(mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route("/cmd/<key>")
    def command(key):
        robot.handle_key(key)
        return "OK"

    return app

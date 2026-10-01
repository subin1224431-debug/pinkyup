"""실행: python3 main.py  (best_ncnn_model 폴더와 같은 위치에서)"""
import threading

from pinkylib import Camera, Motor

from pinky_drive import config as C
from pinky_drive.detector import TextDetector
from pinky_drive.odometry import Odometry
from pinky_drive.remote import wait_for_start
from pinky_drive.robot import Robot
from pinky_drive.web import create_app


def main():
    motor = Motor()
    camera = Camera()
    motor.enable_motor()
    camera.start()

    # ★ 오도메트리 (직진 / 글자 정렬 에서만 사용)
    #   엔코더 / IMU 를 읽는 함수를 넣으면 자동으로 그걸 사용
    #   read_ticks   : () -> (왼쪽 누적 틱, 오른쪽 누적 틱)
    #   read_yaw_deg : () -> 현재 yaw 각도(도)
    #   pinkylib 에서 실제 함수 이름을 확인해서 바꿀 것. 예시:
    #     read_ticks=lambda: motor.get_encoder()          # <- 함수 이름은 예시
    #     read_yaw_deg=lambda: imu.get_yaw()              # <- 함수 이름은 예시
    #   둘 다 None 이면 "속도 x 시간" 추정 모드 (예전 초 단위와 비슷한 정확도)
    odom = Odometry(read_ticks=None, read_yaw_deg=None)

    robot = Robot(camera, motor, TextDetector(C.MODEL_PATH), odom)
    app = create_app(robot)

    threading.Thread(target=robot.run, daemon=True).start()
    threading.Thread(target=wait_for_start, args=(robot,), daemon=True).start()

    print("Pinky server started")
    print(f"http://ROBOT_IP:{C.PORT}")

    try:
        app.run(host="0.0.0.0", port=C.PORT, threaded=True, debug=False)
    finally:
        robot.stop_event.set()
        robot.motion.stop()


if __name__ == "__main__":
    main()

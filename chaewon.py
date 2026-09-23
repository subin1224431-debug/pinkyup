"""
Pinky Pro clean base
- NO IR SENSOR
- Center line following
- Arrow blob trigger
- STOP/STATION YOLO state framework

Flow:
FOLLOW
 -> third arrow trigger
 -> STOP search
 -> STOP align + 3 sec hold
 -> follow until arrow disappears
 -> right turn (~80 deg)
 -> search STATION
 -> STATION align + 3 sec hold
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from pinkylib import Camera, Motor


# =========================
# Parameters
# =========================
BASE_SPEED = 22
KP = 0.12

ALIGN_SPEED = 10
TURN_SPEED = 20

STOP_HOLD = 3.0
STATION_HOLD = 3.0

ARROW_TRIGGER_RATIO = 0.78
TURN_TIME = 1.0

MODEL_PATH = "best_ncnn_model"


# =========================
# Hardware
# =========================
camera = Camera()
motor = Motor()

camera.start()
motor.enable_motor()

model = YOLO(MODEL_PATH)  # YOLO NCNN 모델 (best_ncnn_model)


# =========================
# State
# =========================
state = "FOLLOW"
# FOLLOW -> 3번째 화살표 교차로 회전 -> STOP -> STATION 흐름

arrow_count = 0
arrow_lock = False
intersection_turn_done = False

state_time = 0


# =========================
# Motor
# =========================
def drive(l, r):
    motor.move(int(np.clip(l, -100, 100)),
               int(np.clip(r, -100, 100)))


def stop():
    motor.move(0, 0)


def follow(error):
    corr = KP * error
    drive(BASE_SPEED + corr,
          BASE_SPEED - corr)


# =========================
# Vision
# =========================
def get_center_line(frame):

    h, w = frame.shape[:2]

    roi = frame[int(h*0.55):]

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        np.array([0,0,170]),
        np.array([180,80,255])
    )

    M = cv2.moments(mask)

    if M["m00"] == 0:
        return None

    cx = int(M["m10"]/M["m00"])

    return cx - w//2



def detect_arrow(frame):

    h,w = frame.shape[:2]

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    black = cv2.inRange(
        hsv,
        np.array([0,0,0]),
        np.array([180,150,120])
    )

    cnts,_ = cv2.findContours(
        black,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best=None

    for c in cnts:

        area=cv2.contourArea(c)

        if area < 700:
            continue

        x,y,bw,bh=cv2.boundingRect(c)

        best=(x,y,bw,bh)
        break

    return best



def detect_text(frame, name):

    result=model(
        frame,
        verbose=False
    )[0]

    for box,cls in zip(
        result.boxes.xyxy,
        result.boxes.cls
    ):

        label=model.names[int(cls)]

        if label==name:

            x1,y1,x2,y2=map(
                int,
                box
            )

            return {
                "cx":(x1+x2)//2,
                "size":y2-y1
            }

    return None


# =========================
# Main
# =========================
while True:

    frame = camera.value

    if frame is None:
        continue


    # -------- FOLLOW --------
    if state=="FOLLOW":

        arrow=detect_arrow(frame)

        if arrow and not arrow_lock:

            arrow_count += 1
            arrow_lock=True


        if arrow is None:
            arrow_lock=False


        # 3번째 화살표 끝선 trigger
        if arrow_count>=3 and arrow:

            x,y,w,h=arrow

            if y+h > frame.shape[0]*ARROW_TRIGGER_RATIO:

                state="TURN_INTERSECTION"
                state_time=time.time()



        error=get_center_line(frame)

        if error is not None:
            follow(error)





    # -------- INTERSECTION TURN --------
    elif state=="TURN_INTERSECTION":

        # 기존 화살표 기반 교차로 진입 회전
        drive(
            TURN_SPEED,
            -TURN_SPEED
        )

        if time.time()-state_time > TURN_TIME:

            stop()
            state="SEARCH_STOP"

    # -------- STOP --------
    elif state=="SEARCH_STOP":

        target=detect_text(frame,"STOP")

        if target:

            state="ALIGN_STOP"



    elif state=="ALIGN_STOP":

        target=detect_text(frame,"STOP")

        if target:

            error=target["cx"]-frame.shape[1]//2

            if abs(error)>20:
                drive(
                    -ALIGN_SPEED if error<0 else ALIGN_SPEED,
                    ALIGN_SPEED if error<0 else -ALIGN_SPEED
                )

            elif target["size"]>80:

                stop()
                state_time=time.time()
                state="STOP_HOLD"




    elif state=="STOP_HOLD":

        stop()

        if time.time()-state_time>STOP_HOLD:

            arrow_count=0
            state="FOLLOW_ARROW"



    # -------- arrow disappear then turn --------
    elif state=="FOLLOW_ARROW":

        arrow=detect_arrow(frame)

        if arrow:

            drive(18,18)

        else:

            state_time=time.time()
            state="TURN_STATION"



    elif state=="TURN_STATION":

        drive(
            TURN_SPEED,
            -TURN_SPEED
        )

        if time.time()-state_time>TURN_TIME:

            state="SEARCH_STATION"



    # -------- STATION --------
    elif state=="SEARCH_STATION":

        target=detect_text(frame,"STATION")

        if target:

            state="ALIGN_STATION"



    elif state=="ALIGN_STATION":

        target=detect_text(frame,"STATION")

        if target:

            error=target["cx"]-frame.shape[1]//2

            if abs(error)>20:

                drive(
                    -ALIGN_SPEED if error<0 else ALIGN_SPEED,
                    ALIGN_SPEED if error<0 else -ALIGN_SPEED
                )

            elif target["size"]>80:

                stop()
                state_time=time.time()
                state="STATION_HOLD"



    elif state=="STATION_HOLD":

        stop()

        if time.time()-state_time>STATION_HOLD:

            break

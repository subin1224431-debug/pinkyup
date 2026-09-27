import cv2
import numpy as np
import time
import threading
import os
import json
import zmq

from flask import Flask, Response, render_template_string
from pinkylib import Camera, Motor
from ultralytics import YOLO


# ============================================================
# Raspberry Pi / Pinky Pro
# ============================================================
app = Flask(__name__)

motor = Motor()
camera = Camera()

motor.enable_motor()
camera.start()


# ============================================================
# Network
# ============================================================
PORT = 5000

# 노트북 ZMQ 서버 IP
LAPTOP_ZMQ_IP = "172.20.10.14" # 노트북 아이피
LAPTOP_ZMQ_PORT = 6000


# ============================================================
# 주행 설정
# ============================================================
BASE_SPEED = 20
KP = 0.12
MAX_SPEED = 32
SEARCH_SPEED = 12

TEXT_FOLLOW_SPEED = 16
TEXT_FOLLOW_KP = 0.10
TEXT_FOLLOW_MAX_CORR = 10

TURN_SPEED = 18
ALIGN_SPEED = 6
CENTER_TOL = 30


# ============================================================
# ROI
# ============================================================
ROI_START_RATIO = 0.50


# ============================================================
# HSV
# ============================================================
LOWER_WHITE = np.array([0, 0, 175], dtype=np.uint8)
UPPER_WHITE = np.array([180, 75, 255], dtype=np.uint8)

LOWER_BLACK = np.array([0, 0, 0], dtype=np.uint8)
UPPER_BLACK = np.array([180, 130, 115], dtype=np.uint8)


# ============================================================
# 중심선 검출 설정
# ============================================================
MIN_AREA = 500
STEP = 15
MIN_WIDTH = 25
MAX_JUMP = 60
MAX_POINTS = 8

INTERNAL_GAP_RATIO = 0.30

# ============================================================
# 시작 후 25초 동안 중심선 추종
# ============================================================
CENTERLINE_RUN_SEC = 25.0
auto_start_time = None
initial_25s_done = False


# ============================================================
# YOLO STOP / STATION
# ============================================================
MODEL_PATH = "best_ncnn_model"
YOLO_CONF = 0.45
YOLO_IMGSZ = 320
TEXT_CLASSES = {"STOP", "STATION", "GOAL"}

# 현재 코스에서 찾아야 하는 표지 순서

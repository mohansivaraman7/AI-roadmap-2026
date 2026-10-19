import cv2
import numpy as np
import math

WIDTH = 500
HEIGHT = 500
FPS = 30
FRAMES = 60


def draw_arm(img, shoulder, length, angle_deg):
    angle = math.radians(angle_deg)

    x = int(shoulder[0] + length * math.cos(angle))
    y = int(shoulder[1] - length * math.sin(angle))

    cv2.line(img, shoulder, (x, y), (0, 0, 0), 6)
    return (x, y)


def draw_body(img, left_angle, right_angle):

    head = (250, 80)
    neck = (250, 120)
    hip = (250, 250)

    cv2.circle(img, head, 30, (0,0,0), 3)

    cv2.line(img, neck, hip, (0,0,0), 6)

    left_shoulder = (220,130)
    right_shoulder = (280,130)

    left_hand = draw_arm(img,left_shoulder,90,left_angle)
    right_hand = draw_arm(img,right_shoulder,90,right_angle)

    left_leg = (220,380)
    right_leg = (280,380)

    cv2.line(img,hip,left_leg,(0,0,0),6)
    cv2.line(img,hip,right_leg,(0,0,0),6)


def interpolate(a,b,t):
    return a + (b-a)*t


def make_video(name,start_left,start_right,end_left,end_right):

    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter(name,fourcc,FPS,(WIDTH,HEIGHT))

    for i in range(FRAMES):

        t=i/(FRAMES-1)

        la=interpolate(start_left,end_left,t)
        ra=interpolate(start_right,end_right,t)

        img=np.ones((HEIGHT,WIDTH,3),dtype=np.uint8)*255

        draw_body(img,la,ra)

        cv2.putText(img,
                    name.replace(".mp4",""),
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2)

        out.write(img)

    out.release()


# -------------------------------------------------
# Generate Gestures
# -------------------------------------------------

# Arms hanging down = -90°
# Horizontal = 0°
# Straight up = 90°

make_video(
    "OUT.mp4",
    -90,-90,
    -90,90
)

make_video(
    "SIX.mp4",
    -90,-90,
    90,90
)

make_video(
    "WIDE.mp4",
    -90,-90,
    0,180
)

make_video(
    "FOUR.mp4",
    -90,-90,
    0,45
)

make_video(
    "NO_BALL.mp4",
    -90,-90,
    0,-90
)

print("Done!")
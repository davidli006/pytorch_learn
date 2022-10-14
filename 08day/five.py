import os
import time

import cv2
import numpy as np
import pyautogui

image_path = "../data/body_posture"
net = cv2.dnn.readNetFromTensorflow(f"{image_path}/graph_opt.pb")

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

HEAD = {"Nose", "REye", "LEye", "REar", "LEar"}


def get_file(image_path):
    img_end = [".jpg"]
    video_end = [".mp4"]
    file_list = os.listdir(image_path)
    image_file = {"video": [], "image": []}
    for file in file_list:
        img = os.path.join(image_path, file)
        end = os.path.splitext(img)[-1]
        if not os.path.isfile(img):
            continue
        if end in img_end:
            image_file.get("image").append(img)
        if end in video_end:
            image_file.get("video").append(img)
    return image_file


def show(frame, is_img=True):
    cv2.imshow('OpenCV', frame)
    if not is_img:
        return
    if cv2.waitKey(0):
        cv2.destroyAllWindows()


def draw(frame, is_image=True, head=False):
    width = frame.shape[1]
    height = frame.shape[0]

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heat_map = out[0, i, :, :]

        _, conf, _, point = cv2.minMaxLoc(heat_map)
        x = (width * point[0]) / out.shape[3]
        y = (height * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > 0.2 else None)

    res_head = []
    for partFrom, partTo in POSE_PAIRS:
        idFrom = BODY_PARTS.get(partFrom)
        idTo = BODY_PARTS.get(partTo)

        pointFrom = points[idFrom]
        pointTo = points[idTo]

        if pointFrom and pointTo:
            if partFrom in HEAD and len(res_head)<10:
                res_head.append(pointFrom)
            cv2.line(frame, pointFrom, pointTo, (0, 255, 0), 3)
            cv2.ellipse(frame, pointFrom, (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, pointTo, (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000
    cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    show(frame, is_image)
    if head:
        return res_head

def from_file():
    file_dict = get_file(image_path)
    print(file_dict)
    for key, value in file_dict.items():
        for val in value:
            if key == "image":
                frame = cv2.imread(val)
                draw(frame, True)
            else:
                cap = cv2.VideoCapture(val)
                while cv2.waitKey(1) < 0:
                    has_flg, frame = cap.read()
                    if not has_flg:
                        continue
                    draw(frame, False)

def from_screen(left, top, width, height):
    while cv2.waitKey(1) < 0:
        img = pyautogui.screenshot(region=[left, top, width, height])
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        head = draw(img, False, True)
        if head:
            l, t =  head[0]
            print(l, t, left+l, top+t)
            pyautogui.moveTo(left+l, top+t)
            time.sleep(3)

# esc 键退出
def main():
    from_screen(800, 150, 600, 600)



if __name__ == "__main__":
    main()

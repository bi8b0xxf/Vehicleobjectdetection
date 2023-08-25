import os
import json
import time
import torch
import cv2
import numpy as np
from build_utils import img_utils, torch_utils, utils
from models import Darknet
from draw_box_utils import draw_box, draw_videobox


def main():
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    cfg = "./cfg/my_yolov3.cfg"  # 改成生成的.cfg文件
    weights = "./weights/yolov3spp-99.pt"  # 改成自己训练好的权重文件
    json_path = "./data/pascal_voc_classes.json"  # json标签文件
    # video_path = "1.mp4"

    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    # assert os.path.exists(video_path), "video file {} dose not exist.".format(video_path)

    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()

    category_index = {v: k for k, v in class_dict.items()}
    input_size = (img_size, img_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Darknet(cfg, img_size).to(device)
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    model.eval()

    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img_o = cap.read()

        if not ret:
            break
        if not isinstance(img_o, np.ndarray):
            print("Error: Video frame is not a numpy array.")
            continue

        t1 = time.time()
        img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # scale (0, 255) to (0, 1)
        img = img.unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            pred = model(img)[0]  # only get inference result

        pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]

        if pred is None:
            print("No target detected.")
            continue

        pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
        bboxes = pred[:, :4].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1

        img_o = draw_videobox(img_o[:, :, ::-1], bboxes, classes, scores, category_index)

        if not isinstance(img_o, np.ndarray):
            print("Error: Drawn image is not a numpy array.")
            continue

        t2 = time.time()
        print("Time: {:.3f}s".format(t2-t1))

        cv2.imshow('video', img_o)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

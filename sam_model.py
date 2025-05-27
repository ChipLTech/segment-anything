import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import subprocess

from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

def download_if_missing(url, filename):
    # 检查文件是否存在
    if os.path.exists(filename):
        print(f"文件 '{filename}' 已存在，跳过下载")
        return True

    try:
        # 执行下载命令
        print(f"开始下载 '{filename}'...")
        subprocess.run(
            ["wget", "-O", filename, url],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"下载完成: {filename}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"下载失败: {e.stderr}")
        return False
    except Exception as e:
        print(f"发生错误: {e}")
        return False

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = sys.argv[1:]
    device = args[0]
    image_path = args[1]

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "./sam_vit_h_4b8939.pth"

    download_if_missing("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", sam_checkpoint)

    model_type = "vit_h"

    if device == "dlc":
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).dlc()
    else:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    print(f"Sam model loaded device:{device}")

    predictor = SamPredictor(sam)

    predictor.set_image(image)

    print("Sam predictor encoder finished")

    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.savefig('truck.jpg')

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    for i, (mask, score) in tqdm(enumerate(zip(masks, scores)), ncols=120, desc="show and save jpg"):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f'truck{i}.jpg')

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

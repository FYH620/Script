import cv2
import torch
import numpy as np
from torch import Tensor
from colorsys import hsv_to_rgb
from PIL import ImageDraw, ImageFont, Image


def drawOnePictureWithBox(
    index_to_labels,
    img,
    box_coords,
    box_labels,
    box_scores=None,
    resize_scale=800,
    is_use_scores=True,
    is_ralative_coords=False,
):

    """
    Args:
        index_to_labels(dict):{0:'areoplane',...}   index to labels
        img(ndarray):shape (H,W,C)                  opencv-image
        box_coords(ndarray):shape [num_boxes,4]     coord:[xmin,ymin,xmax,ymax]
        box_labels(ndarray):shape [num_boxes]       number label with each box
        resize_scale(int):eg SSD=300                the resize scale for training
        box_scores(ndarray):shape [num_boxes]       scores with each box
        is_use_scores(bool)                         check if we use scores
        is_ralative_coords(bool)                    check percentage size
    """

    # confirm if it is an opencv image
    if img.shape[2] != 3:
        raise ValueError("We only support RGB images.")

    # Tensor->ndarray
    if isinstance(box_coords, Tensor):
        box_coords = box_coords.data.numpy()
    if isinstance(box_labels, Tensor):
        box_labels = box_labels.data.numpy()
    if isinstance(box_scores, Tensor):
        box_scores = box_scores.data.numpy()
    box_coords = np.array(box_coords)
    box_labels = np.array(box_labels)
    if is_use_scores:
        box_scores = np.array(box_scores)

    # update box coords and some information
    img_height, img_width, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    font = ImageFont.truetype(
        font="C:/Windows/Fonts/simhei.ttf",
        size=np.floor(3e-2 * img_width + 0.5).astype("int32"),
    )
    thickness = max((img_height + img_width) // resize_scale, 1)
    num_classes = len(index_to_labels.keys())
    hsv_tuples = [(x / num_classes, 1, 1) for x in range(num_classes)]
    colors = list(map(lambda x: hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(
            lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors,
        )
    )
    if is_ralative_coords:
        box_coords[:, [0, 2]] *= img_width
        box_coords[:, [1, 3]] *= img_height
    box_coords = box_coords.astype(np.int32)
    box_labels = box_labels.astype(np.int32)
    if is_use_scores:
        box_scores = box_scores.astype(np.float32)

    # draw the bounding boxes
    for i, c in enumerate(box_labels):
        class_name = index_to_labels[c]
        box_coord = box_coords[i]
        if is_use_scores:
            box_score = box_scores[i]

        left, top, right, bottom = box_coord
        left = max(0, np.floor(left))
        top = max(0, np.floor(top))
        right = min(img_width, np.floor(right))
        bottom = min(img_height, np.floor(bottom))

        if is_use_scores:
            label = "{} {:.2f}".format(class_name, box_score)
        else:
            label = "{}".format(class_name)
        draw = ImageDraw.Draw(img)
        label_size = draw.textsize(label, font)
        print(left, top, right, bottom, label)
        label = label.encode("utf-8")

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c],
            )
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c],
        )
        draw.text(text_origin, str(label, "UTF-8"), fill=(0, 0, 0), font=font)
        del draw

    # show the final image
    img.show()


def test_demo():
    index_to_label = {0: "bicycle", 1: "person"}
    img = cv2.imread("test.jpg")
    box_coords = torch.tensor(
        [
            [70, 202, 255, 500],
            [251, 242, 334, 500],
            [1, 144, 67, 436],
            [1, 1, 66, 363],
            [74, 1, 272, 462],
            [252, 19, 334, 487],
        ]
    )
    box_labels = torch.tensor([0, 0, 0, 1, 1, 1])
    box_scores = torch.tensor([1, 0.95, 0.73, 0.87, 0.81, 0.66])
    drawOnePictureWithBox(
        index_to_label,
        img,
        box_coords,
        box_labels,
        box_scores,
        resize_scale=300,
        is_use_scores=True,
        is_ralative_coords=False,
    )


if __name__ == "__main__":
    test_demo()

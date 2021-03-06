import os
import numpy as np


def readTextAnnotations(txt_name):
    boxes_coords = []
    boxes_labels = []
    image_paths = []
    txt_path = os.getcwd() + "\\" + str(txt_name)

    with open(txt_path) as f:
        lines = f.readlines()

    for line in lines:

        content = line.split(" ")
        image_paths += [content[0]]

        boxes_coord = []
        boxes_label = []

        for coord_with_label in content[1:]:

            coord_with_label = coord_with_label.split(",")

            xmin = int(coord_with_label[0])
            ymin = int(coord_with_label[1])
            xmax = int(coord_with_label[2])
            ymax = int(coord_with_label[3])

            boxes_coord += [[xmin, ymin, xmax, ymax]]
            boxes_label += [int(coord_with_label[4])]

        boxes_coords += [np.array(boxes_coord, dtype=np.float32)]
        boxes_labels += [np.array(boxes_label, dtype=np.float32)]

    return boxes_coords, boxes_labels, image_paths


def test_demo():
    boxes_coords, boxes_labels, image_paths = readTextAnnotations(
        txt_name="txt_annotations.txt"
    )
    print(boxes_coords, boxes_labels, image_paths)


if __name__ == "__main__":
    test_demo()

import os


def makeTextAnnotations(boxes_coords, boxes_labels, image_paths):
    """
    Args:
        boxes_coords(list): shape [image_total_num,each_image_bbox_num,4]
        boxes_labels(list): shape [image_total_num,each_image_bbox_num]
        image_paths(list): shape [image_total_num]
    Returns:
        Write a txt file in the directory of this script with a unify data format.
    """

    WRITE_PATH = os.getcwd() + "\\txt_annotations.txt"
    for i in range(len(image_paths)):
        boxes_coord = boxes_coords[i]
        boxes_label = boxes_labels[i]
        image_path = image_paths[i]

        one_bbox_info = list()
        for j in range(len(boxes_label)):

            one_bbox_coords = ",".join([str(k) for k in boxes_coord[j]])
            one_bbox_label = "," + str(boxes_label[j])
            one_bbox_info += [one_bbox_coords + one_bbox_label]

        one_image_line = "" + str(image_path) + " " + " ".join(one_bbox_info) + "\n"
        with open(WRITE_PATH, "a") as f:
            f.writelines(one_image_line)


def test_demo():
    boxes_coords = [
        [[20, 50, 60, 80], [10, 100, 200, 150]],
        [[10, 40, 70, 180], [120, 150, 140, 280], [110, 200, 130, 250]],
    ]
    boxes_labels = [[1, 2], [3, 8, 10]]
    image_paths = ["test\img1.jpg", "test\img2.jpg"]
    makeTextAnnotations(boxes_coords, boxes_labels, image_paths)


if __name__ == "__main__":
    test_demo()

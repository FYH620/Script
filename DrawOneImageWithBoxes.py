import numpy as np
import cv2


class DrawOnePictureWithBBox(object):
    def __init__(
        self, index_to_labels, img, bbox_coords, bbox_labels, bbox_scores=None
    ):

        """
        Args:
            index_to_labels(dict):{0:'areoplane',...}      index to labels
            img(ndarray):a numpy image                     opencv-image
            bbox_coords(tensor/array):shape [num_bboxes,4] absolute coords([xmin,ymin,xmax,ymax])
            bbox_labels(tensor/array):shape [num_bboxes]   number labels with each bbox
            bbox_scores(tensor):shape [num_bboxes]         scores with each bbox
        Return:
            An image with bounding boxes
        """

        self.img = img
        self.bbox_coords = np.array(bbox_coords)
        self.bbox_labels = np.array(bbox_labels)
        self.bbox_scores = bbox_scores
        self.index_to_labels = index_to_labels

    def __call__(self):

        for i in range(len(self.bbox_coords)):

            min_xy = (self.bbox_coords[i, 0], self.bbox_coords[i, 1])
            max_xy = (self.bbox_coords[i, 2], self.bbox_coords[i, 3])
            cv2.rectangle(self.img, min_xy, max_xy, color=(241, 86, 66), thickness=2)

            texts = str(self.index_to_labels[self.bbox_labels[i]])
            if self.bbox_scores is not None:
                texts += str(np.round(self.bbox_scores[i], decimals=2))
            cv2.putText(
                self.img,
                texts,
                min_xy,
                cv2.FONT_HERSHEY_PLAIN,
                color=(16, 213, 243),
                thickness=1,
                fontScale=1,
            )

        cv2.imshow("img-with-box:press q to exit", self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()


a = cv2.imread("train_1.jpg")
b = [[22, 104, 106, 183], [161, 100, 237, 176]]
c = [0, 0]
DrawOnePictureWithBBox({0: "mask", 1: "nomask"}, a, b, c)()

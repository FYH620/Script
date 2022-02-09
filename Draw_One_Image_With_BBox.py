import numpy as np
from PIL import ImageDraw, Image,ImageFont
import cv2

class DrawOnePictureWithBBox(object):
    def __init__(self,index_to_labels,img,bbox_coords,bbox_labels,bbox_scores=None):

        '''
        Args:
            index_to_labels(dict):{0:'areoplane',...}      index to labels
            img(ndarray):a numpy image                     opencv-image
            bbox_coords(tensor/array):shape [num_bboxes,4] absolute coords
            bbox_labels(tensor/array):shape [num_bboxes]   number labels with each bbox
            bbox_scores(tensor):shape [num_bboxes]         scores with each bbox
        Return:
            An image with bounding boxes
        '''

        self.img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        self.bbox_coords = bbox_coords
        self.bbox_labels = bbox_labels
        self.bbox_scores = bbox_scores
        self.index_to_labels = index_to_labels

    def __call__(self):
        draw = ImageDraw.Draw(self.img)
        font = ImageFont.truetype("msyh.ttc",20,encoding="utf-8")
        for i in range(len(self.bbox_coords)):
            draw.rectangle(self.bbox_coords[i], outline='red')
            texts = str(self.index_to_labels[self.bbox_labels[i]])
            if self.bbox_scores is not None:
                texts += str(np.round(self.bbox_scores[i], decimals=2))
            draw.text((self.bbox_coords[i][0], self.bbox_coords[i][1]),texts,fill='blue',font=font)
        self.img.show()

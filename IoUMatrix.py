import torch
import numpy as np

class GetJaccardMatrix(object):
    def __init__(self,boxes_a,boxes_b):

        '''
        Args:
            self.boxes_a: shape [A,4]
            self.boxes_b: shape [B,4]
        Return:
            product with IOU Matrix: shape [A,B]
        '''

        self.boxes_a=torch.tensor(boxes_a)
        self.boxes_b=torch.tensor(boxes_b)
        self.num_boxes_a=self.boxes_a.shape[0]
        self.num_boxes_b=self.boxes_b.shape[0]

    def __call__(self):

        for i in range(self.num_boxes_a):
            if self.boxes_a[i,0]>=self.boxes_a[i,2] or self.boxes_a[i,1]>=self.boxes_a[i,3]:
                raise ValueError('The first boxes group with index {} coord label is wrong.'.format(i))
        for i in range(self.num_boxes_b):
            if self.boxes_b[i,0]>=self.boxes_b[i,2] or self.boxes_b[i,1]>=self.boxes_b[i,3]:
                raise ValueError('The second boxes group with index {} coord label is wrong.'.format(i))

        area_a=(self.boxes_a[:,2]-self.boxes_a[:,0])*(self.boxes_a[:,3]-self.boxes_a[:,1]).unsqueeze(1).expand(self.num_boxes_a,self.num_boxes_b)
        area_b=(self.boxes_b[:,2]-self.boxes_b[:,0])*(self.boxes_b[:,3]-self.boxes_b[:,1]).unsqueeze(0).expand(self.num_boxes_a,self.num_boxes_b)
        inter=self._get_intersact()
        union=area_a+area_b-inter

        return inter/union

    def _get_intersact(self):

        left_top_coord=torch.max(self.boxes_a[:,:2].unsqueeze(1).expand(self.num_boxes_a,self.num_boxes_b,2),
                                 self.boxes_b[:,:2].unsqueeze(0).expand(self.num_boxes_a,self.num_boxes_b,2))
        right_bottom_coord=torch.min(self.boxes_a[:,2:].unsqueeze(1).expand(self.num_boxes_a,self.num_boxes_b,2),
                                     self.boxes_b[:,2:].unsqueeze(0).expand(self.num_boxes_a,self.num_boxes_b,2))
        intersact_width_and_height=torch.clamp((right_bottom_coord-left_top_coord),min=0)

        return intersact_width_and_height[:,:,0]*intersact_width_and_height[:,:,1]


import torch
from torch import nn
from torch.autograd import Variable

class DIoUloss(nn.Module):

    '''
    Args:
        pred_boxes(tensor): shape [N,4]
        gt_boxes(tensor): shape [N,4]
        size_average(bool): use mean or sum
    Return:
        diou_loss
    Attention:
        1.The input args("pred_boxes" and "gt_boxes") must be absolute [xmin,ymin,xmax,ymax] coords.
        2.The coords like [x,y,w,h] or encoded coords(like RCNN encoded coords way) or relative coords
          are not supported.
        3.How to call:see line 60.
    '''

    def __init__(self):
        super(DIoUloss,self).__init__()

    def forward(self,pred_boxes,gt_boxes,size_average=False):

        ixmin=torch.max(pred_boxes[:,0],gt_boxes[:,0])
        iymin=torch.max(pred_boxes[:,1],gt_boxes[:,1])
        ixmax=torch.min(pred_boxes[:,2],gt_boxes[:,2])
        iymax=torch.min(pred_boxes[:,3],gt_boxes[:,3])
        iw=(ixmax-ixmin+1).clamp(min=0)
        ih=(iymax-iymin+1).clamp(min=0)

        inter=torch.clamp(iw*ih,min=0)
        union=(pred_boxes[:, 2]-pred_boxes[:, 0] + 1.0)* \
              (pred_boxes[:, 3]-pred_boxes[:, 1]+1.0) + \
              (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0) * \
              (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0) - inter
        iou=inter/union

        cxpred = (pred_boxes[:, 2] + pred_boxes[:, 0]) / 2
        cypred = (pred_boxes[:, 3] +pred_boxes[:, 1]) / 2
        cxgt = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
        cygt = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2
        inter_distance = (cxpred - cxgt).pow(2) + (cygt - cypred).pow(2)

        oxmin=torch.min(pred_boxes[:,0],gt_boxes[:,0])
        oymin=torch.min(pred_boxes[:,1],gt_boxes[:,1])
        oxmax=torch.max(pred_boxes[:,2],gt_boxes[:,2])
        oymax=torch.max(pred_boxes[:,3],gt_boxes[:,3])
        outer_distance = (oxmax - oxmin).pow(2) + (oymax - oymin).pow(2)

        diou = iou - inter_distance / outer_distance
        diou = torch.clamp(diou,min=-1,max=1)
        diou_loss = 1 - diou

        if size_average:
            return Variable(torch.mean(diou_loss),requires_grad=True)
        return Variable(torch.sum(diou_loss),requires_grad=True)

# if __name__ == '__main__':
#     pred_boxes = torch.tensor([[15, 18, 47, 60],
#                                 [70, 80, 120, 145]], dtype=torch.float)
#     gt_boxes = torch.tensor([[70, 80, 120, 150],[50, 50, 90, 100]], dtype=torch.float)
#     loss=DIoUloss()
#     a=loss(pred_boxes, gt_boxes)
#     print(a)
#     a.backward()
        
import os

class MakeTextAnnotations(object):

    '''
    Args:
        boxes_coords(list): shape [image_total_num,each_image_bbox_num,4]
        boxes_labels(list): shape [image_total_num,each_image_bbox_num]
        image_paths(list): shape [image_total_num]
    Returns:
        Write a txt file in the directory of this script with a unify data format.
    '''

    def __init__(self,boxes_coords,boxes_labels,image_paths):

        self._WRITE_PATH=os.getcwd()+'\\txt_annotations.txt'
        self.boxes_coords=boxes_coords
        self.boxes_labels=boxes_labels
        self.image_paths=image_paths
        
    def __call__(self):

        for i in range(len(self.image_paths)):
            boxes_coord=self.boxes_coords[i]
            boxes_label=self.boxes_labels[i]
            image_path=self.image_paths[i]

            one_bbox_info=list()
            for j in range(len(boxes_label)):
                
                one_bbox_coords=",".join([str(k) for k in boxes_coord[j]])
                one_bbox_label=","+str(boxes_label[j])
                one_bbox_info+=[one_bbox_coords+one_bbox_label]

            one_image_line=''+str(image_path)+" "+" ".join(one_bbox_info)+'\n'
            with open(self._WRITE_PATH,'a') as f:
                f.writelines(one_image_line)

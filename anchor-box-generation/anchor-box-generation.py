import torch
import math

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    if isinstance(feature_size, (int, float)):
        feat_h = feat_w = int(feature_size)
    else:
        feat_h, feat_w = feature_size
        
    if isinstance(image_size, (int, float)):
        img_h = img_w = int(image_size)
    else:
        img_h, img_w = image_size
    
    stride_h = img_h / feat_h
    stride_w = img_w / feat_w
    
    # Tính tọa độ tâm
    shift_y = (torch.arange(0, feat_h) + 0.5) * stride_h
    shift_x = (torch.arange(0, feat_w) + 0.5) * stride_w
    
    grid_y, grid_x = torch.meshgrid([shift_y, shift_x], indexing='ij')
    
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)
    
    boxes = []
    
    # SỬA Ở ĐÂY: scale chính là kích thước hộp vuông tuyệt đối
    for scale in scales:
        for ar in aspect_ratios:
            # Không nhân với stride (base_size) nữa
            h = scale / math.sqrt(ar)
            w = scale * math.sqrt(ar)
            boxes.append(torch.tensor([-w/2, -h/2, w/2, h/2]))
    
    if not boxes:
        return []
        
    base_anchors = torch.stack(boxes)
    
    grid_centers = torch.stack([grid_x, grid_y, grid_x, grid_y], dim=1).unsqueeze(1)
    anchors = grid_centers + base_anchors.unsqueeze(0)
    
    return anchors.reshape(-1, 4).tolist()
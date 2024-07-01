_base_ = [
    './meta-rcnn_r50_c4.py',
]

custom_imports = dict(
    imports=[
        'vfa.vfa_detector',
        'vfa.vfa_roi_head',
        'vfa.vfa_bbox_head'], 
    allow_failed_imports=False)

pretrained = 'open-mmlab://detectron2/resnet101_caffe'
# model settings
model = dict(
    type='VFA',
    pretrained=pretrained,
    backbone=dict(depth=101),
    roi_head=dict(
        type='VFARoIHead',
        shared_head=dict(pretrained=pretrained),
        bbox_head=dict(
            type='VFABBoxHead', num_classes=20, num_meta_classes=20),
            novel_class=(15, 16, 17, 18, 19),
            base_class=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)))

_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_coco_stuff.txt'
)

# dataset settings
dataset_type = 'COCOStuffDataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(
            img_path='images/test2014', seg_map_path='annotations/test2014'),
        pipeline=test_pipeline))
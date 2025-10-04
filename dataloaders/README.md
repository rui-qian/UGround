# Dataset

1. Referring segmentation datasets (**Required for both training and eval**): [(FP-/R-)refcoco(+/g) annotations](https://drive.google.com/file/d/1mA3kcY3QiAZz1Zr89MCKYd7e3LBIwUzl/view?usp=sharing), [COCO images](http://images.cocodataset.org/zips/train2014.zip)

2. Visual Question Answering dataset (**Required for training models for referring segmentation model**): [LLaVA-Instruct-150k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json)

1. Semantic segmentation datasets (**Required for training models for semantic segmentation tasks**): [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), [COCO-Stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip), [PACO-LVIS](https://github.com/facebookresearch/paco/tree/main#dataset-setup), [PASCAL-Part](https://github.com/facebookresearch/VLPart/tree/main/datasets#pascal-part), [COCO Images](http://images.cocodataset.org/zips/train2017.zip)
, [mapillary](https://www.mapillary.com/dataset/vistas)
    Note: For COCO-Stuff, we use the annotation file stuffthingmaps_trainval2017.zip. We only use the PACO-LVIS part in PACO. COCO Images should be put into the `dataset/coco/` directory.

5. Augmented Reasoning segmentation dataset (with false-premise queries): [FP-Aug ReasonSeg](https://drive.google.com/file/d/11WNg1KaV2mk7gTdJRa2aahGqfj4luTDw/view?usp=sharing)

Download them from the above links, and organize them as follows.

```
UGround
├── dataset
│   ├── ade20k
│   │   ├── annotations
│   │   └── images
│   ├── mapillary
│   │   ├── training
│   │   ├── validation
│   │   ├── testing
│   │   ├── config_v1.2.json
│   │   └── config_v2.0.json
│   ├── coco
│   │   └── train2017
│   │       ├── 000000000009.jpg
│   │       └── ...
│   ├── cocostuff
│   │   └── train2017
│   │       ├── 000000000009.png
│   │       └── ...
│   ├── llava_dataset
│   │   └── llava_instruct_150k.json
│   ├── reason_seg
│   │   └── ReasonSeg
│   │       ├── train
│   │       └── val
│   ├── multi_reason_seg    #newly added
│   │   └── MultiReasonSeg
│   │       ├── MUSE_train.json
│   │       ├── MUSE_val.json
│   │       ├── MUSE_test_less.json
│   │       └── MUSE_test_many.json
│   ├── reason_seg_plus  #newly added
│   │   └── ./
│   │       ├── LISA_Plus_Caption.json
│   │       ├── LISA_Plus_Conversations.json
│   │       ├── LISA_Plus_COT.json
│   │       └── LISA_Plus_Instance_Seg.json
│   ├── refer_seg
│   │   ├── images
│   │   |   └── mscoco
│   │   |       └── images
│   │   |           └── train2014
│   │   ├── refclef
│   │   ├── refcoco
│   │   ├── refcoco+
│   │   ├── refcocog
│   │   ├── grefcoco    #newly added
│   │   │    └── ./
│   │   │        ├── grefs(unc).json
│   │   │        └── instances.json
│   │   ├── refzom      #newly added
│   │   │    └── ./
│   │   │        ├── instances.json
│   │   │        ├── Ref_ZOM.p
│   │   │        ├── train2014
│   │   │        └── val2014
│   │   ├── R-refcoco
│   │   ├── R-refcoco+
│   │   ├── R-refcocog
│   │   ├── fprefcoco
│   │   ├── fprefcoco+
│   │   └── fprefcocog
│   ├── vlpart
│   │   ├── paco
│   │   │   └── annotations
│   │   └── pascal_part
│   │       ├── train.json
│   │       └── VOCdevkit
│   ├── clip-vit-large-patch14-336
│   ├── llava-v1.5-7b
│   ├── llava-v1.5-13b
│   ├── llava-v1.6-vicuna-7b  # not necessary
│   ├── llava-v1.6-vicuna-13b # not necessary
│   ├── SESAME-LLaVA-v1.5-7B
│   └── sam_vit_h_4b8939.pth
```

# Face Detection > YOLOv5s
https://universe.roboflow.com/mohamed-traore-2ekkp/face-detection-mik1i

Provided by a Roboflow user
License: CC BY 4.0

## Background Information
This dataset was curated and annotated by [Mohamed Traore](https://blog.roboflow.com/author/mohamed/) and [Justin Brady](https://www.linkedin.com/in/justinbrady) after forking the raw images from the [Roboflow Universe Mask Wearing dataset](https://universe.roboflow.com/joseph-nelson/mask-wearing/4) and remapping the `mask` and `no-mask` classes to `face`.

![Example Image from the Dataset](https://i.imgur.com/jcZfbdF.png)

The main objective is to identify human faces in images or video. However, this model could be used for privacy purposes with changing the output of the bounding boxes to blur the detected face or fill it with a black box.

The original custom dataset *(v1)* is composed of 867 unaugmented (raw) images of people in various environments. 55 of the images are [marked as Null](https://blog.roboflow.com/missing-and-null-image-annotations/) to help with feature extraction and reducing false detections.

Version 2 (v2) includes the augmented and trained version of the model. This version is trained from the COCO model checkpoint to take advantage of [transfer learning](https://blog.roboflow.com/a-primer-on-transfer-learning/) and improve initial model training results.

#### Model Updates:
After a few trainings, and running tests with [Roboflow's webcam model](https://blog.roboflow.com/python-webcam/) and [Roboflow's video inference repo](https://github.com/roboflow-ai/video-inference), it was clear that edge cases like hands sometimes recognized as faces was an issue. I grabbed some images from [Alex Wong's Hand Signs dataset](https://universe.roboflow.com/alex-wong-zz6mu/hand-signs/) (96 images from the dataset) and added them to the project. I uploaded the images, without the annotation files, labeled all the faces, and retrained the model (*version 5*).

The dataset is available under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

Includes images from:
```
@misc{ person-hgivm_dataset,
    title = { person Dataset },
    type = { Open Source Dataset },
    author = { Abner },
    howpublished = { \url{ https://universe.roboflow.com/abner/person-hgivm } },
    url = { https://universe.roboflow.com/abner/person-hgivm },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2021 },
    month = { aug },
    note = { visited on 2022-10-14 },
}
```
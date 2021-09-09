from pycocotools.coco import COCO
from mrcnn import utils
from mrcnn.config import Config
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


class CabbageConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cabbage"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 1
    NUM_CLASSES = 1 + 1 + 1  # bg, cabbage, occluded

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CabbageDataset(utils.Dataset):
    def load_cabbage(self, dataset_dir, subset, class_ids=None,
                     class_map=None, return_coco=False, auto_download=False):
        coco = COCO(
            "{}/annotations/instances_{}.json".format(dataset_dir, subset))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}".format(dataset_dir, subset)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # occludedを含む画像を許可するかどうか

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("cabbage", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "cabbage", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco


if __name__ == "__main__":
    dataset_train = CabbageDataset()
    dataset_train.load_cabbage("cabbage", "train")
    dataset_train.prepare()

    print(dataset_train.image_ids)
    res = []
    for x in dataset_train.image_info:
        occluded = [y['attributes']['occluded'] for y in x['annotations']]
        print(occluded)
        if True in occluded:
            continue
        else:
            res.append(x['id'])

    print(res)

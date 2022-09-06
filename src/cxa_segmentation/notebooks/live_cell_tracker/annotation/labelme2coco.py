# ----------------------------------------------------------------------------
# Created By: labelme2coco authors
# Adapated by: Ke
# Source: https://github.com/fcakyon/labelme2coco
# ---------------------------------------------------------------------------

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import list_files_recursively, load_json, save_json
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)


class labelme2coco:
    def __init__(self):
        raise RuntimeError("Use labelme2coco.convert() or labelme2coco.get_coco_from_labelme_folder() instead.")


def get_coco_from_labelme_folder(
    labelme_folder: str, coco_category_list: List = None, is_image_filename_same_as_json=True, image_file_ext="tif"
) -> Coco:
    """
    Args:
        labelme_folder: folder that contains labelme annotations and image files
        coco_category_list: start from a predefined coco cateory list
    """
    # get json list
    _, abs_json_path_list = list_files_recursively(labelme_folder, contains=[".json"])
    labelme_json_list = abs_json_path_list

    # init coco object
    coco = Coco()

    if coco_category_list is not None:
        coco.add_categories_from_coco_category_list(coco_category_list)

    def _load_image(json_path, labelme_data):
        """load an image
        1) if is_image_filename_same_as_json is True, then load image from the same folder as json, with extension replaced by image_file_ext
        2) if is_image_filename_same_as_json is False, load from labelme's json data['imagePath']
        """
        image_path = str(Path(labelme_folder) / labelme_data["imagePath"])
        if is_image_filename_same_as_json:
            image_path = json_path.replace(".json", "." + image_file_ext)
        print("loading image from:", image_path)
        return Image.open(image_path), image_path

    # parse labelme annotations
    category_ind = 0
    for json_path in tqdm(labelme_json_list, "Converting labelme annotations to COCO format"):
        data = load_json(json_path)
        # get image size
        image, image_path = _load_image(json_path, data)
        width, height = image.size

        # init coco image
        coco_image = CocoImage(file_name=image_path, height=height, width=width)
        # iterate over annotations
        for shape in data["shapes"]:
            # set category name and id
            category_name = shape["label"]
            category_id = None
            for (
                coco_category_id,
                coco_category_name,
            ) in coco.category_mapping.items():
                if category_name == coco_category_name:
                    category_id = coco_category_id
                    break
            # add category if not present
            if category_id is None:
                category_id = category_ind
                coco.add_category(CocoCategory(id=category_id, name=category_name))
                category_ind += 1
            # parse bbox/segmentation
            if shape["shape_type"] == "rectangle":
                x1 = shape["points"][0][0]
                y1 = shape["points"][0][1]
                x2 = shape["points"][1][0]
                y2 = shape["points"][1][1]
                coco_annotation = CocoAnnotation(
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    category_id=category_id,
                    category_name=category_name,
                )
            elif shape["shape_type"] == "polygon":
                segmentation = [np.asarray(shape["points"]).flatten().tolist()]
                coco_annotation = CocoAnnotation(
                    segmentation=segmentation,
                    category_id=category_id,
                    category_name=category_name,
                )
            else:
                raise NotImplementedError(f'shape_type={shape["shape_type"]} not supported.')
            coco_image.add_annotation(coco_annotation)
        coco.add_image(coco_image)

    return coco


def convert(
    labelme_folder: str,
    export_dir: str = "runs/labelme2coco/",
    train_split_rate: float = 1,
    is_image_filename_same_as_json=True,
    image_file_ext="tif",
):
    """
    Args:
        labelme_folder: folder that contains labelme annotations and image files
        export_dir: path for coco jsons to be exported
        train_split_rate: ration fo train split
    """
    coco = get_coco_from_labelme_folder(
        labelme_folder, is_image_filename_same_as_json=is_image_filename_same_as_json, image_file_ext=image_file_ext
    )
    if train_split_rate < 1:
        result = coco.split_coco_as_train_val(train_split_rate)
        # export train split
        save_path = str(Path(export_dir) / "train.json")
        save_json(result["train_coco"].json, save_path)
        logger.info(f"Training split in COCO format is exported to {save_path}")
        # export val split
        save_path = str(Path(export_dir) / "val.json")
        save_json(result["val_coco"].json, save_path)
        logger.info(f"Validation split in COCO format is exported to {save_path}")
    else:
        save_path = str(Path(export_dir) / "dataset.json")
        save_json(coco.json, save_path)
        logger.info(f"Converted annotations in COCO format is exported to {save_path}")


if __name__ == "__main__":
    labelme_folder = "tests/data/labelme_annot"
    coco = get_coco_from_labelme_folder(labelme_folder)

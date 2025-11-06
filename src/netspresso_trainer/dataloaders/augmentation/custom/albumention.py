from typing import Any, Dict, List, Optional

import albumentations as A
import numpy as np
import PIL.Image as Image
import random


def _pil_to_ndarray(image: Image.Image) -> np.ndarray:
    return np.asarray(image)


def _ndarray_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class Albumention:
    visualize = True

    def __init__(self, transforms: List[Dict[str, Any]]):
        self.transforms_cfg = transforms
        self._has_mosaic = any(t.get('name') == 'Mosaic' for t in transforms)
        self._pipeline = self._build_pipeline(transforms)
    def _build_tranform(self, tranform_config):
        name = tranform_config.get('name')

        params = {k: v for k, v in tranform_config.items() if k != 'name'}
        # Convert common param aliases
        if 'p' not in params:
            params['p'] = 1.0
        a_cls = getattr(A, name)
        return a_cls(**params)
    
    def _build_pipeline(self, transforms: List[Dict[str, Any]]):
        a_list = []
        for t in transforms:
            name = t.get('name')
            if name is None:
                continue
            if name=="OneOf":
                sub_transforms = t.get('transforms', [])
                sub_list = []
                for sub_t in sub_transforms:
                    sub_name  = sub_t.get('name')
                    if sub_name is None:
                        continue
                    sub_tranform = self._build_tranform(sub_t)
                    sub_list.append(sub_tranform)
                prob = t.get('p', 1.0)
                a_list.append(A.OneOf(sub_list, p=prob))
            else:
                tranform = self._build_tranform(t)
                a_list.append(tranform)
            
        return A.Compose(a_list, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        img_np = _pil_to_ndarray(image)

        bboxes = []
        class_labels = []
        if bbox is not None:
            bboxes = bbox.tolist() if hasattr(bbox, 'tolist') else bbox
            class_labels = label.reshape(-1).tolist() if hasattr(label, 'reshape') else (label if label is not None else [])
        call_kwargs: Dict[str, Any] = { 'image': img_np, 'bboxes': bboxes, 'class_labels': class_labels }

        # Provide required metadata for Albumentations Mosaic if present
        if self._has_mosaic:
            if dataset is None:
                raise ValueError("Albumentations Mosaic requires dataset to sample additional images.")
            # Collect 3 additional items (image, label, bbox)
            items = []
            for _ in range(3):
                rand_idx = random.randint(0, len(dataset) - 1)
                im_i, lb_i, bx_i = dataset.pull_item(rand_idx)
                items.append((im_i, lb_i, bx_i))

            mosaic_images = [img_np]
            mosaic_bboxes = [bboxes if bboxes is not None else []]
            mosaic_class_labels = [class_labels if class_labels is not None else []]
            for im_i, lb_i, bx_i in items:
                im_np_i = _pil_to_ndarray(im_i)
                mosaic_images.append(im_np_i)
                bb_i = bx_i.tolist() if hasattr(bx_i, 'tolist') else (bx_i if bx_i is not None else [])
                lb_list_i = lb_i.reshape(-1).tolist() if hasattr(lb_i, 'reshape') else (lb_i if lb_i is not None else [])
                mosaic_bboxes.append(bb_i)
                mosaic_class_labels.append(lb_list_i)

            call_kwargs['mosaic_metadata'] = {
                'images': mosaic_images,
                'bboxes': mosaic_bboxes,
                'class_labels': mosaic_class_labels,
            }

        result = self._pipeline(**call_kwargs)
        out_img = _ndarray_to_pil(result['image'])

        out_bbox = np.array(result.get('bboxes', []), dtype=np.float32) if len(result.get('bboxes', [])) > 0 else bbox
        out_label = np.array(result.get('class_labels', []), dtype=np.int64).reshape(-1, 1) if len(result.get('class_labels', [])) > 0 else label

        return out_img, out_label, mask, out_bbox, keypoint

    def __repr__(self):
        return f"Albumention(transforms={self.transforms_cfg})"



from pathlib import Path
import cv2
import numpy as np
import torch as tc
from ultralytics import YOLO
import sys

# np.set_printoptions(threshold=sys.maxsize)


def display_masks(apples, branches, metal_wire, background, rgb):
    (H, W) = np.array(branches).shape
    apple = np.zeros((H, W)).astype("uint8")
    for appl in apples:
        apple = np.maximum(appl, apple)
    apple = (255 * apple).clip(0, 255).astype("uint8")
    branches = (190 * branches).clip(0, 255).astype("uint8")
    metal_wire = (127 * metal_wire).clip(0, 255).astype("uint8")
    background = (64 * background).clip(0, 255).astype("uint8")

    labeled_image = apple + branches + background + metal_wire

    labeled_image_rgb = cv2.cvtColor(labeled_image, cv2.COLOR_GRAY2RGB)
    imagefinal = np.concatenate((labeled_image_rgb, rgb), axis=1)

    cv2.imshow("labeled image", imagefinal)
    print('Press "q" to exit')
    cv2.waitKey(0) & 0xFF == ord("q")


def detect_masks(
    model_path="/workspaces/devcontainer-gui-gpu/scripts/1.Training/runs/segment/train3/weights/best.pt",
    image_obj="/workspaces/devcontainer-gui-gpu/scripts/color.png",
    show_opt=True,
    conf_apples=0.5,
    conf_branches=0.25,
    conf_metal_wire=0.1,
    conf_background=0.35,
    save_opt=False,
    apple_mask_shrink_pixels=10  # New parameter for shrinking apple masks
):
    m = YOLO(model=model_path)
    res = m.predict(image_obj, save=save_opt)
    
    if isinstance(image_obj, str):
        image_ = cv2.imread(image_obj)
    else:
        image_ = image_obj
    (H, W, _) = np.array(image_).shape

    apple_mss = []
    branches_ms = np.zeros((H, W), dtype=np.uint8)
    metal_wire_ms = np.zeros((H, W), dtype=np.uint8)
    background_ms = np.zeros((H, W), dtype=np.uint8)

    # Iterate detection results
    for r in res:
        img = np.copy(r.orig_img)
        
        # Iterate each object contour
        for ci, c in enumerate(r):      
            label = c.names[c.boxes.cls.tolist()[0]]
            confidence = c.boxes.conf.cpu().numpy()[0].astype(float)

            if not c.masks or not c.masks.xy:
                print(f"Skipping contour {ci} due to empty mask")
                continue

            try:
                contour = c.masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
                if contour.size == 0:
                    print(f"Skipping contour {ci} due to empty contour")
                    continue

                b_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                mask = np.array(b_mask).clip(0, 1).astype(np.uint8)
                
                if label == "apple" and confidence >= conf_apples:
                    # Shrink the apple mask
                    kernel = np.ones((apple_mask_shrink_pixels, apple_mask_shrink_pixels), np.uint8)
                    eroded_mask = cv2.erode(mask, kernel, iterations=1)
                    apple_mss.append(eroded_mask)
                    # start debug
                    """ mask = (255 * mask).clip(0, 255).astype("uint8")
                    eroded_masks = (120 * eroded_mask).clip(0, 255).astype("uint8")
                    image = eroded_masks + mask
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    imagefinal = np.concatenate((rgb_image, image_), axis=1) 
                    cv2.imshow("apple diff", imagefinal)
                    print('Press "q" to exit')
                    cv2.waitKey(0) & 0xFF == ord("q") """
                    # end debug
                    
                elif label == "branches" and confidence >= conf_branches:
                    branches_ms = np.maximum(branches_ms, mask)
                elif label == "metal_wire" and confidence >= conf_metal_wire:
                    metal_wire_ms = np.maximum(metal_wire_ms, mask)
                elif label == "background" and confidence >= conf_background:
                    background_ms = np.maximum(background_ms, mask)

            except Exception as e:
                print(f"Error processing contour {ci}: {str(e)}")

    if show_opt:
        display_masks(
            apples=apple_mss,
            background=branches_ms,
            metal_wire=metal_wire_ms,
            branches=background_ms,
            rgb = image_
        )
    
    return apple_mss, branches_ms, metal_wire_ms, background_ms



if __name__ == "__main__":

    apples_masks, background_masks, metal_wire_masks, branches_masks = detect_masks(
        show_opt=False
    )
    display_masks(
        apples=apples_masks,
        background=background_masks,
        metal_wire=metal_wire_masks,
        branches=branches_masks,
    )

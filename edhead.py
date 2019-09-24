import sys

import dlib
import numpy as np


MAX_DIMENSION = 1024
WIDTH_MARGIN = 0.18
TOP_SHIFT = 0.2


class Edhead(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.overlay = dlib.load_rgb_image('A1opZLgQdoL.jpg')

    def _preprocess(self, image):
        """Load image and resize if any dimension is greater than MAX_DIMENSION"""
        img = dlib.load_rgb_image(image)
        print(f"Input image shape {img.shape}")
        max_dim = img.shape[0] if img.shape[0] > img.shape[1] else img.shape[1]
        if max_dim > MAX_DIMENSION:
            img = dlib.resize_image(img, MAX_DIMENSION / max_dim)
            print(f"Resized image shape {img.shape}")
        return img

    def edheadify(self, image, view=False):
        """Edheadify an image
        image: Path to a portrait image.
            For best result, check out https://unsplash.com/s/photos/portrait for inspiration.
            Output image saved with suffix '_edhead.jpg'.
        view: If True, show output image.
        """
        img = self.preprocess(image)
        dets, scores, _ = self.detector.run(img, 1, -1)
        most_likely = np.argmax(scores)
        d = dets[most_likely]

        print("Left: {} Top: {} Right: {} Bottom: {}".format(d.left(), d.top(), d.right(), d.bottom()))

        face_center = [round((d.top() + d.bottom()) / 2.0), round((d.left() + d.right()) / 2.0)]
        face_center[0] = round(face_center[0] / (1 + TOP_SHIFT))

        padded_width = round((d.right() - d.left()) * (1 + 2 * WIDTH_MARGIN))

        scale = padded_width / overlay.shape[1]
        overlay = dlib.resize_image(overlay, scale)
        height, width, _ = overlay.shape

        overlay_top_left = (max([0, face_center[0] - height // 2]), face_center[1] - width // 2)

        img[overlay_top_left[0]: overlay_top_left[0] + height,
            overlay_top_left[1]: overlay_top_left[1] + width] = overlay

        if view:
            win = dlib.image_window()
            win.clear_overlay()
            win.set_image(img)
            dlib.hit_enter_to_continue()

        dlib.save_image(img, image + '_edhead.jpg')


if __name__ == "__main__":
    edhead = Edhead()
    input_image = sys.argv[1]
    edhead.edheadify(input_image)

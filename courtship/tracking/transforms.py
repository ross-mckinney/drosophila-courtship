
import numpy as np

import skimage.transform as skt
from skimage.color import (
    rgb2gray,
    gray2rgb
)

def center_image_in_frame(image, centroid, size):
    """Centers a point in an image.

    Parameters
    ----------
    image : 2d np.ndarray
        Image.

    centroid : 2-tuple of int/float
        Point to center. Coordinates should be (rr, cc).

    size : 2-tuple of int
        Size of returned-image.

    Returns
    -------
    np.ndarray of shape [size[0], size[1]].
    """
    image = image.copy()

    # get distances from centroid to sides of box
    height, width = image.shape
    top2centroid = int(centroid[0])
    left2centroid = int(centroid[1])
    bottom2centroid = int(height - centroid[0])
    right2centroid = int(width - centroid[1])

    # if the centroid is in the upper half of the bbox, we want to
    # expand the upper half of the box.
    if top2centroid < bottom2centroid:
        added_top = (bottom2centroid-top2centroid)
        new_bbox_height = height + added_top
        # if the centroid is in the left half of the bbox, we want to
        # expand the left side of the box
        if left2centroid < right2centroid:
            added_left = (right2centroid-left2centroid)
            new_bbox_width = width + added_left
            mask = np.zeros(
                shape=(new_bbox_height, new_bbox_width), dtype=np.uint8)
            mask[added_top:,added_left:] = image
        else:
            added_right = (left2centroid - right2centroid)
            new_bbox_width = width + added_right
            mask = np.zeros(
                shape=(new_bbox_height, new_bbox_width), dtype=np.uint8)
            mask[added_top:,:width] = image

    # if the centroid is in the bottom half of the bbox
    else:
        added_bottom = (top2centroid - bottom2centroid)
        new_bbox_height = height + added_bottom
        # if the centroid is in the left half of the bbox, we want to
        # expand the left half of the box
        if left2centroid < right2centroid:
            added_left = (right2centroid-left2centroid)
            new_bbox_width = width + added_left
            mask = np.zeros(
                shape=(new_bbox_height, new_bbox_width), dtype=np.uint8)
            mask[:height,added_left:] = image
        else:
            added_right = (left2centroid - right2centroid)
            new_bbox_width = width + added_right
            mask = np.zeros(
                shape=(new_bbox_height, new_bbox_width), dtype=np.uint8)
            mask[:height,:width] = image

    return trim_image2d(mask, size)


def center_image_in_frame3d(image, centroid, size):
    """Centers a point in an image.

    Parameters
    ----------
    image : 3d np.ndarray
        Image.

    centroid : 2-tuple of int/float
        Point to center. Coordinates should be (rr, cc).

    size : 2-tuple of int
        Size of returned-image.

    Returns
    -------
    np.ndarray of shape [size[0], size[1]].
    """
    image = image.copy()

    # get distances from centroid to sides of box
    height, width, channels = image.shape
    top2centroid = int(centroid[0])
    left2centroid = int(centroid[1])
    bottom2centroid = int(height - centroid[0])
    right2centroid = int(width - centroid[1])

    # if the centroid is in the upper half of the bbox, we want to
    # expand the upper half of the box.
    if top2centroid < bottom2centroid:
        added_top = (bottom2centroid-top2centroid)
        new_bbox_height = height + added_top
        # if the centroid is in the left half of the bbox, we want to
        # expand the left side of the box
        if left2centroid < right2centroid:
            added_left = (right2centroid-left2centroid)
            new_bbox_width = width + added_left
            mask = np.zeros(
                shape=(new_bbox_height, new_bbox_width, channels),
                dtype=np.uint8)
            mask[added_top:,added_left:, :] = image
        else:
            added_right = (left2centroid - right2centroid)
            new_bbox_width = width + added_right
            mask = np.zeros(
                shape=(new_bbox_height, new_bbox_width, channels),
                dtype=np.uint8)
            mask[added_top:,:width, :] = image

    # if the centroid is in the bottom half of the bbox
    else:
        added_bottom = (top2centroid - bottom2centroid)
        new_bbox_height = height + added_bottom
        # if the centroid is in the left half of the bbox, we want to
        # expand the left half of the box
        if left2centroid < right2centroid:
            added_left = (right2centroid-left2centroid)
            new_bbox_width = width + added_left
            mask = np.zeros(
                shape=(new_bbox_height, new_bbox_width, channels),
                dtype=np.uint8)
            mask[:height,added_left:, :] = image
        else:
            added_right = (left2centroid - right2centroid)
            new_bbox_width = width + added_right
            mask = np.zeros(
                shape=(new_bbox_height, new_bbox_width, channels),
                dtype=np.uint8)
            mask[:height,:width, :] = image

    return trim_image3d(mask, size)


def center_bbox_in_frame(image, centroid, bbox, size):
    """Places a box into the center of an image.

    Parameters
    ----------
    image : 2d np.ndarray
        Image containing box to center.

    centroid : 2-tuple of float/int
        Center of box. Coordinates should be (rr, cc) (pixel units).

    bbox : 4-tuple of int
        Pixel-coordinates of box to center (min_row, min_col, max_row, max_col).

    size : 2-tuple of int
        Size of image to return.

    Returns
    -------
    np.ndarray of shape [size[0], size[1]]
    """
    image = image.copy()

    # get distances from centroid to sides of box
    top2centroid = int(centroid[0] - bbox[0])
    left2centroid = int(centroid[1] - bbox[1])
    bottom2centroid = int(bbox[2] - centroid[0])
    right2centroid = int(bbox[3] - centroid[1])
    height = int(bbox[2] - bbox[0])
    width = int(bbox[3] - bbox[1])

    # check to see if the images is a bbox image or a complete image
    if image.shape[0] != height:
        image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    # if the centroid is in the upper half of the bbox, we want to
    # expand the upper half of the box.
    if top2centroid < bottom2centroid:
        added_top = (bottom2centroid-top2centroid)
        new_bbox_height = height + added_top
        # if the centroid is in the left half of the bbox, we want to
        # expand the left side of the box
        if left2centroid < right2centroid:
            added_left = (right2centroid-left2centroid)
            new_bbox_width = width + added_left
            mask = np.zeros(
                shape=(new_bbox_height, new_bbox_width), dtype=np.uint8)
            mask[added_top:,added_left:] = image
        else:
            added_right = (left2centroid - right2centroid)
            new_bbox_width = width + added_right
            mask = np.zeros(
                shape=(new_bbox_height, new_bbox_width), dtype=np.uint8)
            mask[added_top:,:width] = image

    # if the centroid is in the bottom half of the bbox
    else:
        added_bottom = (top2centroid - bottom2centroid)
        new_bbox_height = height + added_bottom
        # if the centroid is in the left half of the bbox, we want to
        # expand the left half of the box
        if left2centroid < right2centroid:
            added_left = (right2centroid-left2centroid)
            new_bbox_width = width + added_left
            mask = np.zeros(
                shape=(new_bbox_height, new_bbox_width), dtype=np.uint8)
            mask[:height,added_left:] = image
        else:
            added_right = (left2centroid - right2centroid)
            new_bbox_width = width + added_right
            mask = np.zeros(
                shape=(new_bbox_height, new_bbox_width), dtype=np.uint8)
            mask[:height,:width] = image

    image = reshape_image(mask, size)
    return image


def reshape_image(image, size):
    """ reshapes an image smaller than 200px x 200px to a specified size.
        size = (height, width) -- tuple
    """
    if image.shape != size:
        mask = np.zeros(shape=size, dtype=np.uint8)
        height, width = image.shape

        left = int((size[1] - width)/2.)
        right = int(left + width)
        top = int((size[0] - height)/2.)
        bottom = int(top + height)

        mask[top:bottom, left:right] = image
        image = mask
    return image


def trim_image2d(image, size):
    """Trims an image to a specified size.

    Trimming always occurs with respect to the center of the image.

    Parameters
    ----------
    image : 2d np.ndarray
        Image to trim.

    size : 2-tuple of int
        Specified as (height, width) in pixels.

    Returns
    -------
    np.ndarray of shape [size[0], size[1]]
    """
    image_height, image_width = image.shape
    height_trim = int((image_height - size[0])/2)
    width_trim = int((image_width - size[1])/2)
    new_image = image[
        height_trim:(image_height - height_trim),
        width_trim:(image_width - width_trim)
        ]

    if new_image.shape != size:
        if new_image.shape[0] > size[0]: #too many rows
            remove_rows = new_image.shape[0] - size[0]
            new_image = new_image[:-remove_rows,:]
        if new_image.shape[1] > size[1]: #too many cols
            remove_cols = new_image.shape[1] - size[1]
            new_image = new_image[:,:-remove_cols]
    return new_image


def trim_image3d(image, size):
    """Trims an image to a specified size.

    Trimming always occurs with respect to the center of the image.

    Parameters
    ----------
    image : 3d np.ndarray
        Image to trim.

    size : 2-tuple of int
        Specified as (height, width) in pixels.

    Returns
    -------
    np.ndarray of shape [size[0], size[1], 3]
    """
    image_height, image_width, channels = image.shape
    height_trim = int((image_height - size[0])/2)
    width_trim = int((image_width - size[1])/2)
    new_image = image[
        height_trim:(image_height - height_trim),
        width_trim:(image_width - width_trim),
        :
        ]

    if new_image.shape != size:
        if new_image.shape[0] > size[0]: # too many rows
            remove_rows = new_image.shape[0] - size[0]
            new_image = new_image[:-remove_rows,:, :]
        if new_image.shape[1] > size[1]: # too many cols
            remove_cols = new_image.shape[1] - size[1]
            new_image = new_image[:,:-remove_cols, :]

    return new_image


def crop_and_rotate(image, region, shape, color=None):
    """Crops an image to a specific shape centered on a specific region.

    Also rotates entire image so that the orientation contained
    within the region is aligned along the horizontal axis.

    Parameters
    ----------
    image : 2D or 3D np.ndarray of type np.uint8
        Image to crop.

    region : skimage.measure.RegionProps object
        Region of within image to crop around.

    shape : 2-tuple of int
        Height x width to crop image to.

    color : 3-tuple of uint8 (Blue, Green, Red) or None
        Color to shade coordinates contained within region. If None,
        no shading will be applied. Note that if a color is given, 
        and an input image is 2D, it will be transformed to 3D

    Returns
    -------
    cropped_image : 2D or 3D np.ndarray | shape = [shape]
    """
    revert = False
    image = image.copy()

    # convert all images to 3D.
    if len(image.shape) == 2:
        image = gray2rgb(image).astype(np.uint8)
        revert = True
    elif len(image.shape) == 3:
        pass
    else:
        raise AttributeError("Image needs to be either 2D or 3D.")

    if color is not None:
        revert = False
        image[region.coords, :] = color

    # rotate image around region centroid and center.
    image = skt.rotate(
        image=image, 
        angle=-region.orientation * (180 / np.pi),
        center=region.centroid,
        preserve_range=True
        ).astype(np.uint8)

    R = np.array([
        [np.cos(-region.orientation), -np.sin(-region.orientation)],
        [np.sin(-region.orientation), np.cos(-region.orientation)]
            ])

    rotated_centroid = np.dot(R, np.array(region.centroid).T).T

    image = center_image_in_frame3d(
        image, 
        centroid=rotated_centroid, 
        size=shape)

    if revert:
        image = rgb2gray(image).astype(np.uint8)

    return image
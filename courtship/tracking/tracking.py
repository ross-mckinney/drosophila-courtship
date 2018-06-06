"""
"""
import os

import numpy as np
from scipy.spatial.distance import pdist

import skimage.measure as sk_measure
import skimage.morphology as sk_morph
import skimage.transform as sk_transform

from .arena import CircularArena
from .errors import NoPropsDetected
from .female import Female
from .transforms import (
    center_image_in_frame,
    trim_image2d
)


def high_pass_threshold_binary(image, threshold):
    """Returns a binary image where all pixels below a threshold are set to 0.

    Parameters
    ----------
    image : 2D np.ndarray of type np.uint8
        Image to threshold

    threshold : int
        Pixels with a value below this threshold
        will be set to 0. All other pixels will be set to 1.

    Returns
    -------
    binary_image : np.ndarray of type np.uint8 | shape = [image.shape]
        Binary image holding high pass threshold.
    """
    mask = np.zeros_like(image, dtype = np.uint8)
    mask[np.where(image > threshold)] = 1
    assert mask.dtype == np.uint8, "Image not of type np.uint8"
    return mask


def high_pass_threshold(image, threshold):
    """Sets pixels below a specified threshold to 0.

    Parameters
    ----------
    image : 2D np.ndarray of type np.uint8
        Image to threshold

    threshold : int
        Pixels with a value below this threshold
        will be set to 0.

    Returns
    -------
    thresholded_image : np.ndarray of type np.uint8 | shape = [image.shape]
        Thresholded image.
    """
    mask1 = high_pass_threshold_binary(image, threshold)
    mask2 = np.zeros_like(image)
    mask2[np.where(mask1)] = image[np.where(mask1)]
    return mask2


def low_pass_threshold_binary(image, threshold):
    """Returns a binary image where all pixels above a threshold are set to 0.

    Parameters
    ----------
    image : 2D np.ndarray of type np.uint8
        Image to threshold

    threshold : int
        Pixels with a value above this threshold
        will be set to 0. All other pixels will be set to 1.

    Returns
    -------
    binary_image : np.ndarray of type np.uint8 | shape = [image.shape]
        Binary image holding low pass threshold.
    """
    mask = np.zeros_like(image)
    mask[np.where(image < threshold)] = 1
    # assert mask.dtype == np.uint8, "Image not of type np.uint8"
    return mask.astype(np.uint8)


def low_pass_threshold(image, threshold):
    """Sets all pixels above a specified threshold to 0.

    Parameters
    ----------
    image : 2D np.ndarray of type np.uint8
        Image to threshold

    threshold : int
        Pixels with a value above this threshold
        will be set to 0.

    Returns
    -------
    thresholded_image : np.ndarray of type np.uint8 | shape = [image.shape]
        Thresholded image.
    """
    mask1 = low_pass_threshold_binary(image, threshold)
    mask2 = np.zeros_like(image)
    mask2[np.where(mask1)] = image[np.where(mask1)]
    return mask2


def subtract_background(image, background_image):
    """Subtracts background image from a specified image.

    Returns
    -------
    bs_image : np.ndarray of type np.int | shape = [image.shape]
        Background-subtracted image.
    """
    image = image.copy().astype(np.int)
    background = background_image.copy().astype(np.int)

    bs_image = image - background
    return bs_image.astype(np.int)


def find_male(image, female, arena, lp_threshold):
    """Gets the region properties which describe the position of a male.

    Parameters
    ----------
    image : np.ndarray of type np.uint8
        Image to find male within

    female : female.Female object
        Object which describes female.

    arena : arena.CircularArena object
        Object which describes arena.

    lp_threshold : int
        Low pass threshold to use for filtering out high pixels.

    Returns
    -------
    male_props : skimage.RegionProps object or None
        Properties describing region occupied by male fly. Returns None
        if no region properties were found within the input image.
    """
    try:
        binary_image = low_pass_threshold_binary(image, lp_threshold)
        female_mask = female.get_female_mask()
        arena_mask = arena.get_arena_mask()
        binary_image[np.where(female_mask)] = 0
        binary_image[np.where(arena_mask == 0)] = 0

        labeled_image = sk_measure.label(binary_image)
        props = sk_measure.regionprops(labeled_image)

        if len(props) == 0:
            raise NoPropsDetected("No region props detected.")

        #sort list of region properties by area
        areas = [prop.area for prop in props]

        #assume that the male is the regionprop with the largest area
        male_ix = np.argsort(areas)[-1]

        return props[male_ix]

    #Catch errors which are due to female or arena not being of the correct type.
    except (TypeError, AttributeError):
        return None


def find_female(image, female, lp_threshold):
    """Gets the region properties which describe the position of a female.

    Parameters
    ----------
    image : np.ndarray
        Image within which to find female.

    female : female.Female object
        Female defined by user.

    lp_threshold : int
        Low pass threshold to use for filtering out high pixels.

    Returns
    -------
    female_props : skimage.measure.RegionProps object

    female_head : length 2 tuple (rr, cc)
        Position of female's head.

    female_rear : length 2 tuple (rr, cc)
        Position of female's rear.
    """
    binary_image = low_pass_threshold_binary(image, lp_threshold)
    female_mask = female.get_female_mask()
    binary_image[np.where(female_mask == 0)] = 0

    labeled_image = sk_measure.label(binary_image)
    props = sk_measure.regionprops(labeled_image)
    areas = [prop.area for prop in props]

    if len(props) == 0:
        raise NoPropsDetected("No region props detected.")

    female_ix = np.argsort(areas)[-1]
    female_props = props[female_ix]

    # find the minimum & maximum distances between the female's head and
    # a point in female_props.coords

    female_head_pts = np.vstack((
            female.head,
            female_props.coords
        ))

    # condensed distance matrix
    head_pt_dists = pdist(female_head_pts)[:female_props.coords.shape[0]]

    head_ix = np.argsort(head_pt_dists)[0] # closest point == head
    rear_ix = np.argsort(head_pt_dists)[-1] # furthest point == rear

    female_head = female_props.coords[head_ix, :]
    female_rear = female_props.coords[rear_ix, :]

    return female_props, female_head, female_rear


def set_female_props(female, props, head, rear, i):
    """Sets given RegionProps to female Fly object.

    Parameters
    ----------
    female : courtanal.objects.fly.Fly object
        Female fly to be set.

    props : skimage.measure.RegionProps object
        RegionProps to set to female.

    head : np.ndarray of size 2
        Position of female head.

    rear : np.ndarray of size 2
        Position of female rear.

    i : int
        Which frame/position in female array to set.
    """
    female.body.centroid.row[i], female.body.centroid.col[i] = props.centroid
    female.body.head.row[i], female.body.head.col[i] = head
    female.body.rear.row[i], female.body.rear.col[i] = rear
    female.body.orientation[i] = props.orientation
    female.body.major_axis_length[i] = props.major_axis_length
    female.body.minor_axis_length[i] = props.minor_axis_length
    # female.body.area[i] = props.area


def tighten_female_ellipse(female, female_props):
    """Updates a female object with specified properties.

    This function is designed to decrease (or increase) the
    user-defined ellipse surrounding a fixed female. The
    updated ellipse parameters will be set as follows:

    female major axis length <-
        female_props.major_axis_length + 10% of female_props.major_axis_length
    female minor axis length <-
        female_props.minor_axis_length + 10% of female_props.minor_axis_length

    Parameters
    ----------
    female : gui.objects.female.Female object
        Female object to update.

    female_props : skimage.measure.RegionProps object
        Object containing new properties.
    """
    # uncomment this and the print function if you want to see how these
    # parameters were changed.
    # oc = female.center
    # omaj = female.maj_ax_rad
    # omin = female.min_ax_rad
    # oo = female.orientation

    female.center = (int(female_props.centroid[0]), int(female_props.centroid[1]))
    female.maj_ax_rad = int(np.ceil(
        female_props.major_axis_length / 2. + female_props.major_axis_length * 0.1))
    female.min_ax_rad = int(np.ceil(
        female_props.minor_axis_length / 2. + female_props.minor_axis_length * 0.1))
    female.orientation = int(female_props.orientation * 180 / np.pi)

    # print (
    # 	'Updated center: {} -> {} \nUpdated maj ax len: {} -> {}'.format(
    # 			oc, female.center,
    # 			omaj, female.maj_ax_rad,
    # 		)

    # 	+

    # 	'\nUpdated min ax len: {} -> {}\nUpdated ori: {} -> {}'.format(
    # 			omin, female.min_ax_rad,
    # 			oo, female.orientation
    # 		)
    # 	)


def _get_male_body_axis(in_shape, male_props, rotation, out_shape, head='Right'):
    """Finds the coordinates that define the male's head & rear.

    This function works by getting a binary image of the male fly's body
    and then determining the right-most and left-most indices of positive
    pixels.

    Update: This now just works by assuming a male is centered &
    oriented to either the left or the right, and adding or subtracting 1/2
    of the size of the output image to 1/2 the length of the major axes to
    define the head & rear coords.

    Parameters
    ----------
    in_shape : tuple (int, int)
        Shape of initial image (video frame) containing male.

    male_props : skimage.measure.RegionProps object
        Properties that define the male to locate. This should
        be calculated from centroid.find_male_tight().

    rotation : int
        Angle in degrees that the image needs to be rotated to
        align the male correctly.

    out_shape : tuple (int, int)
        Shape to trim input image to.

    head : string (default = 'Right')
        Which way defines the head?

    Returns
    -------
    head_position : 1D np.array of length [2]
        Position of male head. This is relative to the specified out_shape.

    rear_position : 1D np.array of length [2]
        Position of male rear. This is also relative to the specified out_shape.
    """

    # get a body-only image of a male
    # image = get_body_image(in_shape, male_props, rotation, out_shape)

    # rr, cc = np.where(image)
    # if rr.size == 0 or cc.size == 0:
    # 	return np.array([0, 0], dtype = np.int), np.array([0, 0], dtype = np.int)

    # update so that head position is based off of fitted ellipse.
    right = np.array(
        [out_shape[0] / 2, out_shape[1] / 2 + male_props.major_axis_length / 2]
        ).astype(np.int)
    left = np.array(
        [out_shape[0] / 2, out_shape[1] / 2 - male_props.major_axis_length / 2]
        ).astype(np.int)

    if head == 'Right':
        return right, left

    return left, right


def _get_props(image):
    """Returns region props for the largest region in a binary image.

    Parameters
    ----------
    image : np.ndarray
        Binary image

    Returns
    -------
    props : skimage.measure.RegionProps object or -1.
        Returns -1 if no regions were detected in image.

    """
    labeled_image = sk_measure.label(image)
    props = sk_measure.regionprops(labeled_image)

    if len(props) > 0:
        # sort the props according to area
        areas = [prop.area for prop in props]
        ix = np.argsort(areas)[-1]

        return props[ix]

    raise NoPropsDetected("No region props detected.")


def _orient_wing_image(image, orientation, head='Right',
    return_rotation_degrees=False):
    """Given a binary, wing-only image, this function determines which
    way a fly is facing.

    Parameters
    ----------
    image : np.ndarray
        2D, binary wing image. The body should already be subtracted out.
        Futher, the region of interest should be at the center of the image.

    orientation : float (-pi/2, pi/2)
        Angle needed to rotate fly so that it is aligned along the horizontal.
        This angle should be the same as that contained within
        skimage.RegionProps.orientation for a specific region.

    head : string ('Right' or 'Left')
        Should the returned image contain a fly facing to the left or
        to the right?

    return_rotation_degrees : bool
        Should the angle needed to rotate the image into its proper
        orientation be returned?

    Returns
    -------
    rotated_image : np.ndarray of type np.uint8 | shape = [image.shape]
        Rotated binary image.

    rotation_degrees : int
        Angle in degrees needed to rotate image so that the fly contained
        within it is facing either to the right or to the left.
    """
    rotation_angle = -orientation * 180 / np.pi

    # rotate image into two possible orientations
    im1 = sk_transform.rotate(image, rotation_angle)
    im2 = sk_transform.rotate(image, rotation_angle + 180)

    # integrate over the vertical axis of the image
    sum1 = np.sum(im1, axis = 0)
    sum2 = np.sum(im2, axis = 0)

    # simply: if the left half of the image contains more pixels,
    # then the fly is facing toward the right. This doesn't always work,
    # as the fly can extend it's wings to >90 degrees from its
    # antero-posterior axis; but it works well as an approximation.
    if np.sum(sum1[:sum1.size / 2]) > np.sum(sum2[:sum2.size / 2]):
        facing = 'Right'
    else:
        facing = 'Left'

    if facing == head:
        im_ang_list = [(im1 * 255).astype(np.uint8), rotation_angle]
    else:
        im_ang_list = [(im2 * 255).astype(np.uint8), rotation_angle + 180]

    if return_rotation_degrees:
        return im_ang_list

    return im_ang_list[0]


def _split_image(image, axis='Horizontal'):
    """Splits an image into two halves and returns each half.

    Parameters
    ----------
    image : np.ndarray
        Image to split in half.

    axis : string (default = 'Horizontal')
        Which axis to split the image. If 'Horizontal', upper and lower halves
        of the specified image are returned. If 'Vertical', left and right
        halves of the specified image are returned.

    Returns
    -------
    half1, half2 : np.ndarrays of type np.uint8
        Image halves, either upper and lower or left and right.
    """
    nrows, ncols = image.shape
    if axis == 'Horizontal':
        half1 = image[:nrows/2, :] # upper half
        half2 = image[nrows/2:, :] # lower half
        return half1, half2

    half1 = image[:, :ncols/2] # left half
    half2 = image[:, ncols/2:] # right half

    return half1, half2


def get_body_image(
    in_shape,
    male_props,
    rotation,
    out_shape = (100, 100)):
    """Gets a binary, body-only image from a specified frame."""
    image = np.zeros(shape=in_shape, dtype=np.float)
    image[male_props.coords[:, 0], male_props.coords[:, 1]] = 1.

    image = center_image_in_frame(
            image,
            centroid=male_props.centroid,
            size=out_shape
        )

    image = sk_transform.rotate(image * 255, rotation)
    return image.astype(np.uint8)


def get_wing_image(image,
    female,
    arena,
    male_props,
    loose_threshold,
    shape=(100, 100),
    head='Right',
    subtract_body=True):
    """Gets a binary, wing-only image from a specified frame.

    Parameters
    ----------
    image : np.ndarray
        Frame within which to find wings.

    female : objects.Female object
        Female object so that we can mask out the female.

    arena : objects.CircularArena object
        Arena object to mask out areas outside of arena, and subtract
        the background image.

    male_props : skimage.measure.RegionProps object
        Properties that define the male image to look at. This should
        be calculated from find_male_tight().

    loose_threshold : int
        Low pass threshold to use to identify entirety of male body
        following background subtraction.

    shape : tuple of length 2, int (default = (100, 100))
        Shape of return image (height, width).

    head : string (default = 'Right')
        Which way returned wing image should be oriented.
        Options are:
        'Right' -- Image will contain a fly oriented with its head to the right.
        'Left' -- Image will contain a fly oriented with its head to the left.

    subtract_body : bool (default = True)
        Whether or not to subtract the body from the image.

    Returns
    -------
    wing_image : np.ndarray of type np.uint8 | shape = [shape]
        Binary image containing only wings.

    rotation : int
        Angle in degrees needed to rotate wing image so that it is aligned
        to either the right or left.

    """
    # this contains negative values, where the male fly of interest
    # is generally the most negative region in the background-subtracted image.
    bs_img = subtract_background(image, arena.background_image)

    # set all pixels below the negative threshold to 1,
    # and everything else to 0.
    bin_img = low_pass_threshold_binary(bs_img, -loose_threshold)

    # set the location of the female to 0, and the outside of the arena to 0.
    bin_img[np.where(female.get_female_mask())] = 0
    bin_img[np.where(arena.get_arena_mask() == 0)] = 0

    # set the body of the male to 0.
    if subtract_body:
        bin_img[male_props.coords[:, 0], male_props.coords[:, 1]] = 0
    # bin_img = sk_morph.binary_dilation(bin_img).astype(np.uint8)

    image = center_image_in_frame(
            bin_img,
            centroid = male_props.centroid,
            size=(200, 200)
        )

    image, rotation_angle = _orient_wing_image(
            image,
            male_props.orientation,
            head=head,
            return_rotation_degrees=True
        )

    image = trim_image2d(image, size=shape)
    return image, rotation_angle


def find_wings(image, female, arena, male_props, loose_threshold, logger,
    frame_ix, shape=(100, 100)):
    """Gets region props for left and right wing for a specified male.

    Parameters
    ----------
    image : np.ndarray
        Frame within which to find wings.

    female : objects.Female object
        Female object so that we can mask out the female.

    arena : objects.CircularArena object
        Arena object to mask out areas outside of arena, and subtract
        the background image.

    male_props : skimage.measure.RegionProps object
        Properties that define the male image to look at. This should
        be calculated from find_male_tight().

    loose_threshold : int
        Low pass threshold to use to identify entirety of male body
        following background subtraction.

    logger : Qt.QTextEdit widget
        Used to pass error messages to the tracking log.

    frame_ix : int
        Frame currently being tracked.

    shape : length 2 tuple (int, int) default = (100, 100)
        What shape to trim the wing image to.

    Returns
    -------
    male_fly_summary : dictionary

        Contains the following keys:

        left_wing_centroid -- np.ndarray of size 2
            Coordinate of left wing centroid (rr, cc).

        right_wing_centroid -- np.ndarray of size 2
            Coordinate of right wing centroid (rr, cc).

        left_wing_props -- skimage.measure.RegionProps object
            Region props describing left wing.

        right_wing_props -- skimage.measure.RegionProps object
            Region props describing right wing.

        head -- np.ndarray of size 2
            Coordinates of male head (rr, cc).

        rear -- np.darray of size 2
            Coordinates of male rear (rr, cc).

        rotation -- int
            Angle in degrees reqired to rotate male so that he is facing
            towards the right.

        loose_male_props -- skimage.measure.RegionProps object
            Region props describing male found with the loose threshold.
    """
    # get a binary wing image with the returned fly shape
    # oriented so that it is facing right.
    wing_image, rotation = get_wing_image(
            image=image,
            female=female,
            arena=arena,
            male_props=male_props,
            loose_threshold=loose_threshold,
            shape=shape,
            head='Right'
        )

    # split image into upper and lower regions
    upper, lower = _split_image(wing_image, axis='Horizontal')

    # make sure we can find wing properties in a given frame.
    try:
        left_wing = _get_props(upper)
    except NoPropsDetected as NPD:
        left_wing = NPD.props
        logger.append(
            '\t' + NPD.message + ' Left Wing @ frame {}'.format(frame_ix))

    try:
        right_wing = _get_props(lower)
    except NoPropsDetected as NPD:
        right_wing = NPD.props
        logger.append(
            '\t' + NPD.message + ' Right Wing @ frame {}'.format(frame_ix))

    # need to shift the row-coordinates of the right wing downward.
    right_wing_centroid = np.array(right_wing.centroid)
    right_wing_centroid[0] += wing_image.shape[0] / 2

    # find the head and rear positions of the male.
    head, rear = _get_male_body_axis(
            in_shape=image.shape,
            male_props=male_props,
            rotation=rotation,
            out_shape=shape,
            head='Right'
        )

    # rotate all coordinates back into their true locations.
    R = np.array([
            [np.cos(-rotation * np.pi/180), -np.sin(-rotation * np.pi/180)],
            [np.sin(-rotation * np.pi/180), np.cos(-rotation * np.pi/180)]
        ])

    body_coords = np.vstack((
            left_wing.centroid,
            right_wing_centroid,
            head,
            rear
        ))
    try:
        body_coords -= shape[0] / 2
    except TypeError:
        logger.append(
            '\t' + 'TypeError @ frame {}... Proceeding.'.format(frame_ix))
        return None

    rotated_coords = np.dot(R, body_coords.T).T
    rotated_coords += male_props.centroid

    # finally, get properties for male region with wings attached.
    full_male_img, _ = get_wing_image(
            image=image,
            female=female,
            arena=arena,
            male_props=male_props,
            loose_threshold=loose_threshold,
            shape=shape,
            head='Right',
            subtract_body=False
        )

    try:
        l_male_props = _get_props(full_male_img)
    except NoPropsDetected as NPD:
        l_male_props = NPD.props
        logger.append(
            '\t' + NPD.message + ' Loose Body Props @ frame {}'.format(frame_ix))

    return {
        'left_wing_centroid': rotated_coords[0, :],
        'right_wing_centroid': rotated_coords[1, :],
        'left_wing_props': left_wing,
        'right_wing_props': right_wing,
        'head': rotated_coords[2, :],
        'rear': rotated_coords[3, :],
        'rotation': rotation,
        'loose_male_props': l_male_props
    }


def set_male_props(male, props, i):
    """Sets all properties for a given male.

    .. warning:: This does not set the male centroid or the orientation.
                 These attributes should be set elsewhere.

    Parameters
    ----------
    male : Fly
        Male whose attributes to set.

    props : dictionary
        Properties returned from a call to find_wings().
    """
    # set left wing attributes
    male.left_wing.centroid.row[i], male.left_wing.centroid.col[i] = \
        props['left_wing_centroid']
    male.left_wing.major_axis_length[i] = \
        props['left_wing_props'].major_axis_length
    male.left_wing.minor_axis_length[i] = \
        props['left_wing_props'].minor_axis_length
    male.left_wing.orientation[i] = \
        props['left_wing_props'].orientation
    # male.left_wing.area[i] = props['left_wing_props'].area

    # set right wing attributes
    male.right_wing.centroid.row[i], male.right_wing.centroid.col[i] = \
        props['right_wing_centroid']
    male.right_wing.major_axis_length[i] = \
        props['right_wing_props'].major_axis_length
    male.right_wing.minor_axis_length[i] = \
        props['right_wing_props'].minor_axis_length
    male.right_wing.orientation[i] = \
        props['right_wing_props'].orientation
    # male.right_wing.area[i] = props['right_wing_props'].area

    # set body attributes
    male.body.head.row[i], male.body.head.col[i] = props['head']
    male.body.rear.row[i], male.body.rear.col[i] = props['rear']
    male.body.rotation_angle[i] = props['rotation']
    male.body.major_axis_length[i] = props['loose_male_props'].major_axis_length
    male.body.minor_axis_length[i] = props['loose_male_props'].minor_axis_length
    # male.body.area[i] = props['loose_male_props'].area


# if __name__ == '__main__':

# 	import motmot.FlyMovieFormat.FlyMovieFormat as FMF
# 	import matplotlib.pyplot as plt

# 	video_file = '/media/ben-shahar/DATA/test_data/videos/may_control_03.fmf'

# 	video = FMF.FlyMovie(video_file)
# 	arena = CircularArena(video)
# 	arena.calculate_background()
# 	arena.center = (arena.height / 2, arena.width / 2)
# 	arena.radius = int(arena.height / 2)
# 	img = arena.draw_arena()
# 	# plt.imshow(img); plt.show()

# 	mask = np.zeros_like(img)
# 	mask[arena.get_arena_coordinates()] = 255
# 	# plt.imshow(mask);plt.show()

# 	female = Female(arena)
# 	female.center = (244, 299)
# 	female.head = (233, 318)
# 	female.rear = (256, 281)
# 	female.maj_ax_rad = 28
# 	female.min_ax_rad = 14
# 	female.orientation = -31

# 	# mask = female.get_female_mask()
# 	# masked_image = arena.background_image.copy()
# 	# masked_image[np.where(mask)] = 0
# 	bs_img = subtract_background(video.get_frame(0)[0], arena.background_image)
# 	bs_bin = low_pass_threshold_binary(bs_img, -80)
# 	fig, ax = plt.subplots(nrows = 1, ncols = 3)
# 	ax[0].imshow(arena.background_image); ax[0].set_title('background')
# 	ax[1].imshow(bs_img); ax[2].imshow(bs_bin); plt.show()
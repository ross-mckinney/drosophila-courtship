
from PyQt5.QtGui import (
    QImage
)

def get_q_image(image):
    """Converts a numpy array to a QImage.

    Parameters
    ----------
    image : 2D or 3D np.ndarray
        Numpy array to convert to a QImage.

    Returns
    -------
    qimage : 2D or 3D QImage
        QImage object conatining the specified numpy array.
    """
    if len(image.shape) == 3:
        rgb = True
    elif len(image.shape) == 2:
        rgb = False

    height, width = image.shape[:2]

    if not rgb:
        try:
            return QImage(
                image.tostring(),
                width,
                height,
                QImage.Format_Indexed8
                )
        except:
            return QImage(
                image.data,
                width,
                height,
                QImage.Format_Indexed8
                )
    else:
        try:
            return QImage(
                image.data,
                width,
                height,
                QImage.Format_RGB888
                )
        except:
            return QImage(
                image.tostring(),
                width,
                height,
                QImage.Format_RGB888
                )


def get_mouse_coords(event):
    """Gets the position of the mouse following some event.

    Parameters
    ----------
    event : QEvent
        Event following which to find mouse position.

    Returns
    -------
    rr, cc : int, int
        Row-position of mouse, Col-position of mouse.
    """
    cc = event.pos().x()
    rr = event.pos().y()
    return rr, cc


def clear_list(list_widget):
    """Clears all items from a list widget.

    Parameters
    ----------
    list_widget : QListWidget
        List from which to remove all items.
    """
    while True:
        item = list_widget.takeItem(0)
        if not item:
            return
        del item

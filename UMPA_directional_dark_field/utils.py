import numpy as np


def gauss_from_transform2(v, sig_max=10):
    '''note sigma here referes to the traditional sigma'''
    array_to_fill = np.zeros_like(v)
    thetas = np.arctan2(v[:, :, 1], v[:, :, 0])
    epsilon = (v[:, :, 0] ** 2) + (v[:, :, 1] ** 2)
    sig_1s = np.sqrt(v[:, :, 2] ** 2 / (1 + epsilon))
    sig_2s = np.sqrt(v[:, :, 2] ** 2 / (1 - epsilon))

    for i in range(v.shape[0]):
        for j in range(v.shape[1]):

            array_to_fill[i, j, 1] = sig_2s[i, j]
            array_to_fill[i, j, 0] = sig_1s[i, j]
            array_to_fill[i, j, 2] = thetas[i, j]
    return array_to_fill


def gauss_from_transform(v, sig_max=10):
    '''note sigma here referes to the traditional sigma'''
    array_to_fill = np.zeros_like(v)
    thetas = ((np.arctan2(v[:, :, 1], v[:, :, 0]) * 180 / np.pi) +270) % 180
    epsilon = (v[:, :, 0] ** 2) + (v[:, :, 1] ** 2)
    sig_1s = np.sqrt(v[:, :, 2] ** 2 / (1 + epsilon))
    sig_2s = np.sqrt(abs(v[:, :, 2] ** 2 / (1 - epsilon))) # abs incase epsilon ? 1 - illegal but possible

    sig_1s = np.clip(sig_1s, 0, sig_max)
    sig_2s = np.clip(sig_2s, 0, sig_max)

    for i in range(v.shape[0]):
        for j in range(v.shape[1]):

            if sig_1s[i, j] < sig_2s[i, j]:
                array_to_fill[i, j, 0] = sig_2s[i, j]
                array_to_fill[i, j, 1] = sig_1s[i, j]
                if thetas[i, j] > 90:
                    array_to_fill[i, j, 2] = thetas[i, j] - 90
                else:
                    array_to_fill[i, j, 2] = thetas[i, j] + 90

            else:
                array_to_fill[i, j, 0] = sig_1s[i, j]
                array_to_fill[i, j, 1] = sig_2s[i, j]
                array_to_fill[i, j, 2] = thetas[i, j]

    return array_to_fill

def abc_from_transform_c_notc(sx, sy, sig, t = 0.45, max_sig=10):
    ''' Note: set t to 0.45 for ddf paper data'''
    #theta, sig_x_sq, sig_y_sq = 0
    add_cost = 0

    # get gauss coords

    eps = (sx*sx) + (sy*sy)
    theta = np.arctan2(sy, sx)

    if eps >= 1:
        add_cost = (eps - 1)
        if eps >=1:
            eps = eps/(eps + 0.01)

    if sig >max_sig:
        add_cost = add_cost + (sig - 10) * 10
    elif sig < 0:
        add_cost = add_cost + ((sig)* -10)
        sig = 0


    # squaring now to avoid squaring twice, rewriting over variables because I can
    t = t * t
    sig = sig * sig

    sig_x_sq = (sig/ (1 + eps)) + t
    sig_y_sq = (sig/ (1 - eps)) + t

    sinv_sq = np.sin(theta)**2
    cosv_sq = np.cos(theta)**2
    sin2v = np.sin(2*theta)

    #a,b,c = 0

    a = 0.5 * ((cosv_sq/sig_x_sq) + (sinv_sq/sig_y_sq))
    b = 0.5 * ((sin2v/sig_x_sq) - (sin2v/sig_y_sq))
    c = 0.5 * ((sinv_sq/sig_x_sq) + (cosv_sq/sig_y_sq))

    return a,b,c,add_cost

def abc_from_transform(v, t=0.45):


    if np.asarray(v).ndim == 3:

        out = np.zeros_like(v)
        gp = gauss_from_transform2(v)
    else:

        out = np.zeros((1,1,3))
        gp = gauss_from_transform2(np.array(v).reshape(1,1,3))
    sig_x = np.sqrt(gp[:,:,0]**2 + t**2)
    sig_y = np.sqrt(gp[:,:,1]**2 + t**2)
    theta = gp[:,:,2]

    out[:,:,0] = (np.cos(theta) ** 2 / (2 * (sig_x ** 2))) + (np.sin(theta) ** 2 / (2 * (sig_y ** 2)))
    out[:,:,1] = 2*((np.sin(2 * theta) / (4 * (sig_x ** 2))) + (-np.sin(2 * theta) / (4 * (sig_y ** 2))))
    out[:,:,2] = (np.sin(theta) ** 2 / (2 * (sig_x ** 2))) + (np.cos(theta) ** 2 / (2 * (sig_y ** 2)))

    return np.squeeze(out)


def generate_rgb(gauss_properties, log_scale=False):
    direction = gauss_properties[:, :, 2]
    #
    # if log_scale:
    #     gauss_properties[:,:,1] = np.log(1 + gauss_properties[:,:,1])
    #     gauss_properties[:, :, 2] = np.log(1 + gauss_properties[:, :, 2])

    magnitude = np.sqrt(gauss_properties[:,:,0]**2 + gauss_properties[:,:,1]**2)

    directionality = 1- ((gauss_properties[:,:,1]**2) / (gauss_properties[:,:,0]**2))
    directionality = np.nan_to_num(directionality)

    hsv = np.empty((3, gauss_properties.shape[0], gauss_properties.shape[1]))

    hsv[0, :, :] = direction / 180.
    hsv[1, :, :] = directionality / np.max(directionality)
    hsv[2, :, :] = magnitude / np.max(magnitude)

    if log_scale:
        hsv[2,:,:] = np.log(hsv[2,:,:] + 1)
        hsv[2, :, :] = hsv[2, :, :]/np.max(hsv[2, :, :])

    rgb = hsv2rgb(hsv)

    return rgb.astype(int), direction, directionality, magnitude


def hsv2rgb(hsv):
    """\
    HSV (Hue,Saturation,Value) to RGB (Red,Green,Blue) transformation.
    Parameters
    ----------
    hsv : array-like
        Input must be two-dimensional. **First** axis is interpreted
        as hue,saturation,value channels.
    Returns
    -------
    rgb : ndarray
        Three dimensional output. **Last** axis is interpreted as
        red, green, blue channels.
    See also
    --------
    complex2rgb
    complex2hsv
    rgb2hsv

    BORROWED FROM PTYPY PACKAGE AND MODIFIED
    """
    # HSV channels
    h, s, v = hsv

    i = (6. * h).astype(int)
    f = (6. * h) - i
    p = v * (1. - s)
    q = v * (1. - s * f)
    t = v * (1. - s * (1. - f))
    i0 = (i % 6 == 0)
    i1 = (i == 1)
    i2 = (i == 2)
    i3 = (i == 3)
    i4 = (i == 4)
    i5 = (i == 5)

    rgb = np.zeros(h.shape + (3,), dtype=h.dtype)
    rgb[:, :, 0] = 255 * (i0 * v + i1 * q + i2 * p + i3 * p + i4 * t + i5 * v)
    rgb[:, :, 1] = 255 * (i0 * t + i1 * v + i2 * v + i3 * q + i4 * p + i5 * p)
    rgb[:, :, 2] = 255 * (i0 * p + i1 * p + i2 * t + i3 * v + i4 * v + i5 * q)

    return rgb

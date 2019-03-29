import cv2
import numpy as np
import pdb
def coord_trans(point3d, rvec, tvec):
    rvec = np.asarray(rvec)
    pdb.set_trace()
    tvec = np.asarray(tvec).reshape((3, 1))
    point3d = np.asarray(point3d).reshape((3, 1))
    R = rvec
    if np.prod(R.shape) == 3:
        R, _ = cv2.Rodrigues(rvec)
#     new_coord = np.dot(R, point3d) + tvec
    new_coord = np.dot(R.T, point3d - tvec)
    return new_coord

def euler2rotation(euler_angles):
    """Convert Euler angles to rotation matrix.

    Args:
      euler_angles: {list, numpy.ndarray}
          Euler rotation angles in radians, specified as a 1x3 list
          or numpy.array of [yaw, pitch, roll]. The default order for
          Euler angle rotations is "ZYX".
    Returns:
      rotation_matrix: numpy.ndarray
          rotation matrix, return as a 3x3 numpy.ndarray.
    """
    yaw, pitch, roll = euler_angles[0], euler_angles[1], euler_angles[2]
    rotation_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    rotation_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
                           [-np.sin(pitch), 0,
                            np.cos(pitch)]])
    rotation_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)],
                           [0, np.sin(roll), np.cos(roll)]])
    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
    return rotation_matrix
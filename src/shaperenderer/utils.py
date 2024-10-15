import math
from typing import Optional

import numpy as np
from numpy.linalg import norm


def translation(tx: float, ty: float, tz: float) -> np.array:
    """Translation matrix"""
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])


def scaling(sx: float, sy: float, sz: float) -> np.array:
    """Scaling matrix"""
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])


def xrotation(angle_in_radians: float) -> np.array:
    """Rotation matrix around x-axis"""
    c = math.cos(angle_in_radians)
    s = math.sin(angle_in_radians)
    
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])


def yrotation(angle_in_radians: float) -> np.array:
    """Rotation matrix around y-axis"""
    c = math.cos(angle_in_radians)
    s = math.sin(angle_in_radians)
    
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])


def zrotation(angle_in_radians: float) -> np.array:
    """Rotation matrix around z-axis"""
    c = math.cos(angle_in_radians)
    s = math.sin(angle_in_radians)
    
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def translate(m: np.array, tx: float, ty: float, tz: float) -> np.array:
    """
    """
    return np.dot(translation(tx, ty, tz), m)


def scale(m: np.array, sx: float, sy: float, sz: float) -> np.array:
    """
    """
    return np.dot(scaling(sx, sy, sz), m)


def xrotate(m: np.array, angle_in_radians: float) -> np.array:
    """
    """
    return np.dot(xrotation(angle_in_radians), m)


def yrotate(m: np.array, angle_in_radians: float) -> np.array:
    """
    """
    return np.dot(yrotation(angle_in_radians), m)


def zrotate(m: np.array, angle_in_radians: float) -> np.array:
    """
    """
    return np.dot(zrotation(angle_in_radians), m)


def zreflect(m: np.array) -> np.array:
    """
    """
    R = np.array([
        [1, 0,  0],
        [0, 1,  0],
        [0, 0, -1],
    ])
    return np.dot(R, m)


def homogenize(vec3: np.array) -> np.array:
    """
    """
    try:
        # set of vertex vectors (mesh)
        d1, d2 = vec3.shape
        assert d1 == 3, "Error: vertex must have three coordinates!"

        stack = np.vstack

    except ValueError: # not enough values to unpack (expected 2, got 1)
        # single vertex vector
        d2 = 1
        stack = np.concatenate

    return stack((
        vec3,
        np.ones(d2, dtype=vec3.dtype)
    ))


def lookat(
    eye: np.array,
    at: np.array = np.array([0., 0., 0.]),
    up: np.array = np.array([0., 1., 0.])
) -> np.array:
    """
    https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml

    Set up a viewing matrix derived from the eye/camera position, a reference point and an UP vector in OpenGL way.
    The matrix maps the reference point to the negative z-axis and the eye point to the origin.
    It transforms vertices from the model (local) space to the camera space.

    Parameters
    ----------
    eye : np.ndarray of shape (3,)
        Specifies the position of the eye/camera point

    at : np.ndarray of shape (3,), default=np.array([0., 0., 0.])
        Specifies the position of the reference point, i.e. the point we want the camera to point at

    up : np.ndarray of shape (3,), default=np.array([0., 1., 0.])
        Specifies the direction of the up vector of the camera assuming the camera is straight up to the positive y-axis.
        The UP vector must not be parallel to the line of sight from the eye point to the reference point.

    Returns
    -------
    m : np.ndarray of shape (4, 4)
        Viewing matrix to perform the inverse transformation so that the camera is at the origin and
        is looking at the reference point along the negative z-axis

    References
    ----------
    [1] Breakdown of the LookAt function in OpenGL
        https://www.geertarien.com/blog/2017/07/30/breakdown-of-the-lookAt-function-in-OpenGL/

    [2] Placing a Camera: the LookAt Function
        https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function

    [3] OpenGL Camera
        http://www.songho.ca/opengl/gl_camera.html

    [4] Understanding the View Matrix
        https://www.3dgep.com/understanding-the-view-matrix/#Look_At_Camera
    """
    m = np.zeros((4, 4))
    # compute the forward vector from target to eye
    f = eye - at
    f = f / norm(f)
    
    # set the world up vector dependig on direction of the forward vector
    # calculate inclination angle (theta) to find the direction of the forward vector
    theta = math.acos(f[1] / norm(f))
    theta = math.degrees(theta)

    if 180 - 1e-3 <= theta <= 180 + 1e-3: # forward vector points down
        up = np.array([0., 0., 1.])
    elif 0 - 1e-3 <= theta <= 0 + 1e-3: # forward vector points up
        up = np.array([0., 0., -1.])
    
    # compute the right vector
    r = np.cross(up, f)
    r = r / norm(r)
    
    # recompute the orthonormal up vector
    u = np.cross(f, r)
    
    # populate the viewing matrix
    m[0, :-1] = r
    m[1, :-1] = u
    m[2, :-1] = f
    
    m[0, -1] = -np.dot(r, eye)
    m[1, -1] = -np.dot(u, eye)
    m[2, -1] = -np.dot(f, eye)
    m[3, -1] = 1
    
    return m


def perspective(
    fovy: float = math.radians(90),
    aspect: float = 1.,
    znear: float = 0.1,
    zfar:float = 100.
) -> np.array:
    """
    https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml

    Set up a perspective projection matrix in OpenGL style.
    The matrix generates the viewing frustum/volume of the camera defined by the camera's field of view,
    the near and far clipping planes and the image aspect ratio.
    It takes the eye coordinates from the vertex data and transforms them to the clip space. Then clip coordinates
    are transformed to the normalized device coordinates (NDC).

    The goal of a projection matrix is to remap the values projected onto the image plane to a unit cube,
    whose minimum and maximum extents are (-1,-1,-1) and (1,1,1) respectively.

    Parameters
    ----------
        fovy: float, default=90 (degress in radians)
            The vertical (in y direction) field of view angle, in radians.
            
        aspect: float, default=1.
            Aspect ratio of the display window that determines the field of view in the x direction.
            The aspect ratio is the ratio of x (width) to y (height).
            
        znear: float, default=0.1
            The distance from the viewer to the near (front) clipping plane of the viewing frustum (always positive).
            
        zfar: float, default=100.
            The distance from the viewer to the far (back) clipping plane of the viewing frustum (always positive).
    
    Returns
    -------
    m : np.ndarray of shape (4, 4)
        4x4 perspective projection matrix generated by the viewing frustum to transform the eye coordinates
        from the vertex data to the NDC coordinates.
    
    Notes
    -----
    Depth buffer precision is affected by the values of near and far planes. The greater the ratio of far to near
    is, the less effective the depth buffer will be at distinguishing between surfaces that are near each other.
    Because the ratio approaches infinity as near approaches 0, near must never be set to 0.

    References
    ----------
    [1] The Perspective and Orthographic Projection Matrix. The OpenGL Perspective Projection Matrix
        https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
    
    [2] OpenGL Projection Matrix
        http://www.songho.ca/opengl/gl_projectionmatrix.html

    [3] WebGL 3D Perspective
        https://webglfundamentals.org/webgl/lessons/webgl-3d-perspective.html
    """
    
    f = math.tan(0.5 * fovy)
    t = f * znear
    b = -t
    r = t * aspect
    l = -r
    
    return np.array([
        [2 * znear / (r - l), 0, (r + l) / (r - l), 0],
        [0, 2 * znear / (t - b), (t + b) / (t - b), 0],
        [0, 0, -(zfar + znear) / (zfar - znear), -2 * zfar * znear / (zfar - znear)],
        [0, 0, -1, 0]
    ])


def sample_on_sphere(
        low: float = 0.,
        high: float = 1.,
        size: tuple[int, int] = (1, 1),
        rng: Optional[np.random.Generator] = None,
) -> tuple[np.array, np.array]:
    """
    """
    if rng:
        sample = rng.uniform
    else:
        sample = np.random.random
    v = sample(low=low, high=high, size=size[0])
    u = sample(low=low, high=high, size=size[1])

    thetas = np.degrees(math.pi/2 - np.arccos(2*v - 1)) # substitute polar angle with elevation angle
    phies = np.degrees(2*math.pi*u)

    return thetas, phies


def sample_interval_union(
        space: list[tuple[float, float]],
        size: int = 1,
        rng: Optional[np.random.Generator] = None
):
    def foo(random_number: float) -> float:
        for start, end in space:
            if random_number <= (end - start):
                return start + random_number

            random_number = random_number - (end - start)

        return end

    vmin = 0
    vmax = sum(end - start for start, end in space)

    if rng:
        sample = rng.uniform
    else:
        sample = np.random.uniform
    
    values = sample(low=vmin, high=vmax, size=size)
    values = list(map(foo, values))

    return np.array(values)
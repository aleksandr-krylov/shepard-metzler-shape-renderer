import logging
import math
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib.patches import Patch, Polygon
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage

from .geometry import MetzlerShape
from . import utils


# set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)


class Camera:
    """
    Camera class represents the virtual camera used to view the scene.
    The camera itself is not rendered, but its transform affects the apparent placement of the objects
    in the rendered image of the scene.

    Parameters
    ----------
    fovy: float, default=90 (degrees in radians)
            The vertical (in y direction) field of view angle, in radians.

    znear: float, default=0.1
        The distance from the viewer to the near (front) clipping plane of the viewing frustum (always positive).

    zfar: float, default=100.
        The distance from the viewer to the far (back) clipping plane of the viewing frustum (always positive).
    """

    def __init__(
        self,
        fovy: float = math.radians(90),
        znear: float = 0.1,
        zfar: float = 100.
    ) -> None:
        # by default, camera is placed at the origin
        self.position = np.array([0., 0., 0.])
        self.fovy = fovy
        self.znear = znear
        self.zfar = zfar


    def setPosition(
        self,
        xangle: float,
        yangle: float,
        zangle: float,
        tx: float,
        ty: float,
        tz: float
    ) -> None:
        """
        Set the camera position in the world space.

        Parameters
        ----------
        xangle : float
            Rotation angle around x-axis

        yangle : float
            Rotation angle around y-axis

        zangle : float
            Rotation angle around z-axis

        tx : float
            How much to move along x-axis

        ty : float
            How much to move along y-axis

        tz : float
            How much to move along z-axis


        Returns
        -------
        Updated camera position
        """
        transform = np.identity(4)
        transform = utils.translate(transform, tx, ty, tz)
        transform = utils.zrotate(transform, math.radians(zangle))
        transform = utils.yrotate(transform, math.radians(yangle))
        transform = utils.xrotate(transform, math.radians(xangle))
        # the final position of the camera can be extracted from
        # the entries of the last column of the transform matrix
        self.position = transform[:-1, -1]


    def setSphericalPosition(self, r: float, theta: float, phi: float) -> None:
        """
        Set the camera position in the world space usinig spherical coordinates.

        Parameters
        ----------
        r : float
            Radius of the sphere

        theta : float
            Elevation angle (in degrees)

        phi : float
            Azimuth angle (in degrees)

        Returns
        -------
        Updated camera position
        """
        elevation = math.radians(theta)
        azimuth = math.radians(phi)
        radius = np.array([0., 0., r])

        rotation = utils.yrotation(azimuth) @ utils.xrotation(elevation)
        position = rotation @ utils.homogenize(radius)

        self.position = position[:-1]


    def setLookAtMatrix(
        self,
        at: np.array = np.array([0., 0., 0.]),
        up: np.array = np.array([0., 1., 0.])
    ) -> np.array:
        """
        Set up LookAt matrix defining the transformation required to view the object

        Parameters
        ----------
        at : np.ndarray of shape (3,), default=np.array([0., 0., 0.])
            Specifies the position of the reference point, i.e. the point we want the camera to point at

        up : np.ndarray of shape (3,), default=np.array([0., 1., 0.])
            Specifies the direction of the up vector of the camera assuming the camera is straight up to the positive y-axis.
            The UP vector must not be parallel to the line of sight from the eye point to the reference point.

        Returns
        -------
        LookAt matrix : np.ndarray of shape (4, 4)
        """
        return utils.lookat(eye=self.position, at=at, up=up)


    def setProjectionMatrix(self, aspect: float = 1.) -> np.array:
        """
        Set up perspective projection matrix

        Parameters
        ----------
        aspect: float, default=1.
            Aspect ratio of the display window that determines the field of view in the x direction.


        Returns
        -------
        Projection matrix : np.ndarray of shape (4, 4)
        """
        return utils.perspective(self.fovy, aspect, self.znear, self.zfar)


class Object3D:
    """
    """
    def __init__(
        self,
        shape: MetzlerShape,
        facecolor: str | None, # color (e.g. "white", "blue", etc.) / "random-by-face" / "random-by-cube"
        edgecolor: str = "black",
        edgewidth: float = 1.5
    ) -> None:
        self.vertices = shape.vertices
        self.edges = shape.edges
        self.faces = shape.faces
        self.com = shape.com
        self.attributes = {
            "edgecolor": edgecolor,
            "edgewidth": edgewidth,
        }

        if facecolor == "random-by-face": # each face will have a different color
            # generate random RGB colors
            colors = np.random.randint(0, 256, size=(self.n_faces, 3))
            # normalize colors into the [0.0, 1.0] interval
            colors = [
                mpl.colors.Normalize(vmin=0, vmax=255)(color).data.tolist()
                for color in colors
            ]
        elif facecolor == "random-by-cube": # each cube will have a different color
            # generate random RGB colors
            colors = np.random.randint(0, 256, size=(10, 3))
            # normalize colors into the [0.0, 1.0] interval
            colors = list(map(
                lambda color: mpl.colors.Normalize(vmin=0, vmax=255)(color).data.tolist(),
                colors
            ))
            colors = np.repeat(colors, 6, axis=0).tolist()

        else: # each face has same color (in word or hex)
            colors = [
                facecolor
                for _ in range(self.n_faces)
            ]

        self.attributes["facecolor"] = colors


    @property
    def n_vertices(self) -> int:
        return len(self.vertices)


    @property
    def n_edges(self) -> int:
        return len(self.edges)


    @property
    def n_faces(self) -> int:
        return len(self.faces)


    def setModelMatrix(self) -> np.array:
        """
        Set up model matrix defining transformation from the model space to the world space

        By default, the object is placed at the origin of world's coordinate system, i.e.
        object's origin coincides with the world's.
        """
        return np.identity(4)


class VertexShader:
    """
    """
    def __init__(self) -> None:
        self.zbuffer = []  # depth buffer


    def transform(self, mvp: np.array, object: Object3D) -> list[np.array]:
        """
        """
        # project vertex 3D positions from the camera space to the 2D coordinates in the screen space
        # now vertices are defined in the clip space
        projected = mvp @ utils.homogenize(object.vertices)
        assert projected.shape == (
            4, object.vertices.shape[1]), "Error: projection is done incorrectly!"

        # keep the depth information in Z-buffer for visibility determination in later rendering
        self.zbuffer = [projected[-1, face] for face in object.faces]
      
        # normalize the coordinates by doing perspective divide
        normalized = [
            projected[:, face] / self.zbuffer[idx] for idx, face in enumerate(object.faces)
        ]  # now vertices are defined in NDC space [-1, 1]

        # screen coordinates X, Y
        return [face_vertices[:2, :] for face_vertices in normalized]


class GeometryShader:
    """
    """
    def __init__(self, geometry: Patch = Polygon) -> None:
        self.primitive = geometry


    def generate(
        self,
        vertices: list[np.array],
        zbuffer: list[np.array],
        properties: dict[str, str]
    ) -> list[Patch]:
        """
        """
        collection = []  # a list of generated primitives given the vertex data
        for cnt, face in enumerate(vertices):
            collection.append(
                self.primitive(
                    xy=face.T,
                    zorder=-np.mean(zbuffer[cnt]),
                    fill=True if properties.get("facecolor")[0] else False,
                    facecolor=properties.get("facecolor")[cnt],
                    edgecolor=properties.get("edgecolor"),
                    linewidth=properties.get("edgewidth"),
                )
            )

        return collection


class Renderer:
    """
    """
    def __init__(
        self,
        imgsize: tuple[int, int] = (128, 128),
        dpi: int = 100,
        bgcolor: str = "white",
        format: str = "png"
    ) -> None:
        self.vbuffer = []  # stores vertex data

        # configure matplotlib properties and styles
        mpl.rcParams["figure.figsize"] = (
            imgsize[0] / dpi, imgsize[1] / dpi) 	# figure size in inches
        # figure dots per inch
        mpl.rcParams["figure.dpi"] = dpi
        # when True, automatically adjust subplot
        mpl.rcParams["figure.autolayout"] = True
        # parameters to make the plot fit the figure
        # using `tight_layout`
        mpl.rcParams["figure.facecolor"] = bgcolor
        mpl.rcParams["figure.edgecolor"] = bgcolor

        mpl.rcParams["axes.facecolor"] = bgcolor
        mpl.rcParams["axes.edgecolor"] = bgcolor

        # figure dots per inch or 'figure'
        mpl.rcParams["savefig.dpi"] = dpi
        # figure format {png, ps, pdf, svg} when saving
        mpl.rcParams["savefig.format"] = format
        # figure face color when saving
        mpl.rcParams["savefig.facecolor"] = bgcolor
        # figure edge color when saving
        mpl.rcParams["savefig.edgecolor"] = bgcolor

        plt.close("all")  # close all the figures created by matplotlib before
        # create a figure to render the object in it
        self.figure = plt.figure(
            tight_layout={
                "pad": 0.0, "w_pad": None,
                "h_pad": None, "rect": None
            }
        )


    def render(self, object: Object3D, camera: Camera) -> None:
        """
        """
        def draw_primitives(pcollection: list[Patch]) -> None:
            """
            Helper function for rendering an object from a collection of object's primitives (e.g. face).
            """
            self.figure.clf()  # clear the current figure
            plot = self.figure.add_subplot(aspect='equal')

            # configure axes
            plot.set_xlim(-1/2, 1/2)
            plot.set_ylim(-1/2, 1/2)
            plot.axis('off')

            # place primitives from the collection onto the plot
            for patch in pcollection:
                plot.add_patch(patch)

        self.vbuffer = []  # empy the vertex buffer every time we want to render the object

        # model matrix to move the object from its local space to the world space,
        # i.e. now all object's vertices will be defined relative to the center of the world
        model = object.setModelMatrix()

        # view matrix to go from the world space to the camera space,
        # i.e. now the object's coordinates will be defined relative to the camera
        view = camera.setLookAtMatrix()

        # projection matrix to map the coordinates to the screen space
        projection = camera.setProjectionMatrix()
        # joint matrix
        mvp = projection @ view @ model

        vshader = VertexShader()
        # 2D coordinates on flat screen
        self.vbuffer += vshader.transform(mvp, object)

        gshader = GeometryShader(geometry=Polygon)
        # generate collection of primitives to render the object
        pcollection = gshader.generate(self.vbuffer, vshader.zbuffer, object.attributes)

        # draw generated primitives to make the object
        draw_primitives(pcollection)


    def save_figure_to_file(
        self,
        fname: Path | str = "figure",
        verbose: bool = False
    ) -> None:
        """
        Save the rendered figure in a file.

        Parameters
        ----------
        fname : str, default="figure"
            File name with the relative path to location where the figure needs to be saved
        """
        save_path = fname.absolute() if isinstance(fname, Path) \
                    else Path(fname).absolute()

        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.figure.savefig(
            save_path,
            transparent=True if mpl.rcParams["figure.facecolor"] == "none" \
            else False
        )
        
        if verbose: logger.info(
            "Image saved to {filepath}.{extension}".format(filepath=save_path, extension=mpl.rcParams["savefig.format"])
        )


    def save_figure_to_numpy(
            self,
            color_channel_first: bool = False
    ) -> np.array:
        """
        Save the rendered figure in a numpy array
        using mplfig_to_npimage function from moviepy.

        source: https://github.com/Zulko/moviepy/blob/bc8d1a831d2d1f61abfdf1779e8df95d523947a5/moviepy/video/io/bindings.py#L8
        """

        figure_numpy = mplfig_to_npimage(self.figure) # converts a matplotlib figure to an RGB frame after updating the canvas

        if color_channel_first:
            return figure_numpy.transpose(2, 0, 1)
        
        return figure_numpy


from enum import Enum
import itertools as it
from typing import Optional
import math

import numpy as np

from .utils import yrotation, translate, homogenize


class Axis(Enum):
    """
    # TODO : ADD DOCSTRING
    """
    X = 0
    Y = 1
    Z = 2


class Direction(Enum):
    """
    # TODO : ADD DOCSTRING
    """
    L = 0 # left
    D = 1 # down
    B = 2 # backward
    R = 3 # right
    U = 4 # up
    F = 5 # forward


    @classmethod
    def opposite(cls, of: str):
        """
        Returns a direction opposite to the direction given in the argument <of>
        """
        opp_value = cls[of].value + 3
        opp_value = opp_value if opp_value < 6 else opp_value - 6

        return cls(opp_value)
    
    @property
    def axis(self) -> Axis:
        """
        0 - X axis, 1 - Y axis, 2 - Z axis
        """
        return Axis(self.value % 3)
    

    @property
    def orientation(self) -> int:
        """
        Can be positive (+1) / negative (-1).
        """
        return 2 * (self.value // 3) - 1


class Plane(Enum):
    """
    # TODO : ADD DOCSTRING
    """
    YZ = 0 # plane X = 0
    XZ = 1 # plane Y = 0
    XY = 2 # plane Z = 0


    @classmethod
    def sample(cls, rng: Optional[np.random.Generator] = None):
        """
        # TODO : ADD DOCSTRING
        """
        val = rng.integers(0, len(cls), size=1) if rng \
              else np.random.randint(0, len(cls), size=1)
        
        return cls(val)
    

class Quadrant(Enum):
    """
    """
    I = 0
    II = 1
    III = 2
    IV = 3

    @property
    def angle(self):
        return -90 * self.value


class ShapeString:
    """
    """
    def __init__(self, path: str) -> None:
        self.shape = path.upper()


    def __repr__(self) -> str:
        return self.shape.lower()


    def __iter__(self):
        return (d for d in self.shape)
    

    def encode(self) -> list[int]:
        """
        # TODO : ADD DOCSTRING
        """
        return list(map(lambda d: Direction[d].value, self.shape))
    

    def to_vector(self) -> np.ndarray:
        """
        # TODO : ADD DOCSTRING
        """
        v = np.zeros((9, 4, 1))

        encoded = list(
            map(
                lambda d: Direction[d],
                self.shape
            )
        )
        axes = list(map(lambda d: d.axis.value, encoded))
        orientations = list(map(lambda d: d.orientation, encoded))

        for cube, (ax, orient) in enumerate(zip(axes, orientations)): v[cube, ax] = orient * 1

        v[:, -1, :] = 1.

        return v
    

    def change_quadrant(self, quadrant: Quadrant):
        """
        # TODO : ADD DOCSTRING
        """
        v = self.to_vector()
        v_rotated = np.round(yrotation(math.radians(quadrant.angle)) @ v)

        indices_transformed = list(map(lambda v: np.nonzero(v[:3, -1])[0].item(), v_rotated))
        signs_transformed = list(map(lambda tpl: int(tpl[0][tpl[1]].item() > 0), zip(v_rotated, indices_transformed)))

        shape_encoded_in_new_quadrant = list(map(lambda tpl: tpl[0] + 3 if tpl[1] > 0 else tpl[0], zip(indices_transformed, signs_transformed)))

        shape_new = "".join(list(map(lambda d: Direction(d).name, shape_encoded_in_new_quadrant)))
        
        return ShapeString(shape_new)
    


    def reverse(self):
        """
        # TODO : ADD DOCSTRING
        """
        return ShapeString(
            "".join(
                map(
                    lambda d: Direction.opposite(of=d).name,
                    self.shape[::-1]
                )
            )
        )
    

    def reflect(self, over: Plane):
        """
        # TODO : ADD DOCSTRING
        """
        # step 1: get an axis of the reflection
        # step 2: get shape indices that need to be reflected
        # step 3: create a reflected shape

        axis_of_reflection = Axis(over.value) # axis of reflection: 0 - X axis, 1 - Y axis, 2 - Z axis
    
        mask = list(
            map(
                lambda d: Direction[d].axis == axis_of_reflection,
                self.shape
            )
        )
        
        return ShapeString(
            "".join([
                Direction.opposite(of=d).name if m else d
                for m, d in zip(mask, self.shape)
            ])
        )


    def count_orientations(self) -> int:
        """
        # TODO : ADD DOCSTRING
        """
        shape_orientations = list(
            map(
                lambda d: Direction[d].axis.value,
                self.shape
            )
        )
        return len(np.unique(shape_orientations))
    

    @property
    def axis_of_elongation(self) -> Axis:
        """
        # TODO : ADD DOCSTRING
        """
        breakdown = sorted([(k, len(list(g))) for k, g in it.groupby(str(self))], key=lambda item: item[1])

        if breakdown[-1][-1] < 5:
            return None
        
        if breakdown[-1][0] in ("u", "d"):
            return "Y"
        
        if breakdown[-1][0] in ("r", "l"):
            return "X"
        
        if breakdown[-1][0] in ("b", "f"):
            return "Z"
    

class ShapeGenerator:
    """
    Generator of shape's path describing the sequence of directions one needs to take to produce the shape.

    Parameters
    ----------
    probability : {'uniform', 'random'}, default='uniform'
        Class of the probability distribution for all the possible directions to walk.
        By default, all the directions are equally possible.

    random_state : int, default=None
        Controls the randomness of drawing the next direction in the walk. Initializes the new instance of
        default_rng() random generator. Same as random seed.
    """
    def __init__(
        self,
        random_state: int | None,
        probability: str = "uniform",
    ) -> None:
        # instantiate Random Generator for reproducibility sake
        self.rng = np.random.default_rng(seed=random_state)

        if probability == 'uniform':
            self.probabilities = np.repeat(
                1 / len(Direction),
                len(Direction)
            )
        elif probability == 'random':
            self.probabilities = self.rng.integers(
                100,
                size=len(Direction)
            )
            self.probabilities /= np.sum(self.probabilities)
            assert np.sum(self.probabilities) == 1., \
                "Error: probabilities don't sum up to 1"
        else:
            raise ValueError(
                f"{probability} is not defined! \
                Please, put one of the accepted arguments for probability: \
                ('uniform', 'random')"
            )


    def update_probabilities(
        self,
        shape: str,
        overlap_likely: bool = False,
        loop_likely: bool = False
    ) -> None:
        """
        Update the probability distribution for all possible directions given the last d by
        distributing the probability mass released by prohibited directions
        among the ones still available for the next d.

        Parameters
        ----------
        shape : str
            Shape's route at this moment.
            The route is represented by a sequence of direction codes in the walking order. 

        overlap_likely: bool, default=False
            # TODO : ADD COMMENT

        loop_likely : bool, default=False
            Indicates the possibility for the closed loop to occure in the next d.


        Returns
        -------
        self : ShapeGenerator
            Updated probability distribution vector.
        """
        # get the index of the last direction
        last_idx = Direction[shape[-1]].value
        # calculate index of the opposite direction of the last one
        last_opp_idx = Direction.opposite(of=shape[-1]).value

        to_be_masked = [last_idx, last_opp_idx]  # canceled directions

        if overlap_likely:
            # leave out the direction which leads to having both arms overlap by the next d
            # get the previous direction we walked before taking turn
            overlap_idx = Direction.opposite(of=shape[-2]).value

            to_be_masked += [overlap_idx]

        if loop_likely:
            # need to update probabilities one more time
            # to rule out the loop to pop up after the next d

            # count number of upward ds
            ups = shape.count("U") + 1
            # count number of backward ds
            backwards = shape.count("B") + 1
            # count number of downward ds
            downs = shape.count("D") + 1
            # count number of the rest of ds left
            forwards = 10 - (ups + downs + backwards - 3)

            if ((downs < ups) and (forwards >= backwards)) or ((downs == ups) and (forwards > backwards)):
                to_be_masked += [Direction["F"].value]

        # create a mask to keep track of prohibited and accessible directions to walk
        mask = np.ones_like(self.probabilities, dtype=bool)
        mask[np.unique(to_be_masked)] = False

        # set probs of two/three prohibited directions to zero
        self.probabilities[~mask] = 0.
        # recalculate the probability mass
        self.probabilities[mask] += (1 - np.sum(self.probabilities[mask])) / np.sum(mask)
        assert np.sum(
            self.probabilities) == 1., "Error: probabilities don't sum up to 1"


    def reset_probabilities(self) -> None:
        """
        Set probabilities to default values, i.e. the ones defined at generator's init time

        Returns
        -------
        self : ShapeGenerator
            Default probability distribution over directions.
        """
        self.probabilities = np.repeat(
            1 / len(Direction),
            len(Direction)
        )


    def draw_direction(self) -> str:
        """
        Draw the next direction to walk with respect to the probability distribution over all possible directions

        Returns
        -------
        d : str
            Direction code for the next d.
        """
        return self.rng.choice(Direction, p=self.probabilities).name


    def check_for_loop(self, shape: str) -> bool:
        """
        Scans shape's path for the possible loop at the next d.
        The loop is likely to occur if we have been walking in a plane defined by two orthogonal directions.

        We need to look out for the loop after 3rd d/hop by comparing the directions taken at first and last ds.
        If these two directions come to be opposite, e.g. 'l' and 'r', there is a chance to enter the loop next time.

        Parameters
        ----------
        shape : str
            Shape's path at this moment.

        Returns
        -------   
        b : bool
            Boolean indication for the possibility of a loop at the next d.
        """
        d_start, d_end = shape[0], shape[-1]  # directions at t=1 and t=3
        return Direction[d_start].axis == Direction[d_end].axis


    def check_for_overlap(self, bend_point_1: int, bend_point_2: int) -> bool:
        """
        """
        return (bend_point_2 - bend_point_1) == 1


    def generate(self) -> ShapeString:
        """
        Generates the shape's path/route by walking in 3D space and
        iteratevely updating the distribution over possible directions to walk at each d.

        Parameters
        ----------
        n_arms : int, default=4
            Number of arms the shape should have

        d_size : int, default=2
            How much we need to walk along the direction at each d.


        Returns
        -------
        path : ShapeString
            The seqeunce of characters outlining the path/route one needs to walk to form the arm-like shape.
            The path length is calculated by the following rule: len = n_hops x d_size + 1
        """
        path = ""
        overlap_likely = False
        loop_likely = False

        # make 1st arm
        bend_1 = self.rng.integers(1, 6)
        bend_2 = self.rng.integers(bend_1 + 2, 8)
        bend_3 = self.rng.integers(bend_2 + 1, 9)

        d = "U"  # always start with walking in the up direction
        for t in range(9):
            if t == bend_1:
                d = "B"

            if t == bend_2:
                # reset all the probabilities to default values
                self.reset_probabilities()
                # update the probability distribution of possible directions to go next
                # we need to mask the last direction drew and the one opposite to it
                # and distribute the probability mass among the rest of the available directions
                self.update_probabilities(path, overlap_likely, loop_likely)
                d = self.draw_direction()

            if t == bend_3:
                # check 1
                overlap_likely = self.check_for_overlap(bend_2, bend_3)
                # check 2
                loop_likely = self.check_for_loop(path)

                self.reset_probabilities()
                self.update_probabilities(path, overlap_likely, loop_likely)

                d = self.draw_direction()

            path += d

        return ShapeString(path)


class Cuboid:
    """
    Box/cube geometry class.

    Generates three-dimensional box-like shape. A cube has 8 vertices, 12 edges and 6 faces.

    Parameters
    ----------
    x : float, default=0.
        x-coordinate of cube's center

    y : float, default=0.
        y-coordinate of cube's center

    z : float, default=0.
        z-coordinate of cube's center

    width : float, default=2.
        cube's width

    height : float, default=2.
        cube's height

    depth : float, default=2.
        cube's depth
    """

    def __init__(
        self,
        x: float = 0.,
        y: float = 0.,
        z: float = 0.,
        width: float = 2.,
        height: float = 2.,
        depth: float = 2.
    ) -> None:
        self.xc = x
        self.yc = y
        self.zc = z
        self.w = width
        self.h = height
        self.d = depth

        self.vertices = np.array([
            # bottom face
            (x - width/2, y - height/2, z + depth/2),
            (x - width/2, y - height/2, z - depth/2),
            (x + width/2, y - height/2, z - depth/2),
            (x + width/2, y - height/2, z + depth/2),
            # top face
            (x - width/2, y + height/2, z + depth/2),
            (x - width/2, y + height/2, z - depth/2),
            (x + width/2, y + height/2, z - depth/2),
            (x + width/2, y + height/2, z + depth/2),
        ]).T

        self.edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (3, 7),
            (2, 6)
        ]

        self.faces = [
            # start from bottom left vertex and move clock-wise
            (0, 1, 2, 3),  # bottom
            (4, 5, 6, 7),  # top
            (0, 4, 7, 3),  # front
            (1, 5, 6, 2),  # back
            (0, 1, 5, 4),  # left
            (3, 2, 6, 7),  # right
        ]

    @property
    def com(self) -> np.array:
        """Coordinates of cube's center of mass"""
        return np.mean(self.vertices, axis=1)


class MetzlerShape:
    """
    Generates coordinates of vertices for Metzler shape from the input shape string.

    Metzler shape is composed of ten solid cubes attached face-to-face
    forming a rigid armlike structure with exactly three right-angled "elbows".

    Parameters
    ----------
    shape : shape string of length 9
        Sequence of direction codes outlining the 3D shape
    """

    def __init__(self, shape: ShapeString | str) -> None:
        self.centers = [
            [0, 0, 0]  # always start off with the first cube located at the origin
        ]
        # calculate centroid coordinates of cubes forming the shape
        for t, d in enumerate(shape):
            # find along which axis we need to walk
            # and in which direction (positive or negative)
            axis, direct = Direction[d].axis.value, Direction[d].orientation
            
            # add the coordinates for the next cube by copying the last existing cube coordinates
            self.centers += [list(self.centers[t])]
            # adjust the coordinates by following the path
            self.centers[t+1][axis] += 2*direct

        # compute vertices of each cube in the shape
        self.cubes = [Cuboid(*center) for center in self.centers]
        self.vertices = np.hstack([cube.vertices for cube in self.cubes])
        assert self.vertices.shape == (self.cubes[0].vertices.shape[0], len(self.cubes) * self.cubes[0].vertices.shape[1]), \
            f"Error: incorrect shape for the vertex data, {self.vertices.shape}!"

        # position the wireframe in such a way that
        # its center of mass (COM) is at the origin of its local coordinate system
        self.vertices = translate(
            homogenize(self.vertices),
            *(-1 * np.mean(self.centers, axis=0))
        )[:-1, :]
        # update shape's COM coordinates
        self.com = np.mean(self.vertices, axis=1)
        
        assert np.allclose(
            self.com,
            np.zeros_like(self.com)
        ), "Error: shape's center of mass is not at the origin (0, 0, 0)!"

        self.edges = []
        for cnt, cube in enumerate(self.cubes):
            self.edges += (np.array(cube.edges) + 8*cnt).tolist()

        self.faces = []
        for cnt, cube in enumerate(self.cubes):
            self.faces += (np.array(cube.faces) + 8*cnt).tolist()

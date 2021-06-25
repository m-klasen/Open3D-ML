import numpy as np
import open3d as o3d

Colors = [[0., 0., 0.], [0.96078431, 0.58823529, 0.39215686],
		[0.96078431, 0.90196078, 0.39215686],
		[0.58823529, 0.23529412, 0.11764706],
		[0.70588235, 0.11764706, 0.31372549], [1., 0., 0.],
		[0.11764706, 0.11764706, 1.], [0.78431373, 0.15686275, 1.],
		[0.35294118, 0.11764706, 0.58823529], [1., 0., 1.],
		[1., 0.58823529, 1.], [0.29411765, 0., 0.29411765],
		[0.29411765, 0., 0.68627451], [0., 0.78431373, 1.],
		[0.19607843, 0.47058824, 1.], [0., 0.68627451, 0.],
		[0., 0.23529412, 0.52941176], [0.31372549, 0.94117647, 0.58823529],
		[0.58823529, 0.94117647, 1.], [0., 0., 1.], [1.0, 1.0, 0.25],
		[0.5, 1.0, 0.25], [0.25, 1.0, 0.25], [0.25, 1.0, 0.5],
		[0.25, 1.0, 1.25], [0.25, 0.5, 1.25], [0.25, 0.25, 1.0],
		[0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.375, 0.375, 0.375],
		[0.5, 0.5, 0.5], [0.625, 0.625, 0.625], [0.75, 0.75, 0.75],
		[0.875, 0.875, 0.875]]

class BoundingBox3D:
    """Class that defines an axially-oriented bounding box."""

    next_id = 1

    def __init__(self,
                 center,
                 front,
                 up,
                 left,
                 size,
                 label_class,
                 confidence,
                 meta=None,
                 show_class=False,
                 show_confidence=False,
                 show_meta=None,
                 identifier=None,
                 arrow_length=1.0):
        """Creates a bounding box. Front, up, left define the axis of the box
        and must be normalized and mutually orthogonal.

        center: (x, y, z) that defines the center of the box
        front: normalized (i, j, k) that defines the front direction of the box
        up: normalized (i, j, k) that defines the up direction of the box
        left: normalized (i, j, k) that defines the left direction of the box
        size: (width, height, depth) that defines the size of the box, as
            measured from edge to edge
        label_class: integer specifying the classification label. If an LUT is
            specified in create_lines() this will be used to determine the color
            of the box.
        confidence: confidence level of the box
        meta: a user-defined string (optional)
        show_class: displays the class label in text near the box (optional)
        show_confidence: displays the confidence value in text near the box
            (optional)
        show_meta: displays the meta string in text near the box (optional)
        identifier: a unique integer that defines the id for the box (optional,
            will be generated if not provided)
        arrow_length: the length of the arrow in the front_direct. Set to zero
            to disable the arrow (optional)
        """
        assert (len(center) == 3)
        assert (len(front) == 3)
        assert (len(up) == 3)
        assert (len(left) == 3)
        assert (len(size) == 3)

        self.center = np.array(center, dtype="float32")
        self.front = np.array(front, dtype="float32")
        self.up = np.array(up, dtype="float32")
        self.left = np.array(left, dtype="float32")
        self.size = size
        self.label_class = label_class
        self.confidence = confidence
        self.meta = meta
        self.show_class = show_class
        self.show_confidence = show_confidence
        self.show_meta = show_meta
        if identifier is not None:
            self.identifier = identifier
        else:
            self.identifier = "box:" + str(BoundingBox3D.next_id)
            BoundingBox3D.next_id += 1
        self.arrow_length = arrow_length

    def __repr__(self):
        s = str(self.identifier) + " (class=" + str(
            self.label_class) + ", conf=" + str(self.confidence)
        if self.meta is not None:
            s = s + ", meta=" + str(self.meta)
        s = s + ")"
        return s

    @staticmethod
    def create_lines(boxes, lut=None):
        """Creates and returns an open3d.geometry.LineSet that can be used to
        render the boxes.

        boxes: the list of bounding boxes
        lut: a ml3d.vis.LabelLUT that is used to look up the color based on the
            label_class argument of the BoundingBox3D constructor. If not
            provided, a color of 50% grey will be used. (optional)
        """
        nverts = 14
        nlines = 12
        points = np.zeros((nverts * len(boxes), 3), dtype="float32")
        indices = np.zeros((nlines * len(boxes), 2), dtype="int32")
        colors = np.zeros((nlines * len(boxes), 3), dtype="float32")

        for i in range(0, len(boxes)):
            box = boxes[i]
            pidx = nverts * i
            x = 0.5 * box.size[0] * box.left
            y = 0.5 * box.size[1] * box.up
            z = 0.5 * box.size[2] * box.front
            # It seems to be substantially faster to assign directly for the
            # points, as opposed to points[pidx:pidx+nverts] = np.stack((...))
            points[pidx] = box.center + x + y + z
            points[pidx + 1] = box.center - x + y + z
            points[pidx + 2] = box.center - x + y - z
            points[pidx + 3] = box.center + x + y - z
            points[pidx + 4] = box.center + x - y + z
            points[pidx + 5] = box.center - x - y + z
            points[pidx + 6] = box.center - x - y - z
            points[pidx + 7] = box.center + x - y - z
            points[pidx + 8] = box.center + z

        # It is faster to break the indices and colors into their own loop.
        for i in range(0, len(boxes)):
            box = boxes[i]
            pidx = nverts * i
            idx = nlines * i
            indices[idx:idx +
                    nlines] = ((pidx, pidx + 1), (pidx + 1, pidx + 2),
                               (pidx + 2, pidx + 3), (pidx + 3, pidx),
                               (pidx + 4, pidx + 5), (pidx + 5, pidx + 6),
                               (pidx + 6, pidx + 7), (pidx + 7, pidx + 4),
                               (pidx + 0, pidx + 4), (pidx + 1, pidx + 5),
                               (pidx + 2, pidx + 6), (pidx + 3, pidx + 7)
                               )

            if lut is not None and len(lut)==len(boxes):
                c = lut[i]
            else:
                c = Colors[i] if i<len(Colors) else Colors[i-len(Colors)]

            colors[idx:idx +
                   nlines] = c  # copies c to each element in the range

        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(points)
        lines.lines = o3d.utility.Vector2iVector(indices)
        lines.colors = o3d.utility.Vector3dVector(colors)

        return lines

    @staticmethod
    def parse_o3d_boxes(boxes):
        return boxes

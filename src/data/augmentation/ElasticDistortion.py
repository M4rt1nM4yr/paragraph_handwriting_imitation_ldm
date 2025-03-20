#TODO citation?
import itertools

from PIL import Image
from numpy import random, floor


class ElasticDistortion:
    """
    Elastic Distortion adapted from https://github.com/IntuitionMachines/OrigamiNet
    Used in "OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold",
        Yousef, Mohamed and Bishop, Tom E., The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
    """
    def __init__(self, grid, magnitude, min_sep):

        self.grid_width, self.grid_height = grid
        self.xmagnitude, self.ymagnitude = magnitude
        self.min_h_sep, self.min_v_sep = min_sep

    def __call__(self, x):
        w, h = x.size

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []
        shift = [[(0, 0) for x in range(horizontal_tiles)] for y in range(vertical_tiles)]

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

                sm_h = min(self.xmagnitude, width_of_square - (self.min_h_sep + shift[vertical_tile][horizontal_tile - 1][
                    0])) if horizontal_tile > 0 else self.xmagnitude
                sm_v = min(self.ymagnitude, height_of_square - (self.min_v_sep + shift[vertical_tile - 1][horizontal_tile][
                    1])) if vertical_tile > 0 else self.ymagnitude

                dx = random.randint(-sm_h, self.xmagnitude)
                dy = random.randint(-sm_v, self.ymagnitude)
                shift[vertical_tile][horizontal_tile] = (dx, dy)

        shift = list(itertools.chain.from_iterable(shift))

        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for id, (a, b, c, d) in enumerate(polygon_indices):
            dx = shift[id][0]
            dy = shift[id][1]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        self.generated_mesh = generated_mesh

        return x.transform(x.size, Image.MESH, self.generated_mesh, resample=Image.BICUBIC)

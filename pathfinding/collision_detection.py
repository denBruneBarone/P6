def check_intersect(x1, y1, x2, y2, blockage_x, blockage_y, blockage_width, blockage_height, z1=None, z2=None,
                    blockage_z=None, blockage_depth=None):
    """
    Check if a line segment defined by points (x1, y1) and (x2, y2) intersects with any side of a rectangle.

    Parameters:
        x1, y1: Coordinates of the first endpoint of the line segment.
        x2, y2: Coordinates of the second endpoint of the line segment.
        blockage_x, blockage_y: Coordinates of the top-left corner of the rectangle.
        blockage_width: Width of the rectangle.
        blockage_height: Height of the rectangle.
        z1, z2: Coordinates of the first and second endpoint of the line segment along the z-axis (only for 3D).
        blockage_z: Coordinate of the front-top-left corner of the 3D blockage (only for 3D).
        blockage_depth: Depth of the 3D blockage along the z-axis (only for 3D).

    Returns:
        True if the line segment intersects with any side of the rectangle, False otherwise.
    """

    # Nested function to determine if three points are in a counter-clockwise orientation
    def ccw(ax, ay, bx, by, cx, cy):
        """
        Determine if three points are in a counter-clockwise orientation.

        Parameters:
            ax, ay: Coordinates of the first point.
            bx, by: Coordinates of the second point.
            cx, cy: Coordinates of the third point.

        Returns:
            True if the points are in a counter-clockwise orientation, False otherwise.
        """
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

    # Nested function to check if a line segment intersects any side of the rectangle
    def intersects_side(x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Check if a line segment intersects with any side of a rectangle.

        Parameters:
            x1, y1: Coordinates of the first endpoint of the line segment.
            x2, y2: Coordinates of the second endpoint of the line segment.
            x3, y3: Coordinates of one endpoint of a side of the rectangle.
            x4, y4: Coordinates of the other endpoint of the same side of the rectangle.

        Returns:
            True if the line segment intersects with the side of the rectangle, False otherwise.
        """
        return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and ccw(x1, y1, x2, y2, x3, y3) != ccw(x1,
                                                                                                                 y1,
                                                                                                                 x2,
                                                                                                                 y2,
                                                                                                                 x4,
                                                                                                                 y4)

    if z1 is None or z2 is None:  # 2D case
        # Check for intersection with each side of the rectangle
        return (
                intersects_side(x1, y1, x2, y2, blockage_x, blockage_y, blockage_x + blockage_width, blockage_y) or
                intersects_side(x1, y1, x2, y2, blockage_x + blockage_width, blockage_y,
                                blockage_x + blockage_width,
                                blockage_y + blockage_height) or
                intersects_side(x1, y1, x2, y2, blockage_x, blockage_y + blockage_height,
                                blockage_x + blockage_width,
                                blockage_y + blockage_height) or
                intersects_side(x1, y1, x2, y2, blockage_x, blockage_y, blockage_x, blockage_y + blockage_height)
        )
    else:  # 3D case
        # Nested function to check if a line segment intersects with any side of a cuboid in 3D
        def intersects_side_3d(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
            return (intersects_side(x1, y1, x2, y2, x3, y3, x4, y4) and min(z1, z2) <= min(z3, z4) <= max(z1,
                                                                                                          z2)) or \
                (intersects_side(x1, y1, x2, y2, x3, y3, x4, y4) and min(z1, z2) <= max(z3, z4) <= max(z1, z2)) or \
                (intersects_side(x1, y1, x2, y2, x3, y3, x4, y4) and min(z3, z4) <= min(z1, z2) <= max(z3, z4)) or \
                (intersects_side(x1, y1, x2, y2, x3, y3, x4, y4) and min(z3, z4) <= max(z1, z2) <= max(z3, z4))

        # Check for intersection with each side of the 3D blockage
        return (
                intersects_side_3d(x1, y1, z1, x2, y2, z2, blockage_x, blockage_y, blockage_z,
                                   blockage_x + blockage_width, blockage_y, blockage_z) or
                intersects_side_3d(x1, y1, z1, x2, y2, z2, blockage_x + blockage_width, blockage_y, blockage_z,
                                   blockage_x + blockage_width, blockage_y + blockage_height, blockage_z) or
                intersects_side_3d(x1, y1, z1, x2, y2, z2, blockage_x, blockage_y + blockage_height, blockage_z,
                                   blockage_x + blockage_width, blockage_y + blockage_height, blockage_z) or
                intersects_side_3d(x1, y1, z1, x2, y2, z2, blockage_x, blockage_y, blockage_z, blockage_x,
                                   blockage_y + blockage_height, blockage_z) or
                intersects_side_3d(x1, y1, z1, x2, y2, z2, blockage_x, blockage_y, blockage_z + blockage_depth,
                                   blockage_x + blockage_width, blockage_y, blockage_z + blockage_depth) or
                intersects_side_3d(x1, y1, z1, x2, y2, z2, blockage_x + blockage_width, blockage_y,
                                   blockage_z + blockage_depth, blockage_x + blockage_width,
                                   blockage_y + blockage_height, blockage_z + blockage_depth) or
                intersects_side_3d(x1, y1, z1, x2, y2, z2, blockage_x, blockage_y + blockage_height,
                                   blockage_z + blockage_depth, blockage_x + blockage_width,
                                   blockage_y + blockage_height, blockage_z + blockage_depth) or
                intersects_side_3d(x1, y1, z1, x2, y2, z2, blockage_x, blockage_y, blockage_z + blockage_depth,
                                   blockage_x, blockage_y + blockage_height, blockage_z + blockage_depth) or
                intersects_side_3d(x1, y1, z1, x2, y2, z2, blockage_x, blockage_y, blockage_z, blockage_x,
                                   blockage_y, blockage_z + blockage_depth) or
                intersects_side_3d(x1, y1, z1, x2, y2, z2, blockage_x + blockage_width, blockage_y, blockage_z,
                                   blockage_x + blockage_width, blockage_y, blockage_z + blockage_depth) or
                intersects_side_3d(x1, y1, z1, x2, y2, z2, blockage_x + blockage_width,
                                   blockage_y + blockage_height, blockage_z, blockage_x + blockage_width,
                                   blockage_y + blockage_height, blockage_z + blockage_depth) or
                intersects_side_3d(x1, y1, z1, x2, y2, z2, blockage_x, blockage_y + blockage_height, blockage_z,
                                   blockage_x, blockage_y + blockage_height, blockage_z + blockage_depth) or
                intersects_side_3d(x1, y1, z1, x2, y2, z2, blockage_x, blockage_y, blockage_z, blockage_x,
                                   blockage_y, blockage_z + blockage_depth)
        )


def check_segment_intersects_blockage(xs, ys, zs, blockage):
    if zs is not None:
        # If 3-Dimensional
        for i in range(len(xs) - 1):
            # Accessing the coordinates of the line segment
            x1, y1, z1 = xs[i], ys[i], zs[i]
            x2, y2, z2 = xs[i + 1], ys[i + 1], zs[i + 1]

            block_size_x = blockage[0].shape[0]
            block_size_y = blockage[0].shape[1]
            block_size_z = blockage[0].shape[2]
            block_pos_x1 = blockage[1][0]
            block_pos_y1 = blockage[1][1]
            block_pos_z1 = blockage[1][2]

            block_pos_x2 = blockage[1][0] + block_size_x
            block_pos_y2 = blockage[1][1] + block_size_y
            block_pos_z2 = blockage[1][2] + block_size_z

            # Check if both endpoints are inside the blockage
            if (block_pos_x1 <= x1 < block_pos_x2 and
                    block_pos_y1 <= y1 < block_pos_y2 and
                    block_pos_z1 <= z1 < block_pos_z2 and
                    block_pos_x1 <= x2 < block_pos_x2 and
                    block_pos_y1 <= y2 < block_pos_y2 and
                    block_pos_z1 <= z2 < block_pos_z2):
                print('Collision in 3D Block found')
                return True

            # Check if the line segment intersects with the blockage's bounding box
            intersect = check_intersect(x1, y1, z1, x2, y2, z2, block_pos_x1, block_pos_y1, block_pos_z1,
                                        block_size_x, block_size_y, block_size_z)
            if intersect:
                print('Collision in 3D Block found')
                return True

            # Check if either endpoint of the line segment is within the blockage
            if ((
                    block_pos_x1 <= x1 <= block_pos_x2 and block_pos_y1 <= y1 <= block_pos_y2 and block_pos_z1 <= z1 <= block_pos_z2) or
                    (
                            block_pos_x1 <= x2 <= block_pos_x2 and block_pos_y1 <= y2 <= block_pos_y2 and block_pos_z1 <= z2 <= block_pos_z2) or
                    # If the line segment crosses the blockage along x-axis
                    (x1 < block_pos_x1 and x2 > block_pos_x2 and
                     (block_pos_y1 <= y1 <= block_pos_y2 and block_pos_z1 <= z1 <= block_pos_z2 or
                      block_pos_y1 <= y2 <= block_pos_y2 and block_pos_z1 <= z2 <= block_pos_z2)) or
                    # If the line segment crosses the blockage along y-axis
                    (y1 < block_pos_y1 and y2 > block_pos_y2 and
                     (block_pos_x1 <= x1 <= block_pos_x2 and block_pos_z1 <= z1 <= block_pos_z2 or
                      block_pos_x1 <= x2 <= block_pos_x2 and block_pos_z1 <= z2 <= block_pos_z2)) or
                    # If the line segment crosses the blockage along z-axis
                    (z1 < block_pos_z1 and z2 > block_pos_z2 and
                     (block_pos_x1 <= x1 <= block_pos_x2 and block_pos_y1 <= y1 <= block_pos_y2 or
                      block_pos_x1 <= x2 <= block_pos_x2 and block_pos_y1 <= y2 <= block_pos_y2))):
                print('Collision in 3D Block found')
                return True

        return False

    else:
        # If 2-Dimensional
        for i in range(len(xs) - 1):
            # Accessing the first zero'th element of the 1st list. That is (30,30,0) -> 30 (The blockages position)
            x1, y1 = xs[i], ys[i]
            x2, y2 = xs[i + 1], ys[i + 1]

            block_size_x = blockage[0].shape[0]
            block_size_y = blockage[0].shape[1]
            block_pos_x1 = blockage[1][0]
            block_pos_y1 = blockage[1][1]

            block_pos_x2 = blockage[1][0] + block_size_x
            block_pos_y2 = blockage[1][1] + block_size_y

            # Check if both endpoints are inside the blockage
            if (block_pos_x1 <= x1 < block_pos_x2 and
                    block_pos_y1 <= y1 < block_pos_y2 and
                    block_pos_x1 <= x2 < block_pos_x2 and
                    block_pos_y1 <= y2 < block_pos_y2):
                print('Collision in 2D Block found')
                return True

            # Check if the line segment intersects with the blockage's bounding box
            intersect = check_intersect(x1, y1, x2, y2, block_pos_x1, block_pos_y1, block_size_x,
                                        block_size_y)
            if intersect:
                print('Collision in 2D Block found')
                return True

            # Check if either endpoint of the line segment is within the blockage
            if ((block_pos_x1 <= x1 <= block_pos_x2 and block_pos_y1 <= y1 <= block_pos_y2) or
                    (block_pos_x1 <= x2 <= block_pos_x2 and block_pos_y1 <= y2 <= block_pos_y2) or
                    # If the line segment crosses the blockage horizontally
                    (x1 < block_pos_x1 and x2 > block_pos_x2 and
                     (block_pos_y1 <= y1 <= block_pos_y2 or block_pos_y1 <= y2 <= block_pos_y2)) or
                    # If the line segment crosses the blockage vertically
                    (y1 < block_pos_y1 and y2 > block_pos_y2 and
                     (block_pos_x1 <= x1 <= block_pos_x2 or block_pos_x1 <= x2 <= block_pos_x2))):
                print('Collision in 2D Block found')
                return True
        return False

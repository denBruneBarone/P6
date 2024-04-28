import numpy as np


def ray_intersects_blockage(start, end, block_position, block_size):
    """
    Check if a ray defined by a line segment intersects with a blockage in 3D space.
    """

    # Calculate direction vector of the ray--
    direction = end - start

    # Check for intersection with each face of the blockage
    for i in range(3):  # Iterate over each axis (x, y, z)
        if direction[i] != 0:  # Ensure non-zero direction component along the axis
            # Calculate parameters where the ray intersects the faces of the blockage along the axis
            t1 = (block_position[i] - start[i]) / direction[i]
            t2 = ((block_position[i] + block_size[i]) - start[i]) / direction[i]
            tmin = min(t1, t2)
            tmax = max(t1, t2)

            # Intersection along this axis
            if tmax < 0 or tmin > 1:
                return False  # No intersection on this axis
        else:
            # Ray is parallel to the plane, check if it's inside the blockage on this axis
            if start[i] < block_position[i] or start[i] > block_position[i] + block_size[i]:
                return False  # No intersection


    return block_position[2] + block_size[2]  # Z-coordinate of the upper surface of the blockage


def check_segment_intersects_blockages(xs, ys, zs, blockages, return_intersection_z_value=False):
    """
    Check if a line segment intersects with any blockage in 3D space.
    """
    # TODO: ændr formattet til tuples eller Nodes

    max_z_intersection = float("-inf")

    # Check if xs, ys, zs are arrays or integers
    if isinstance(xs, (int, float)):
        xs = [xs]
        ys = [ys]
        zs = [zs]

    for blockage in blockages:
        # Extract blockage properties
        block_position = blockage[1]
        block_size_x, block_size_y, block_size_z = blockage[0].shape

        for i in range(len(xs) - 1):
            # Accessing coordinates of the line segment
            start = np.array([xs[i], ys[i], zs[i]])
            end = np.array([xs[i + 1], ys[i + 1], zs[i + 1]])

            # Check if the segment intersects with the blockage using ray tracing
            intersect = ray_intersects_blockage(start, end, block_position, (block_size_x, block_size_y, block_size_z))
            if intersect is not False:
                if return_intersection_z_value:
                    max_z_intersection = max(max_z_intersection, max(start[2], end[2]), intersect)
                else:
                    return True

    if return_intersection_z_value:
        return max_z_intersection
    else:
        return False
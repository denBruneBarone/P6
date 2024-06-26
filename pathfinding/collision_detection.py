import numpy as np


# def ray_intersects_blockage(start, end, block_position, block_size):
#     """
#     Check if a ray defined by a line segment intersects with a blockage in 3D space.
#     """
#
#     # Calculate direction vector of the ray
#     direction = end - start
#
#     # Check for intersection with each face of the blockage
#     for i in range(3):  # Iterate over each axis (x, y, z)
#         if direction[i] != 0:  # Ensure non-zero direction component along the axis
#             # Calculate parameters where the ray intersects the faces of the blockage along the axis
#             t1 = (block_position[i] - start[i]) / direction[i]
#             t2 = ((block_position[i] + block_size[i]) - start[i]) / direction[i]
#             tmin = min(t1, t2)
#             tmax = max(t1, t2)
#
#             # Intersection along this axis
#             if tmax < 0 or tmin > 1:
#                 return False  # No intersection on this axis
#         else:
#             # Ray is parallel to the plane, check if it's inside the blockage on this axis
#             if start[i] < block_position[i] or start[i] > block_position[i] + block_size[i]:
#                 return False  # No intersection
#
#     return True  # Intersection detected


def ray_intersects_blockage(start, end, block_position, block_size):
    """
    Check if a ray defined by a line segment intersects with a blockage in 3D space.
    """

    # Calculate direction vector of the ray
    direction = end - start

    # Initialize variables to track intersection along each axis
    intersects = [False, False, False]

    # Check for intersection with each face of the blockage along each axis
    for i in range(3):  # Iterate over each axis (x, y, z)
        if direction[i] != 0:  # Ensure non-zero direction component along the axis
            # Calculate parameters where the ray intersects the faces of the blockage along the axis
            t1 = (block_position[i] - start[i]) / direction[i]
            t2 = ((block_position[i] + block_size[i]) - start[i]) / direction[i]
            tmin = min(t1, t2)
            tmax = max(t1, t2)

            # Intersection along this axis
            if tmax >= 0 and tmin <= 1:  # Intersection occurs within the segment
                intersects[i] = True
        else:
            # Ray is parallel to the plane, check if it's inside the blockage on this axis
            if start[i] >= block_position[i] and start[i] <= block_position[i] + block_size[i]:
                intersects[i] = True  # Ray starts inside the blockage on this axis

    # Check if there's an intersection along all axes
    if all(intersects):
        return True
    else:
        return False


def check_segment_intersects_blockages(xs, ys, zs, blockages):
    """
    Check if a line segment intersects with any blockage in 3D space.
    """

    # Check if xs, ys, zs are arrays or integers
    if isinstance(xs, (int, float)):
        xs = [xs]
        ys = [ys]
        zs = [zs]

    for blockage in blockages:
        # Extract blockage properties
        block_position = blockage.positions
        block_size_x, block_size_y, block_size_z = blockage.np_array.shape

        for i in range(len(xs) - 1):
            # Accessing coordinates of the line segment
            start = np.array([xs[i], ys[i], zs[i]])
            end = np.array([xs[i + 1], ys[i + 1], zs[i + 1]])

            # Check if the segment intersects with the blockage using ray tracing
            intersect = ray_intersects_blockage(start, end, block_position, (block_size_x, block_size_y, block_size_z))
            if intersect:
                return True

    return False


# The following two functions are for baseline path

# def find_intersection_height(start, end, block_position, block_size):
#     """
#     Find the height of intersection between a line segment and a blockage in 3D space.
#     """
#     direction = end - start
#     t_values = []
#
#     for i in range(3):  # Iterate over each axis (x, y, z)
#         if direction[i] != 0:  # Ensure non-zero direction component along the axis
#             # Calculate parameters where the ray intersects the faces of the blockage along the axis
#             t1 = (block_position[i] - start[i]) / direction[i]
#             t2 = ((block_position[i] + block_size[i]) - start[i]) / direction[i]
#             tmin = min(t1, t2)
#             tmax = max(t1, t2)
#
#             # Check if the ray intersects the blockage along this axis
#             if tmax >= 0 and tmin <= 1:
#                 t_values.append((block_position[2] + block_size[2]) - start[2])
#
#     return max(t_values, default=float("-inf"))

def find_intersection_height(start, end, block_position, block_size):
    """
    Find the height of intersection between a line segment and a blockage in 3D space.
    """
    direction = end - start
    intersection_heights = []

    # Track if intersection occurs along all axes
    intersection_along_axes = [False, False]

    for i in range(2):  # Iterate over each axis (x, y, z)
        if direction[i] != 0:  # Ensure non-zero direction component along the axis
            # Calculate parameters where the ray intersects the faces of the blockage along the axis
            t1 = (block_position[i] - start[i]) / direction[i]
            t2 = ((block_position[i] + block_size[i]) - start[i]) / direction[i]
            tmin = min(t1, t2)
            tmax = max(t1, t2)

            # Check if the ray intersects the blockage along this axis
            if tmax >= 0 and tmin <= 1:
                # Calculate the height of intersection using the parameter t along the z-axis
                intersection_height = block_size[2]
                intersection_heights.append(intersection_height)
                intersection_along_axes[i] = True

    # Check if intersection occurs along all axes
    if all(intersection_along_axes):
        return max(intersection_heights, default=float("-inf"))
    else:
        return float("-inf")  # No intersection along all axes


def find_max_intersection_z(xs, ys, zs, blockages):
    """
    Find the maximum z-coordinate intersection point between a line segment and any blockage in 3D space.
    """
    max_z_intersection = 0

    # Check if xs, ys, zs are arrays or integers
    if isinstance(xs, (int, float)):
        xs = [xs]
        ys = [ys]
        zs = [zs]

    for blockage in blockages:
        # Extract blockage properties
        block_position = blockage.positions
        block_size_x, block_size_y, block_size_z = blockage.np_array.shape

        for i in range(len(xs) - 1):
            # Accessing coordinates of the line segment
            start = np.array([xs[i], ys[i], zs[i]])
            end = np.array([xs[i + 1], ys[i + 1], zs[i + 1]])

            # Find intersection height between the line segment and the blockage
            intersection_height = find_intersection_height(start, end, block_position,
                                                           (block_size_x, block_size_y, block_size_z))
            max_z_intersection = max(max_z_intersection, intersection_height)
    return max_z_intersection

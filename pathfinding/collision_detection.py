
def check_intersect_3d(x1, y1, z1, x2, y2, z2, blockage_x, blockage_y, blockage_z, blockage_width, blockage_height, blockage_depth):
    """
    Check if a line segment intersects with a 3D blockage.
    """
    # Check if any endpoint of the line segment is inside the blockage
    if (blockage_x <= x1 < blockage_x + blockage_width and
        blockage_y <= y1 < blockage_y + blockage_height and
        blockage_z <= z1 < blockage_z + blockage_depth) or \
       (blockage_x <= x2 < blockage_x + blockage_width and
        blockage_y <= y2 < blockage_y + blockage_height and
        blockage_z <= z2 < blockage_z + blockage_depth):
        return True

    # Check if the line segment intersects with any face of the blockage
    # By checking if any endpoint of the line segment is on opposite sides of any of the blockage's faces
    return (x1 < blockage_x and x2 > blockage_x + blockage_width) or \
           (y1 < blockage_y and y2 > blockage_y + blockage_height) or \
           (z1 < blockage_z and z2 > blockage_z + blockage_depth)


def check_segment_intersects_blockage(xs, ys, zs, blockage):
    """
    Check if a line segment intersects with a blockage in 2D or 3D.
    """
    # Extract blockage properties
    block_position = blockage[1]
    block_size_x, block_size_y, block_size_z = blockage[0].shape

    for i in range(len(xs) - 1):
        # Accessing coordinates of the line segment
        x1, y1, z1 = xs[i], ys[i], zs[i]
        x2, y2, z2 = xs[i + 1], ys[i + 1], zs[i + 1]

        # Check if the line segment intersects with the blockage
        intersect = check_intersect_3d(x1, y1, z1, x2, y2, z2, *block_position, block_size_x, block_size_y, block_size_z)
        if intersect:
            print('Collision with Block found')
            return True

    return False

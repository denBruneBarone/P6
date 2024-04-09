import matplotlib.pyplot as plt
import numpy as np


class Workspace:
    def __init__(self, dimensions, max_bounds):
        self.dimensions = dimensions
        self.max_bounds = max_bounds
        self.flight_paths = []
        self.blockages = []
        self.wind_field = []

    def add_wind_field(self, angle, wind_speed):
        # Use this for wind speed
        # call from main.py, just like blockage
        # self.wind_field.append()
        raise NotImplementedError

    def add_blockage(self, blockage_matrix, position):
        if len(blockage_matrix.shape) != self.dimensions:
            raise ValueError(f"Blockage matrix dimensions must match the space dimensions")
        for i in range(self.dimensions):
            if not (0 <= position[i] <= self.max_bounds[i]):
                raise ValueError(f"Blockage position must be within the specified bounds")
            if position[i] + blockage_matrix.shape[i] > self.max_bounds[i]:
                raise ValueError(f"Blockage does not fit within the space dimensions")
        self.blockages.append((blockage_matrix, position))

    def add_flight_path(self, flight_path):
        self.flight_paths.append(flight_path)

    def plot_space(self, dimension='3D', dpi=300):
        if dimension == '3D':
            '''
            The code below: iterates over each flight path and extracts the x, y, and z coordinates.
            Then, it breaks down each flight path into segments.
            For each segment, it checks if there is any intersection with any blockage by calling the check_segment_intersects_blockage method.
            If any segment intersects with any blockage, it plots that segment in orange (color='orange') and then continues to the next segment.
            If a segment does not intersect with any blockage, it plots it in blue (color='b').
            This process continues until all segments of the flight path have been processed.
            '''
            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')

            # Plot origin
            ax.scatter(0, 0, 0, color='k')

            # Plot blockages
            for blockage_matrix, position in self.blockages:
                x, y, z = position
                ax.bar3d(x, y, z, *blockage_matrix.shape, color='k', alpha=0.5, edgecolor='black', linewidth=0.5)

            # Plot flight paths
            for flight_path in self.flight_paths:
                xs, ys, zs = zip(*flight_path)

                # Iterate over each segment of the flight path
                for i in range(len(xs) - 1):
                    # Extract coordinates for the current segment
                    x_coords = [xs[i], xs[i + 1]]
                    y_coords = [ys[i], ys[i + 1]]
                    z_coords = [zs[i], zs[i + 1]]

                    # Check if the current segment intersects with any blockage
                    segment_intersects = any(
                        self.check_segment_intersects_blockage(x_coords, y_coords, z_coords, blockage)
                        for blockage in self.blockages
                    )

                    # Plot the segment in the appropriate color
                    if segment_intersects:
                        ax.plot(x_coords, y_coords, z_coords, color='r', alpha=0.5)
                    else:
                        ax.plot(x_coords, y_coords, z_coords, color='g', alpha=0.5)

            # Set labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim([0, self.max_bounds[0]])
            ax.set_ylim([0, self.max_bounds[1]])
            ax.set_zlim([0, self.max_bounds[2]])

        elif dimension == '2D':
            fig, ax = plt.subplots(dpi=dpi)

            # Plot origin
            ax.scatter(0, 0, color='k')

            # Plot blockages
            for blockage_matrix, position in self.blockages:
                x, y = position[:2]
                ax.add_patch(plt.Rectangle((x, y), blockage_matrix.shape[0], blockage_matrix.shape[1], color='k'))

            # Plot flight paths
            for flight_path in self.flight_paths:
                xs, ys, _ = zip(*flight_path)

                # Iterate over each segment of the flight path
                for i in range(len(xs) - 1):
                    # Extract coordinates for the current segment
                    x_coords = [xs[i], xs[i + 1]]
                    y_coords = [ys[i], ys[i + 1]]

                    # Check if the current segment intersects with any blockage
                    segment_intersects = any(
                        self.check_segment_intersects_blockage(x_coords, y_coords, None, blockage)
                        for blockage in self.blockages
                    )

                    # Plot the segment in the appropriate color
                    if segment_intersects:
                        ax.plot(x_coords, y_coords, color='r', alpha=0.5)
                    else:
                        ax.plot(x_coords, y_coords, color='g', alpha=0.5)

            # Set labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim([0, self.max_bounds[0]])
            ax.set_ylim([0, self.max_bounds[1]])
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.show()

    # def check_intersect(self, x1, y1, x2, y2, blockage_x, blockage_y, blockage_width, blockage_height):
    #     # This function contains nested functions
    #     def ccw(ax, ay, bx, by, cx, cy):
    #         return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
    #
    # # Check if the line segment intersects any side of the rectangle # ccw: counter-clockwise def intersects_side(
    # x1, y1, x2, y2, x3, y3, x4, y4): return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and ccw(x1,
    # y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4)
    #
    #     # Check for intersection with each side of the rectangle
    #     return (
    #             intersects_side(x1, y1, x2, y2, blockage_x, blockage_y, blockage_x + blockage_width, blockage_y) or
    #             intersects_side(x1, y1, x2, y2, blockage_x + blockage_width, blockage_y, blockage_x + blockage_width,
    #                             blockage_y + blockage_height) or
    #             intersects_side(x1, y1, x2, y2, blockage_x, blockage_y + blockage_height, blockage_x + blockage_width,
    #                             blockage_y + blockage_height) or
    #             intersects_side(x1, y1, x2, y2, blockage_x, blockage_y, blockage_x, blockage_y + blockage_height)
    #     )

    # def check_intersect(self, x1, y1, x2, y2, blockage_x, blockage_y, blockage_width, blockage_height):
    #     """
    #     Check if a line segment defined by points (x1, y1) and (x2, y2) intersects with any side of a rectangle.
    #
    #     Parameters:
    #         x1, y1: Coordinates of the first endpoint of the line segment.
    #         x2, y2: Coordinates of the second endpoint of the line segment.
    #         blockage_x, blockage_y: Coordinates of the top-left corner of the rectangle.
    #         blockage_width: Width of the rectangle.
    #         blockage_height: Height of the rectangle.
    #
    #     Returns:
    #         True if the line segment intersects with any side of the rectangle, False otherwise.
    #     """
    #
    #     # Nested function to determine if three points are in a counter-clockwise orientation
    #     def ccw(ax, ay, bx, by, cx, cy):
    #         """
    #         Determine if three points are in a counter-clockwise orientation.
    #
    #         Parameters:
    #             ax, ay: Coordinates of the first point.
    #             bx, by: Coordinates of the second point.
    #             cx, cy: Coordinates of the third point.
    #
    #         Returns:
    #             True if the points are in a counter-clockwise orientation, False otherwise.
    #         """
    #         return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
    #
    #     # Nested function to check if a line segment intersects any side of the rectangle
    #     def intersects_side(x1, y1, x2, y2, x3, y3, x4, y4):
    #         """
    #         Check if a line segment intersects with any side of a rectangle.
    #
    #         Parameters:
    #             x1, y1: Coordinates of the first endpoint of the line segment.
    #             x2, y2: Coordinates of the second endpoint of the line segment.
    #             x3, y3: Coordinates of one endpoint of a side of the rectangle.
    #             x4, y4: Coordinates of the other endpoint of the same side of the rectangle.
    #
    # Returns: True if the line segment intersects with the side of the rectangle, False otherwise. """ return ccw(
    # x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2,
    # x4, y4)
    #
    #     # Check for intersection with each side of the rectangle
    #     return (
    #             intersects_side(x1, y1, x2, y2, blockage_x, blockage_y, blockage_x + blockage_width, blockage_y) or
    #             intersects_side(x1, y1, x2, y2, blockage_x + blockage_width, blockage_y, blockage_x + blockage_width,
    #                             blockage_y + blockage_height) or
    #             intersects_side(x1, y1, x2, y2, blockage_x, blockage_y + blockage_height, blockage_x + blockage_width,
    #                             blockage_y + blockage_height) or
    #             intersects_side(x1, y1, x2, y2, blockage_x, blockage_y, blockage_x, blockage_y + blockage_height)
    #     )

    def check_intersect(self, x1, y1, x2, y2, blockage_x, blockage_y, blockage_width, blockage_height, z1=None, z2=None,
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

    def check_segment_intersects_blockage(self, xs, ys, zs, blockage):
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
                intersect = self.check_intersect(x1, y1, z1, x2, y2, z2, block_pos_x1, block_pos_y1, block_pos_z1,
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
                intersect = self.check_intersect(x1, y1, x2, y2, block_pos_x1, block_pos_y1, block_size_x,
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

    def find_optimal_path(self, mission):
        print('find optimal path')
        # Maybe here call plot_space to show the workspace
        start = mission.start
        end = mission.end
        payload = mission.payload
        blockages = self.blockages
        wind_field = self.wind_field

        path_example = [(0, 0, 0), (50, 50, 0), (200, 200, 0), (225, 225, 0), (250, 250, 0)]

        path = path_example
        return path

    def generate_random_path(self):
        # Access blockages directly from the workspace
        blockages = self.blockages

        # Generate random flight path data
        num_points = 50  # Number of points in the flight path
        min_coord, max_coord = 0, 400  # Range for coordinates in each dimension

        # Generate random x, y, z coordinates for the flight path while avoiding blockages
        flight_path = []
        for _ in range(num_points):
            # Generate random coordinates
            x_coord = np.random.randint(min_coord, max_coord)
            y_coord = np.random.randint(min_coord, max_coord)
            z_coord = np.random.randint(min_coord, max_coord // 4)  # Adjusted for z-axis (height)
            print(f"Generated coordinates: ({x_coord}, {y_coord}, {z_coord})")

            # Check if the generated coordinates are within any blockages
            while any(
                    blockage[1][0] <= x_coord < blockage[1][0] + blockage[0].shape[0] and
                    blockage[1][1] <= y_coord < blockage[1][1] + blockage[0].shape[1] and
                    blockage[1][2] <= z_coord < blockage[1][2] + blockage[0].shape[2]
                    for blockage in blockages
            ):
                # Regenerate coordinates until they are outside all blockages
                x_coord = np.random.randint(min_coord, max_coord)
                y_coord = np.random.randint(min_coord, max_coord)
                z_coord = np.random.randint(min_coord, max_coord // 4)

                print(f"Regenerated coordinates: ({x_coord}, {y_coord}, {z_coord})")

            # Append the valid coordinates to the flight path
            flight_path.append((x_coord, y_coord, z_coord))

        return flight_path

    @staticmethod
    def generate_path_completely_fill_blockage(grid_size=400, blockage_size=50, blockage_x=250, blockage_y=250):
        path = []

        # Moving along the top boundary
        for x in range(0, grid_size):
            path.append((x, 0, 0))

        # Zigzag pattern towards the blockage from the top
        for y in range(1, grid_size):
            if y % 2 == 1:  # Odd rows move right
                for x in range(grid_size - 1, -1, -1):
                    path.append((x, y, 0))
            else:  # Even rows move left
                for x in range(0, grid_size):
                    path.append((x, y, 0))

        return path

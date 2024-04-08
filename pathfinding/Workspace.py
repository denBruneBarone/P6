import matplotlib.pyplot as plt


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
        #self.wind_field.append()
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

    def check_segment_intersects_blockage(self, xs, ys, zs, blockage):
        if zs is not None:
            # If 3-Dimensional
            for i in range(len(xs) - 1):
                if (
                        (blockage[1][0] <= xs[i] < blockage[1][0] + blockage[0].shape[0] and
                         blockage[1][1] <= ys[i] < blockage[1][1] + blockage[0].shape[1] and
                         blockage[1][2] <= zs[i] < blockage[1][2] + blockage[0].shape[2])
                        or
                        (blockage[1][0] <= xs[i + 1] < blockage[1][0] + blockage[0].shape[0] and
                         blockage[1][1] <= ys[i + 1] < blockage[1][1] + blockage[0].shape[1] and
                         blockage[1][2] <= zs[i + 1] < blockage[1][2] + blockage[0].shape[2])
                ):
                    print('Collision in 3D Block found')
                    return True
            return False
        else:
            # If 2-Dimensional
            for i in range(len(xs) - 1):
                # Accessing the first zero'th element of the 1st list. That is (30,30,0) -> 30 (The blockages position)
                if (
                        # Checking if a given point collides with the blockage's position(start) and position+size(end)
                        (blockage[1][0] <= xs[i] < blockage[1][0] + blockage[0].shape[0] and
                         blockage[1][1] <= ys[i] < blockage[1][1] + blockage[0].shape[1])
                        or
                        (blockage[1][0] <= xs[i + 1] < blockage[1][0] + blockage[0].shape[0] and
                         blockage[1][1] <= ys[i + 1] < blockage[1][1] + blockage[0].shape[1])
                ):
                    print('Collision in 2D Block found')
                    return True
            return
        # True: intersection between segment and blockage
        # False: No intersection between segment and blockage

    def find_optimal_path(self, mission):
        print('hello world')
        # Maybe here call plot_space to show the workspace

        flight_path = [(0, 0, 0), (5, 5, 0), (10, 10, 0), (15, 15, 0), (20, 20, 0), (25, 25, 0), (30, 30, 0),
                       (35, 35, 0),
                       (40, 40, 0), (45, 45, 0), (50, 50, 0), (55, 55, 0), (60, 60, 0), (65, 65, 0), (70, 70, 0),
                       (75, 75, 0), (80, 80, 0)]

        return flight_path

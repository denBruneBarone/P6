class Node:
    def __init__(self, x, y, z=None, velocity=None, velocity_x=None, velocity_y=None, velocity_z=None):
        self.x = x
        self.y = y
        self.z = z
        self.velocity = velocity
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.velocity_z = velocity_z

        if velocity is None:
            self.velocity = 0
        if velocity_x is None:
            self.velocity_x = 0
        if velocity_y is None:
            self.velocity_y = 0
        if velocity_z is None:
            self.velocity_z = 0

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __lt__(self, other):
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)

    def __sub__(self, other):
        return self.x - other.x

    def __gt__(self, other):
        return (self.x, self.y, self.z) > (other.x, other.y, other.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z, self.velocity))

    def __str__(self):
        return f"Node(x={self.x}, y={self.y}, z={self.z}, velocity={self.velocity}, " \
               f"velocity_x={self.velocity_x}, velocity_y={self.velocity_y}, velocity_z={self.velocity_z})"

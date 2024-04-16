class Node:
    def __init__(self, x, y, z=None, velocity=None):
        self.x = x
        self.y = y
        self.z = z
        self.velocity = velocity
        if velocity is None:
            self.velocity = 0

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __lt__(self, other):
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)

    def __gt__(self, other):
        return (self.x, self.y, self.z) > (other.x, other.y, other.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z, self.velocity))
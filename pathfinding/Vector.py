class Vector:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def __str__(self):
        return str(self.coordinates)

    def __add__(self, other):
        if len(self.coordinates) != len(other.coordinates):
            raise ValueError("Vectors must have the same length")
        return Vector([x + y for x, y in zip(self.coordinates, other.coordinates)])

    def __mul__(self, scalar):
        return Vector([scalar * x for x in self.coordinates])

    def dot_product(self, other):
        if len(self.coordinates) != len(other.coordinates):
            raise ValueError("Vectors must have the same length")
        return sum(x * y for x, y in zip(self.coordinates, other.coordinates))

# Example usage
v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

# Vector addition
print("Vector addition:", v1 + v2)

# Scalar multiplication
scalar = 2
print("Scalar multiplication:", v1 * scalar)

# Dot product
print("Dot product:", v1.dot_product(v2))

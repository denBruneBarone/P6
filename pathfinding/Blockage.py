import numpy as np


class Blockage:
    def __init__(self, size_x, size_y, size_z, pos_x, pos_y, pos_z, b_type):
        if b_type == 'takeoff':
            self.np_array = np.zeros((size_x, size_y, size_z))
        elif b_type == 'landing':
            self.np_array = np.zeros((size_x, size_y, size_z))
        elif b_type == 'obstacle':
            self.np_array = np.ones((size_x, size_y, size_z))
        elif b_type == 'no-fly-zone':
            self.np_array = np.ones((size_x, size_y, size_z))
        else:
            raise ValueError("Invalid blockage type")
        self.positions = [pos_x, pos_y, pos_z]
        self.type = b_type

class Prediction:
    class BoundingBox:
        def __init__(self, start_x=0, start_y=0, end_x=0, end_y=0):
            self.start_x = start_x
            self.start_y = start_y
            self.end_x = end_x
            self.end_y = end_y

        def area(self):
            return (self.end_x-self.start_x) * (self.end_y-self.start_y)

    def __init__(self, out, bounding_box):
        self.out = out
        self.bounding_box = bounding_box
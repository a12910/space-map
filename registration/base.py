import spacemap

class Registration:
    def __init__(self, name):
        self.name = name
        self.init_grid = None
        
    def load_init_grid(self, grid):
        self.init_grid = grid

    def run(self):
        pass
    
    def load_params_path(self, path):
        pass
    
    def load_params(self, params):
        pass
    
    def output_params(self):
        pass
    
    def output_params_path(self, path):
        pass
    
    def generate_img_grid(self):
        pass
    
    def apply_points2d(self, points):
        pass
    
    def apply_img(self, img):
        if self.init_grid is not None:
            img = spacemap.img.apply_img_by_Grid(img, self.init_grid)
        return img
    
    def load_img(self, imgI, imgJ):
        if self.init_grid is not None:
            imgJ = spacemap.img.apply_img_by_Grid(imgJ, self.init_grid)
        return imgI, imgJ

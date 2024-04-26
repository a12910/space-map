import spacemap
from spacemap import Slice


class LDMPairHE:
    def __init__(self, sliceI: Slice, sliceJ: Slice, fromKey, toKey, gpu=None, err=0.1):
        self.sI = sliceI
        self.sJ = sliceJ
        self.fromKey = fromKey
        self.toKey = toKey
        self.err = err
        self.gpu = gpu
        self.outputs = None
        self.grid = None
    
    def run(self, show, outputs=None):
        # 使用HE同时优化HE和points
        
        sI, sJ = self.sI, self.sJ
        imgI1 = sI.create_img(self.toKey, useDF=False)
        imgJ2 = sJ.create_img(self.fromKey, useDF=False)
        # imgI1, imgJ2 = spacemap.img_norm(imgI1, imgJ2)
        
        # ldm: J -> I
        ldm = spacemap.LDDMM2D(template=imgJ2,target=imgI1,
                              do_affine=1,do_lddmm=0,
                              a=7,
                              optimizer='adam',
                              sigma=20.0,sigmaR=40.0,
                              gpu_number=self.gpu,
                              target_err=0.1,
                              verbose=100,
                              target_step=20000,
                              show_init=False)
        spacemap.Info("Start LDMPair: %s/%s -> %s/%s" % (sJ.index, self.toKey, sI.index, self.fromKey))
        
        if outputs is not None:
            ldm.loadTransforms(*outputs)
            ldm.setParams('target_err_skip', self.err)
            ldm.setParams('epsilon', 1000)
            ldm.setParams('niter', 20000)
            ldm.setParams('do_lddmm', 1)
            ldm.run()
            ldm.setParams('epsilon', 1)
            ldm.setParams('niter', 20000)
            ldm.run()
        else:
            spacemap.lddmm_main(ldm, self.err)
        spacemap.Info("Finish LDMPair: %s/%s -> %s/%s" % (sJ.index, self.toKey, sI.index, self.fromKey))
        
        # 保存J-Img
        imgJ3 = ldm.applyThisTransformNT2d(imgJ2)
        sJ.save_value_img(imgJ3, self.toKey)
        
        # 保存J-points
        psI1 = sI.to_points(self.toKey)
        psI2 = ldm.applyThisTransformPoints2D(psI1)
        grid = spacemap.grid.GridGenerate(imgI1.shape, spacemap.XYD)
        self.grid = grid
        grid.init_db(psI2, psI1)
        grid.generate()
        grid.fix()
        psJ1 = sJ.to_points(self.fromKey)
        psJ2 = grid.grid_sample_points(psJ1)
        sJ.save_value_points(psJ2, self.toKey)
        self.outputs = ldm.outputTransforms()
        if show:
            self.show_err()
        return self.outputs

    def show_err(self, err=None):
        if err is None:
            err = spacemap.find.default()
        imgI1 = self.sI.create_img(self.toKey, useDF=False)
        imgJ2 = self.sJ.create_img(self.fromKey, useDF=False)
        imgJ3 = self.sJ.create_img(self.toKey, useDF=True)
        imgJ3_ = self.sJ.create_img(self.toKey, useDF=False)
        e1 = err.err(imgI1, imgJ2)
        e2 = err.err(imgI1, imgJ3)
        e3 = err.err(imgI1, imgJ3_)
        spacemap.Info("Err: %f -> %f/%f" % (e1, e2, e3))
        
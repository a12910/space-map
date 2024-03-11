lddmm = torch_lddmm.LDDMM(template=template_image,target=target_image,
                          outdir="/content/sample_data",
                          do_affine=1,do_lddmm=0,
                          a=7,niter=100,epsilon=40,
                          v_scale=1.0,
                          minbeta=2,
                          # optimizer='gdr',
                          optimizer='adam',
                          sigma=20.0,sigmaR=40.0,
                          gpu_number=0,
                          target_step=20000)
# lddmm.setParams('epsilonL',1e-7)
# lddmm.setParams('epsilonT',2e-5)
lddmm.setParams('v_scale', 8.0)
lddmm.setParams('minbeta', 2)
lddmm.setParams('epsilon', 10000)
lddmm.setParams('niter', 1000)
lddmm.run()
lddmm.setParams('epsilon', 1000)
lddmm.setParams('v_scale', 4.0)
lddmm.setParams('minbeta', 1)
lddmm.setParams('niter', 1000)
lddmm.run()
lddmm.setParams('epsilon', 50)
lddmm.setParams('v_scale', 1.0)
lddmm.setParams('minbeta', 0.1)
lddmm.setParams('niter', 6000)
lddmm.run()
lddmm.setParams('minbeta', 0.001)
lddmm.setParams('epsilon', 1)
lddmm.setParams('niter', 20000)
lddmm.setParams('do_lddmm', 1)
lddmm.run()

""" 
def range_filter(df, index, startx, endx, starty, endy):
    lis = df[df["layer"] == index].copy()
    lis = lis[(lis["x"] > startx) & (lis["x"] < endx) & (lis["y"] > starty) & (lis["y"] < endy)].copy()
    lis["x"] -= startx
    lis["y"] -= starty
    return lis, [0, endx - startx, 0, endy - starty]

index = 3
TLis, xyr = range_filter(df, index, 2500, 3500, 2000, 3000)

TARGET_S = lddmm.Slice(TLis, index, BASE)
TARGET_S.xyd = 5
TARGET_S.xyrange = xyr
NLis, _ = range_filter(df, index+1, 2500, 3500, 2000, 3000)

NEW_S = lddmm.Slice(NLis, index+1, BASE)
NEW_S.xyd = 5
NEW_S.xyrange = xyr

"""
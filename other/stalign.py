from STalign import STalign
import torch
torch.set_default_device('cuda:0')

iter = 5000
tag = "test1"

def stalign_compare(new, target, iter):
    xJ, yJ = target[:, 0], target[:, 1]
    xI, yI = new[:, 0], new[:, 1]
    XI,YI,I,_ = STalign.rasterize(xI,yI, dx=10, blur=1.5)
    XJ,YJ,J,_ = STalign.rasterize(xJ,yJ,dx=10, blur=1.5)
    extentI = STalign.extent_from_x((YI,XI))
    extentJ = STalign.extent_from_x((YJ,XJ))
    plt.show()
    params = {'niter': iter, 'device':"cuda:0", 'epV': 50}
    out = STalign.LDDMM([YI,XI],I,[YJ,XJ],J,**params)
    plt.show()
    A = out['A']
    v = out['v']
    xv = out['xv']

    phii = STalign.build_transform(xv,v,A,XJ=[YJ,XJ],direction='b')
    phiI = STalign.transform_image_source_to_target(xv,v,A,[YI,XI],I,[YJ,XJ])
    phii = phii.cpu()
    phiI = phiI.cpu()

    tpointsI= STalign.transform_points_source_to_target(xv,v,A, np.stack([yI, xI], 1))
    tpointsI = tpointsI.cpu()
    xI_LDDMM = np.array(tpointsI[:,1])
    yI_LDDMM = np.array(tpointsI[:,0])
    result = np.zeros((xI.shape[0], 2))
    result[:, 0] = xI_LDDMM
    result[:, 1] = yI_LDDMM
    return result

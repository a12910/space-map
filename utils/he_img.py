import scipy.signal as ss
import matplotlib.pyplot as plt 
import numpy as np
import cv2

def init_he_pair(imgI: np.array, imgJ: np.array):
    if len(imgJ.shape) == 3:
        imgJ = cv2.cvtColor(imgJ, cv2.COLOR_BGR2GRAY)
    if len(imgI.shape) == 3:
        imgI = cv2.cvtColor(imgI, cv2.COLOR_BGR2GRAY)
    
    imgI, imgJ = img_norm(imgI, imgJ)
    imgJ1 = fill_square(imgJ)
    
    scale = auto_scale(imgI, imgJ1)
    length2 = int(imgJ1.shape[0] * scale)
    
    imgJ2 = cv2.resize(imgJ1, (length2, length2))
    midJ = img_center(imgJ2, True)
    sIx, sIy = imgI.shape
    
    imgJ3 = np.zeros((length2*2, length2*2))
    
    imgJ3 = cut_img(imgJ2, midJ[0]-sIx//2, midJ[1]-sIy//2, 
                    midJ[0]+sIx//2, midJ[1]+sIy//2)
    H = rotate_H(0, (0, 0), scale)[0]
    H2 = np.eye(3)
    H2[1, 2] = -(midJ[1] - sIy // 2)
    H2[0, 2] = -(midJ[0] - sIx // 2)
    H = np.dot(H2, H)
    
    return imgI, imgJ3, H

def process_he(img):
    img = np.flip(img, axis=1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = split_he_background_otsu(img)
    img[img > 0] = 255
    return img

def conv_2d(img, kernel):
    img1 = ss.convolve2d(img, kernel, mode="same")
    return img1

def finder_err(imgI: np.array, imgJ: np.array):
    i1 = imgI.reshape(-1)
    i2 = imgJ.reshape(-1)
    ii = np.array([i1, i2])
    inter = ii.min(axis=0).sum()
    return -(2 * inter + 0.001) / (i1.sum() + i2.sum() + 0.001)

def rotate_H(rotate, center, scale):
    center = center[1], center[0]
    M = cv2.getRotationMatrix2D(center, rotate, scale)
    H = convert_M_to_H(M)
    return H, M

def rotate_imgH(imgJ, H, scale=None):
    if scale is not None:
        H = H * scale
    return rotate_imgM(imgJ, convert_H_to_M(H))

def rotate_imgM(imgJ, M):
    h, w = imgJ.shape[:2]
    rotatedI = cv2.warpAffine(imgJ, M, (w, h))
    return rotatedI

def convert_H_to_M(H):
    M = [[H[1, 1], H[1, 0], H[1, 2]], [H[0, 1], H[0, 0], H[0, 2]]]
    M = np.array(M)
    return M

def convert_M_to_H(M):
    H0 = [[M[1, 1], M[1, 0], M[1, 2]], [M[0, 1], M[0, 0], M[0, 2]]]
    H = np.eye(3)
    H[:2] = H0
    return H

def rotate_img(imgJ, rotate, center, scale):
    # print(rotate, center, scale)
    imgJ = np.array(imgJ.copy(), dtype=np.uint8)
    H, M = rotate_H(rotate, center, scale)
    rotatedI = rotate_imgH(imgJ, H)
    return rotatedI, H

def img_center(imgJ, base=False):
    if base:
        imgJ = imgJ.copy()
        imgJ[imgJ > np.min(imgJ)] = 255
    if len(imgJ.shape) > 2:
        imgJ = cv2.cvtColor(imgJ, cv2.COLOR_BGR2GRAY)
    Xs = np.sum(imgJ, axis=1)
    Ys = np.sum(imgJ, axis=0)
    dd = np.array(range(imgJ.shape[0]))
    midX = int(np.sum(Xs * dd) / np.sum(Xs))
    midY = int(np.sum(Ys * dd) / np.sum(Ys))
    return midX, midY

def img_norm(imgI, imgJ):
    imgI = imgI / np.max(imgI) * 255
    imgJ = imgJ / np.max(imgJ) * 255
    imgI = imgI.astype(np.uint8)   
    imgJ = imgJ.astype(np.uint8)
    imgI[imgI > 0] = 255
    imgJ[imgJ > 0] = 255
    return imgI, imgJ

def get_scale_base(img):
    img_ = img.copy()
    imgMin = 0
    img_[img_ > imgMin] = 255
    img_[img_ <= imgMin] = 0
    
    # area = np.sqrt(np.sum(img_) / 255)
    
    area1 = np.sum(img_, axis=0)
    limit = img.shape[0] * 255 * 0.1
    area2 = np.zeros_like(area1)
    area2[area1 > limit] = 1
    area = np.sum(area2)
    return area

def cut_img2(img, size):
    result = img[:size, :size]
    return result

def cut_img(img, minx, miny, maxx, maxy):
    if len(img.shape) == 3:
        result = np.zeros((maxy-miny, maxx-minx, img.shape[2]))
    else:
        result = np.zeros((maxy-miny, maxx-minx))
    minx1 = max(0, minx)
    miny1 = max(0, miny)
    maxx1 = min(maxx, img.shape[1])
    maxy1 = min(maxy, img.shape[0])
    
    minx2 = max(0, -minx)
    miny2 = max(0, -miny)
    maxx2 = min(minx2+maxx1-minx1, result.shape[1])
    maxy2 = min(miny2+maxy1-miny1, result.shape[0])
    try:
        result[minx2:maxx2, miny2:maxy2] = img[minx1:maxx1, miny1:maxy1]
    except:
        # 69 28 468 428 
        # 0 0 400 400 
        # 69 28 469 428
        # (468, 468)
        print(minx1, miny1, maxx1, maxy1, 
              minx2, miny2, maxx2, maxy2, 
              minx, miny, maxx, maxy, 
              img.shape)
        raise Exception("Error")
    
    return result

def auto_scale(imgI, imgJ):
    areaI = get_scale_base(imgI)
    areaJ = get_scale_base(imgJ)
    scale = areaI / areaJ
    # print(areaI, areaJ, scale)
    return scale

def fill_square(img, size=None):
    length = max(img.shape[:2])
    img2 = np.zeros((length, length))
    if len(img.shape) == 3:
        img2 = np.zeros((length, length, img.shape[2]))
    sizex = min(img.shape[1], length)
    sizey = min(img.shape[0], length)
    img2[:sizey, :sizex] = img[:sizey, :sizex]
    if size is not None:
        img2 = cut_img(img2, 0, 0, size, size)
    return img2

def test_he(H, img, he):
    he2 = fill_square(he)
    he2 = rotate_imgH(he2, H)
    shape2 = img.shape
    he3 = cut_img(he2, 0, 0, shape2[1], shape2[0])
    plt.imshow(he3)
    plt.show()
    return he3

import tifffile as tf
def __img_resize(img,scale_factor):
    width = int(np.floor(img.shape[1] * scale_factor))
    height = int(np.floor(img.shape[0] * scale_factor))
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

def __write_ome_tif(filename, image, channel_names, photometric_interp, metadata, subresolutions = 7):
    subresolutions = subresolutions

    fn = filename + ".ome.tif"
    with tf.TiffWriter(fn,  bigtiff=True) as tif:
        pixelsize = metadata['PhysicalSizeX']

        options = dict(
            photometric=photometric_interp,
            tile=(1024, 1024),
            dtype=image.dtype,
            compression='jpeg2000',
            resolutionunit='CENTIMETER'
        )

        print("Writing pyramid level 0")
        tif.write(
            image,
            subifds=subresolutions,
            resolution=(1e4 / pixelsize, 1e4 / pixelsize),
            metadata=metadata,
            **options
        )

        scale = 1
        for i in range(subresolutions):
            scale /= 2
            downsample = __img_resize(np.moveaxis(image,0,-1),scale)
            print("Writing pyramid level {}".format(i+1))
            tif.write(
                np.moveaxis(downsample,-1,0),
                subfiletype=1,
                resolution=(1e4 / scale / pixelsize, 1e4 / scale / pixelsize),
                **options
            )
            
def write_ome_tif(filename, outFile, subresolutions = 7):
    with tf.TiffFile(filename) as tif:
        image = tif.asarray()
        if image.shape[-1] < image.shape[0]:
            image = np.moveaxis(image,-1,0)

        if tif.is_ome:
            meta = tf.xml2dict(tif.ome_metadata)

        if tif.pages[0].photometric == 2:
            photometric_interp='rgb'
            channel_names=None

        else:
            photometric_interp='minisblack'
            channel_names=[]
            for i, element in enumerate(meta['OME']['Image']['Pixels']['Channel']):
                channel_names.append(meta['OME']['Image']['Pixels']['Channel'][i]['Name'])

        # metadata={
        #     'PhysicalSizeX': meta['OME']['Image']['Pixels']['PhysicalSizeX'],
        #     'PhysicalSizeXUnit': meta['OME']['Image']['Pixels']['PhysicalSizeXUnit'],
        #     'PhysicalSizeY': meta['OME']['Image']['Pixels']['PhysicalSizeY'],
        #     'PhysicalSizeYUnit': meta['OME']['Image']['Pixels']['PhysicalSizeYUnit'],
        #     'Channel': {'Name': channel_names}
        #     }
        metadata={
            'PhysicalSizeX': 0.88,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': 0.88,
            'PhysicalSizeYUnit': 'µm',
            'Channel': {'Name': channel_names}
        }

        __write_ome_tif(outFile, image, channel_names, 
                        photometric_interp, metadata, 
                        subresolutions=subresolutions)


from PIL import Image

def create_pyramid_tiff_with_tiles(img: np.array, output, layer=1):
    img = Image.fromarray(img)
    layers = [img]
    
    for _ in range(layer):
        next_width = layers[-1].width // 2
        next_height = layers[-1].height // 2
        
        # 创建下一个层次并添加到列表中
        next_layer = layers[-1].resize((next_width, next_height))
        layers.append(next_layer)
    
    layers[0].save(output, format='TIFF', 
                   save_all=True, 
                   append_images=layers[1:], 
                   compression='tiff_deflate', tile=(1024, 1024))
    

def xenium_save_alignment_metric(H: np.array, path):
    with open(path, "w") as f:
        f.write("%f,%f,%f\n" % (H[0, 0], H[0, 1], H[0, 2]))
        f.write("%f,%f,%f\n" % (H[1, 0], H[1, 1], H[1, 2]))
        f.write("0,0,1")
        
def multiply_HH(Hs):
    h0 = np.eye(3)
    for h in Hs:
        h0 = np.dot(h, h0)
    return h0

def multiply_HMH(Hs):
    m0 = np.eye(3)
    for h in Hs:
        m = convert_H_to_M(h)
        m0 = np.multiply(m0, m)
    h0 = convert_M_to_H(m0)
    return h0
        
def xenium_generate_alignemnt_H(genH: np.array, 
                                rawImgShape: np.array, 
                                sampleShape: np.array, 
                                coordsBase: int=4000):
    shape = np.max(rawImgShape[:2])
    bshape =np.max(sampleShape[:2])
    
    scale2 = shape / bshape
    
    def H2M(H):
        M = np.eye(3)
        M[:2] = np.array([[H[1, 1], H[1, 0], H[1, 2]], 
                          [H[0, 1], H[0, 0], H[0, 2]]])
        return M
    
    H00 = np.eye(3)
    H00[0, 0] = -1
    H00[0, 2] = rawImgShape[1]

    H10 = np.eye(3) / scale2
    H10[2, 2] = 1

    HH = genH.copy()
    HH = H2M(HH)

    H11 = np.eye(3) * scale2
    H11[2, 2] = 1
    
    scale3 = bshape / coordsBase * bshape / shape * 2
    H2 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    H2 *= scale3

    HH = multiply_HH([H00, H10, HH, H11, H2])
    return HH

def split_he_background_otsu(img_):
    img = img_.copy()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.max() < 2.0:
        img = img * 255
    img = img.astype(np.uint8)
    thresh, binarized_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    image_ = img_.copy()
    image_[binarized_image > 0] = 0
    return binarized_image, image_

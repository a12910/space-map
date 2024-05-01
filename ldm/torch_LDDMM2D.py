import torch
import numpy as np
import scipy.linalg
import time
import sys
import os
import distutils.version
import nibabel as nib
import spacemap
from spacemap import base

grid_sample = base.grid_sample
irfft = base.irfft
rfft = base.rfft
mygaussian_torch_selectcenter_meshgrid = base.mygaussian_torch_selectcenter_meshgrid
mygaussian_3d_torch_selectcenter_meshgrid = base.mygaussian_3d_torch_selectcenter_meshgrid
mygaussian = base.mygaussian

def get_init2D(imgI, imgJ, gpu=None, verbose=100):
    """ img: J -> I """
    if gpu is None and spacemap.DEVICE != "cpu":
        gpu = spacemap.DEVICE
    ldm = LDDMM2D(template=imgJ,target=imgI,
                              do_affine=1,do_lddmm=0,
                              nt=3,
                              optimizer='adam',
                              sigma=20.0,sigmaR=40.0,
                              gpu_number=gpu,
                              target_err=0.1,
                              verbose=verbose,
                              target_step=20000,
                              show_init=False)
    return ldm
    
class LDDMM2D(base.LDDMMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    # helper function to load images
    def _load(self, template, target, costmask):
        if isinstance(template, str):
            I = [None]
            Ispacing = [None]
            Isize = [None]
            I[0],Ispacing[0],Isize[0] = self.loadImage(template,im_norm_ms=self.params['im_norm_ms'])
        elif isinstance(template, np.ndarray):
            I = [None]
            Ispacing = [None]
            Isize = [None]
            if self.params['im_norm_ms'] == 1:
                I[0] = torch.tensor((template - np.mean(template)) / np.std(template)).type(self.params['dtype']).to(device=self.params['cuda'])
            else:
                I[0] = torch.tensor(template).type(self.params['dtype']).to(device=self.params['cuda'])
            
            Isize[0] = list(template.shape)
            if self.params['dx'] == None:
                Ispacing[0] = np.ones((3,)).astype(np.float32)
            else:
                Ispacing[0] = self.params['dx']
        elif isinstance(template, list):
            if isinstance(template[0],str):
                I = [None]*len(template)
                Ispacing = [None]*len(template)
                Isize = [None]*len(template)
                for i in range(len(template)):
                    I[i],Ispacing[i],Isize[i] = self.loadImage(template[i],im_norm_ms=self.params['im_norm_ms'])
            # assumes images are the same spacing
            elif isinstance(template[0],np.ndarray):
                I = [None]*len(template)
                Ispacing = [None]*len(template)
                Isize = [None]*len(template)
                for i in range(len(template)):
                    if self.params['im_norm_ms'] == 1:
                        I[i] = torch.tensor((template[i] - np.mean(template[i])) / np.std(template[i])).type(self.params['dtype']).to(device=self.params['cuda'])
                    else:
                        I[i] = torch.tensor(template[i]).type(self.params['dtype']).to(device=self.params['cuda'])
                    
                    Isize[i] = template[i].shape
                    if self.params['dx'] == None:
                        Ispacing[i] = np.ones((3,)).astype(np.float32)
                    else:
                        Ispacing[i] = self.params['dx']
            else:
                print('ERROR: received list of unhandled type for template image.')
                return -1
        
        if isinstance(target, str):
            J = [None]
            Jspacing = [None]
            Jsize = [None]
            J[0],Jspacing[0],Jsize[0] = self.loadImage(target,im_norm_ms=self.params['im_norm_ms'])
        elif isinstance(target, np.ndarray):
            J = [None]
            Jspacing = [None]
            Jsize = [None]
            if self.params['im_norm_ms'] == 1:
                J[0] = torch.tensor((target - np.mean(target)) / np.std(target)).type(self.params['dtype']).to(device=self.params['cuda'])
            else:
                J[0] = torch.tensor(target).type(self.params['dtype']).to(device=self.params['cuda'])
            
            Jsize[0] = list(target.shape)
            if self.params['dx'] == None:
                Jspacing[0] = np.ones((3,)).astype(np.float32)
            else:
                Jspacing[0] = self.params['dx']
        elif isinstance(target, list):
            if isinstance(target[0],str):
                J = [None]*len(target)
                Jspacing = [None]*len(target)
                Jsize = [None]*len(target)
                for i in range(len(target)):
                    J[i],Jspacing[i],Jsize[i] = self.loadImage(target[i],im_norm_ms=self.params['im_norm_ms'])
            # assumes images are the same spacing
            elif isinstance(target[0],np.ndarray):
                J = [None]*len(target)
                Jspacing = [None]*len(target)
                Jsize = [None]*len(target)
                for i in range(len(target)):
                    if self.params['im_norm_ms'] == 1:
                        J[i] = torch.tensor((target[i] - np.mean(target[i])) / np.std(target[i])).type(self.params['dtype']).to(device=self.params['cuda'])
                    else:
                        J[i] = torch.tensor(target[i]).type(self.params['dtype']).to(device=self.params['cuda'])
                    
                    Jsize[i] = target[i].shape
                    if self.params['dx'] == None:
                        Jspacing[i] = np.ones((3,)).astype(np.float32)
                    else:
                        Jspacing[i] = self.params['dx']
            else:
                print('ERROR: received list of unhandled type for target image.')
                return -1
        
        # load costmask if the variable exists
        # TODO: make this multichannel
        if isinstance(costmask, str):
            K = [None]
            Kspacing = [None]
            Ksize = [None]
            # never normalize cost mask
            K[0],Kspacing[0],Ksize[0] = self.loadImage(costmask,im_norm_ms=0)
        elif isinstance(costmask,np.ndarray):
            K = [None]
            Kspacing = [None]
            Ksize = [None]
            K[0] = torch.tensor(costmask).type(self.params['dtype']).to(device=self.params['cuda'])
            Ksize[0] = costmask.shape
            if self.params['dx'] == None:
                Kspacing[0] = np.ones((3,)).astype(np.float32)
            else:
                Kspacing[0] = self.params['dx']
        else:
            K = []
            Kspacing = []
            Ksize = []
        
        if len(J) != len(I):
            print('ERROR: images must have the same number of channels.')
            return -1
            
        #if I.shape[0] != J.shape[0] or I.shape[1] != J.shape[1] or I.shape[2] != J.shape[2]:
        #if I.shape != J.shape:
        if not all([x.shape == I[0].shape for x in I+J+K]):
            print('ERROR: the image sizes are not the same.\n')
            return -1
        #elif Ispacing[0] != Jspacing[0] or Ispacing[1] != Jspacing[1] or Ispacing[2] != Jspacing[2]
        #elif np.sum(Ispacing==Jspacing) < len(I.shape):
        elif self.params['dx'] is None and not all([list(x == Ispacing[0]) for x in Ispacing+Jspacing+Kspacing]):
            print('ERROR: the image pixel spacings are not the same.\n')
            return -1
        else:
            self.I = I
            self.J = J
            if costmask is not None:
                self.M = K[0]
            else:
                self.M = torch.tensor(np.ones(I[0].shape)).type(self.params['dtype']).to(device=self.params['cuda']) # this could be initialized to a scalar 1.0 to save memory, if you do this make sure you check computeLinearContrastCorrection to see whether or not you are using torch.sum(w*self.M)
            
            self.dx = list(Ispacing[0])
            self.dx = [float(x) for x in self.dx]
            self.nx = I[0].shape
            return 1
    
    # initialize lddmm kernels
    def initializeKernels2d(self):
        # make smoothing kernel on CPU
        f0 = np.arange(self.nx[0])/(self.dx[0]*self.nx[0])
        f1 = np.arange(self.nx[1])/(self.dx[1]*self.nx[1])
        F0,F1 = np.meshgrid(f0,f1,indexing='ij')
        #a = 3.0*self.dx[0] # a scale in mm
        #p = 2
        self.Ahat = (1.0 - 2.0*(self.params['a']*self.dx[0])**2*((np.cos(2.0*np.pi*self.dx[0]*F0) - 1.0)/self.dx[0]**2 
                                + (np.cos(2.0*np.pi*self.dx[1]*F1) - 1.0)/self.dx[1]**2))**(2.0*self.params['p'])
        self.Khat = 1.0/self.Ahat
        # only move one kernel for now
        # TODO: try broadcasting this instead
        if distutils.version.LooseVersion(torch.__version__) < distutils.version.LooseVersion("1.8.0"): # this is because pytorch fft functions have changed in input and output after 1.8. No longer outputs a two-channel matrix
            self.Khat = torch.tensor(np.tile(np.reshape(self.Khat,(self.Khat.shape[0],self.Khat.shape[1],1)),(1,1,2))).type(self.params['dtype']).to(device=self.params['cuda'])
        else:
            self.Khat = torch.tensor(np.reshape(self.Khat,(self.Khat.shape[0],self.Khat.shape[1]))).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # optimization multipliers (putting this in here because I want to reset this if I change the smoothing kernel)
        self.GDBeta = torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda'])
        #self.GDBetaAffineR = float(1.0)
        #self.GDBetaAffineT = float(1.0)
        self.climbcount = 0
        if self.params['savebestv']:
            self.best = {}
    
    
    # initialize lddmm variables
    def initializeVariables2d(self):
        # TODO: handle 2D and 3D versions
        # helper variables
        self.dt = 1.0/self.params['nt']
        # loss values
        if not hasattr(self,'EMAll'):
            self.EMAll = []
        if not hasattr(self,'ERAll'):
            self.ERAll = []
        if not hasattr(self,'EAll'):
            self.EAll = []
        if self.params['checkaffinestep'] == 1:
            if not hasattr(self,'EMAffineR'):
                self.EMAffineR = []
            if not hasattr(self,'EMAffineT'):
                self.EMAffineT = []
            if not hasattr(self,'EMDiffeo'):
                self.EMDiffeo = []
        
        # load a gaussian filter if v_scale is less than 1
        if self.params['v_scale'] < 1.0:
            size = int(np.ceil(1.0/self.params['v_scale']*5))
            if np.mod(size,2) == 0:
                size += 1
            
            self.gaussian_filter = torch.tensor(base.mygaussian(sigma=1.0/self.params['v_scale'],size=size)).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # image sampling domain
        x0 = np.arange(self.nx[0])*self.dx[0]
        x1 = np.arange(self.nx[1])*self.dx[1]
        X0,X1 = np.meshgrid(x0,x1,indexing='ij')
        self.X0 = torch.tensor(X0-np.mean(X0)).type(self.params['dtype']).to(device=self.params['cuda'])
        self.X1 = torch.tensor(X1-np.mean(X1)).type(self.params['dtype']).to(device=self.params['cuda'])
        '''
        # v and I
        if self.params['gpu_number'] is not None:
            self.vt0 = []
            self.vt1 = []
            self.detjac = []
            self.It = [ [None]*(self.params['nt']+1) for i in range(len(self.I)) ]
            for ii in range(len(self.I)):
                for i in range(self.params['nt']+1):
                    self.It[ii][i] = torch.tensor(self.I[ii][:,:]).type(self.params['dtype']).cuda()
            
            for i in range(self.params['nt']):
                self.vt0.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']).cuda())
                self.vt1.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']).cuda())
                self.detjac.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']).cuda())
        else:
            self.vt0 = []
            self.vt1 = []
            self.detjac = []
            self.It = [ [None]*(self.params['nt']+1) for i in range(len(self.I)) ]
            for ii in range(len(self.I)):
                for i in range(self.params['nt']+1):
                    self.It[ii][i] = torch.tensor(self.I[ii][:,:]).type(self.params['dtype'])
            
            for i in range(self.params['nt']):
                self.vt0.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']))
                self.vt1.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']))
                self.detjac.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']))
        '''
        # v and I
        if self.params['gpu_number'] is not None:
            if not hasattr(self, 'vt0') and self.initializer_flags['lddmm'] == 1: # we never reset lddmm variables
                self.vt0 = []
                self.vt1 = []
                self.detjac = []
                for i in range(self.params['nt']):
                    self.vt0.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                    self.vt1.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                    self.detjac.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
            
            if (self.initializer_flags['load'] == 1 or self.initializer_flags['lddmm'] == 1) and self.params['low_memory'] < 1:
                self.It = [ [None]*(self.params['nt']+1) for i in range(len(self.I)) ]
                for ii in range(len(self.I)):
                    # NOTE: you cannot use pointers / list multiplication for cuda tensors if you want actual copies
                    #self.It.append(torch.tensor(self.I[:,:,:]).type(self.params['dtype']).cuda())
                    for i in range(self.params['nt']+1):
                        if i == 0:
                            self.It[ii][i] = self.I[ii]
                        else:
                            if isinstance(self.I[ii],torch.Tensor):
                                self.It[ii][i] = self.I[ii][:,:].clone().type(self.params['dtype']).to(device=self.params['cuda'])
                            else:
                                self.It[ii][i] = torch.tensor(self.I[ii][:,:].clone().detach()).type(self.params['dtype']).to(device=self.params['cuda'])
        else:
            if not hasattr(self,'vt0') and self.initializer_flags['lddmm'] == 1:
                self.vt0 = []
                self.vt1 = []
                self.detjac = []
                for i in range(self.params['nt']):
                    self.vt0.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']))
                    self.vt1.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']))
                    self.detjac.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']))
            
            #self.It = [[None]]*len(self.I)
            #for i in range(len(self.I)):
            #    self.It[i] = [torch.tensor(self.I[i][:,:,:]).type(self.params['dtype'])]*(self.params['nt']+1)
            if self.initializer_flags['load'] == 1 or self.initializer_flags['lddmm'] == 1:
                self.It = [ [None]*(self.params['nt']+1) for i in range(len(self.I)) ]
                for ii in range(len(self.I)):
                    # NOTE: you cannot use pointers / list multiplication for cuda tensors if you want actual copies
                    #self.It.append(torch.tensor(self.I[:,:,:]).type(self.params['dtype']).cuda())
                    # UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
#   self.It[ii][i] = torch.tensor(self.I[ii][:,:].clone().detach()).type(self.params['dtype'])
                    
                    for i in range(self.params['nt']+1):
                        self.It[ii][i] = self.I[ii][:,:].clone().detach().type(self.params['dtype'])
        
        # affine parameters
        if not hasattr(self,'affineA') and self.initializer_flags['affine'] == 1: # we never automatically reset affine variables
            self.affineA = torch.tensor(np.eye(3)).type(self.params['dtype']).to(device=self.params['cuda'])
            self.lastaffineA = torch.tensor(np.eye(3)).type(self.params['dtype']).to(device=self.params['cuda'])
            self.gradA = torch.tensor(np.zeros((3,3))).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # optimizer update variables
        self.GDBeta = torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda'])
        self.GDBetaAffineR = float(1.0)
        self.GDBetaAffineT = float(1.0)
        
        # contrast correction variables
        if not hasattr(self,'ccIbar') or self.initializer_flags['cc'] == 1: # we never reset cc variables
            self.ccIbar = []
            self.ccJbar = []
            self.ccVarI = []
            self.ccCovIJ = []
            for i in range(len(self.I)):
                self.ccIbar.append(0.0)
                self.ccJbar.append(0.0)
                self.ccVarI.append(1.0)
                self.ccCovIJ.append(1.0)
        
        # weight estimation variables
        if self.initializer_flags['we'] == 1: # if number of channels changed, reset everything
            self.W = [[] for i in range(len(self.I))]
            self.we_C = [[] for i in range(len(self.I))]
            for i in range(self.params['we']):
                if i == 0: # first index is the matching channel, the rest is artifacts
                    for ii in self.params['we_channels']: # allocate space only for the desired channels
                        self.W[ii].append(torch.tensor(0.9*np.ones((self.nx[0],self.nx[1]))).type(self.params['dtype']).to(device=self.params['cuda']))
                        self.we_C[ii].append(torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda']))
                else:
                    for ii in self.params['we_channels']:
                        self.W[ii].append(torch.tensor(0.1*np.ones((self.nx[0],self.nx[1]))).type(self.params['dtype']).to(device=self.params['cuda']))
                        self.we_C[ii].append(torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda']))
        
        # SGD mask initialization
        if self.params['optimizer'] == 'sgd':
            self.sgd_M = torch.ones(self.M.shape).type(self.params['dtype']).to(device=self.params['cuda'])
        self.sgd_maskiter = 0
        
        self.adam = {}
         # adam optimizer variables
        if self.params['optimizer'] == "adam":
            self.sgd_M = torch.ones(self.M.shape).type(self.params['dtype']).to(device=self.params['cuda'])
            self.sgd_maskiter = 0
            self.adam['m0'] = []
            self.adam['m1'] = []
            self.adam['m2'] = []
            self.adam['v0'] = []
            self.adam['v1'] = []
            self.adam['v2'] = []
            for i in range(self.params['nt']):
                self.adam['m0'].append(torch.tensor(np.zeros(
                    (int(np.round(self.nx[0]*self.params['v_scale'])),
                     int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adam['m1'].append(torch.tensor(np.zeros((
                    int(np.round(self.nx[0]*self.params['v_scale'])),
                    int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adam['m2'].append(torch.tensor(np.zeros((
                    int(np.round(self.nx[0]*self.params['v_scale'])),
                    int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adam['v0'].append(torch.tensor(np.zeros((
                    int(np.round(self.nx[0]*self.params['v_scale'])),
                    int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adam['v1'].append(torch.tensor(np.zeros((
                    int(np.round(self.nx[0]*self.params['v_scale'])),
                    int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adam['v2'].append(torch.tensor(np.zeros((
                    int(np.round(self.nx[0]*self.params['v_scale'])),
                    int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
        
        # reset initializer flags
        self.initializer_flags['load'] = 0
        self.initializer_flags['lddmm'] = 0
        self.initializer_flags['affine'] = 0
        self.initializer_flags['cc'] = 0
        self.initializer_flags['we'] = 0
        self.initializer_flags['v_scale'] = 0
    
    # helper function for torch_gradient
    def _allocateGradientDivisors(self):
        if self.J[0].dim() == 3:
            # allocate gradient divisor for custom torch gradient function
            self.grad_divisor_x = np.ones(self.I[0].shape)
            self.grad_divisor_x[1:-1,:,:] = 2
            self.grad_divisor_x = torch.tensor(self.grad_divisor_x).type(self.params['dtype']).to(device=self.params['cuda'])
            self.grad_divisor_y = np.ones(self.I[0].shape)
            self.grad_divisor_y[:,1:-1,:] = 2
            self.grad_divisor_y = torch.tensor(self.grad_divisor_y).type(self.params['dtype']).to(device=self.params['cuda'])
            self.grad_divisor_z = np.ones(self.I[0].shape)
            self.grad_divisor_z[:,:,1:-1] = 2
            self.grad_divisor_z = torch.tensor(self.grad_divisor_z).type(self.params['dtype']).to(device=self.params['cuda'])
        else:
            # allocate gradient divisor for custom torch gradient function
            self.grad_divisor_x = np.ones(self.I[0].shape)
            self.grad_divisor_x[1:-1,:] = 2
            self.grad_divisor_x = torch.tensor(self.grad_divisor_x).type(self.params['dtype']).to(device=self.params['cuda'])
            self.grad_divisor_y = np.ones(self.I[0].shape)
            self.grad_divisor_y[:,1:-1] = 2
            self.grad_divisor_y = torch.tensor(self.grad_divisor_y).type(self.params['dtype']).to(device=self.params['cuda'])
    
    # 2D replication-pad, artificial roll, subtract, single-sided difference on boundaries
    def torch_gradient2d(self,arr, dx, dy, grad_divisor_x_gpu,grad_divisor_y_gpu):
        arr = torch.squeeze(torch.nn.functional.pad(arr.unsqueeze(0).unsqueeze(0),(1,1,1,1),mode='replicate'))
        gradx = torch.cat((arr[1:,:],arr[0,:].unsqueeze(0)),dim=0) - torch.cat((arr[-1,:].unsqueeze(0),arr[:-1,:]),dim=0)
        grady = torch.cat((arr[:,1:],arr[:,0].unsqueeze(1)),dim=1) - torch.cat((arr[:,-1].unsqueeze(1),arr[:,:-1]),dim=1)
        return gradx[1:-1,1:-1]/dx/grad_divisor_x_gpu, grady[1:-1,1:-1]/dy/grad_divisor_y_gpu
    
    # compute current displacement field
    def computeThisDisplacement(self,interpmode='bilinear'):
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        phiinv2_gpu = self.X2.clone()
        # TODO: evaluate memory vs speed for precomputing Xs, Ys, Zs
        for t in range(self.params['nt']):
            # update phiinv using method of characteristics
            if self.params['do_lddmm'] == 1 or hasattr(self,'vt0'):
                phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
                phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X1-self.vt1[t]*self.dt)
                phiinv2_gpu = torch.squeeze(grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X2-self.vt2[t]*self.dt)
            
            if t == self.params['nt']-1 and (self.params['do_affine'] > 0  or (hasattr(self, 'affineA') and not torch.all(torch.eq(self.affineA,torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda']))) ) ): # run this if do_affine == 1 or affineA exists and isn't identity
                phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineVectorized(self.affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
        
        phiinv0_gpu -= self.X0
        phiinv1_gpu -= self.X1
        phiinv2_gpu -= self.X2
        return phiinv0_gpu.cpu().numpy(),phiinv1_gpu.cpu().numpy(),phiinv2_gpu.cpu().numpy()

    def generateTransFromGrid(self):
        xyrange = spacemap.XYRANGE
        xyd = spacemap.XYD
        x0, x1 = int(xyrange[0] // xyd), int(xyrange[1] // xyd)
        y0, y1 = int(xyrange[2] // xyd), int(xyrange[3] // xyd)
        points = []
        for xx in range(x0, x1):
            points += [[xx * xyd, yy * xyd] for yy in range(y0, y1)]
        results = self.applyThisTransformPoints2D(points)
        return base.generateGridFromPoints(results)
             
    def applyThisTransformPoints2D(self, points: np.array, interpmode='bilinear',dtype='torch.FloatTensor'):
        xyrange = spacemap.XYRANGE
        xyd = spacemap.XYD
        # points = convert_xy(points)
        Pt = []
        for i in range(self.params['nt']+1):
            Pt.append(np.array(points))
            # Pt.append(torch.tensor(points).type(dtype).to(device=self.params['cuda']))
        
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        # TODO: evaluate memory vs speed for precomputing Xs, Ys, Zs
        for t in range(self.params['nt']):
            # update phiinv using method of characteristics
            if self.params['do_lddmm'] == 1 or hasattr(self,'vt0'):
                phiinv0_gpu = torch.squeeze(
                    grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),
                                torch.stack(
                                    ((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,
                                     (self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),
                                padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
                phiinv1_gpu = torch.squeeze(
                    grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),
                                torch.stack(
                                    ((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,
                                    (self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),
                                    padding_mode='border')) + (self.X1-self.vt1[t]*self.dt)
            
            if t == self.params['nt']-1 and \
                (self.params['do_affine'] > 0  or \
                    (hasattr(self, 'affineA') and not \
                        torch.all(torch.eq(self.affineA,torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda']))) ) ): 
                # run this if do_affine == 1 or affineA exists and isn't identity
                phiinv0_gpu,phiinv1_gpu = self.forwardDeformationAffineVectorized2d(self.affineA,phiinv0_gpu,phiinv1_gpu)
                
            interpolate = torch.nn.functional.interpolate
            # deform the image
            
            if self.params['v_scale'] != 1.0:
                ph1 = torch.stack((
                    torch.squeeze(interpolate(phiinv1_gpu.unsqueeze(0).unsqueeze(0),
                                                size=(self.nx[0],self.nx[1],self.nx[2]),
                                                mode='trilinear',
                                                align_corners=True))
                    .type(dtype)
                    .to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,
                    torch.squeeze(interpolate(phiinv0_gpu.unsqueeze(0).unsqueeze(0),
                                                size=(self.nx[0],self.nx[1],self.nx[2]),
                                                mode='trilinear',
                                                align_corners=True))
                    .type(dtype)
                    .to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0)
                ph1 = ph1.clone().detach().cpu().numpy()
                Pt[t+1] = base.grid_sample_points(Pt[0], ph1, mode=interpmode, xymax=[xyrange[1], xyrange[3]], xyd=xyd)
                # It[t+1] = torch.squeeze(grid_sample(It[0].unsqueeze(0).unsqueeze(0),ph1,padding_mode='zeros',mode=interpmode))
            else:
                ph1 = torch.stack((
                                phiinv1_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,
                                phiinv0_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2))
                ph1 = ph1.clone().detach().cpu().numpy()
                Pt[t+1] = base.grid_sample_points(Pt[0], ph1, mode=interpmode, xymax=[xyrange[1], xyrange[3]], xyd=xyd)
                # It[t+1] = torch.squeeze(grid_sample(It[0].unsqueeze(0).unsqueeze(0),ph1,dim=3).unsqueeze(0),padding_mode='zeros',mode=interpmode))
        
        # return Pt,phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
        res = Pt[-1]
        # return convert_xy(res)
        res = np.array(res)
        res -= xyd / 2
        return res
    
    # apply current transform to new image
    def applyThisTransform2d(self, I, interpmode='bilinear',dtype='torch.FloatTensor'):
        It = []
        for i in range(self.params['nt']+1):
            It.append(torch.tensor(I).type(dtype).to(device=self.params['cuda']))
        
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        # TODO: evaluate memory vs speed for precomputing Xs, Ys, Zs
        for t in range(self.params['nt']):
            # update phiinv using method of characteristics
            if self.params['do_lddmm'] == 1 or hasattr(self,'vt0'):
                phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
                phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X1-self.vt1[t]*self.dt)
            
            if t == self.params['nt']-1 and (self.params['do_affine'] > 0  or (hasattr(self, 'affineA') and not torch.all(torch.eq(self.affineA,torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda']))) ) ): # run this if do_affine == 1 or affineA exists and isn't identity
                phiinv0_gpu,phiinv1_gpu = self.forwardDeformationAffineVectorized2d(self.affineA,phiinv0_gpu,phiinv1_gpu)
            
            # deform the image
            if self.params['v_scale'] != 1.0:
                It[t+1] = torch.squeeze(grid_sample(It[0].unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv1_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros',mode=interpmode))
            else:
                It[t+1] = torch.squeeze(grid_sample(It[0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv1_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros',mode=interpmode))
        
        return It,phiinv0_gpu, phiinv1_gpu
    
        # apply current transform to new image
    
    # apply current transform to new image
    def applyThisTransformNT(self, I, interpmode='bilinear',dtype='torch.FloatTensor',nt=None):
        It = self.applyThisTransformNT2d(I, interpmode=interpmode,dtype=dtype,nt=nt)
        return It

    # apply current transform to new image
    def applyThisTransformNT2d(self, I, interpmode='bilinear',dtype='torch.FloatTensor',nt=None):
        
        I = torch.tensor(I).type(dtype).to(device=self.params['cuda'])
        grid = self.generateTransFormGridImg(dtype=dtype,nt=nt,cpu=False)
        It = torch.squeeze(grid_sample(I.unsqueeze(0).unsqueeze(0),grid,padding_mode='zeros',mode=interpmode))
        return It.cpu().numpy()
        
        # if nt == None:
        #     nt = self.params['nt']
        
        # phiinv0_gpu = self.X0.clone()
        # phiinv1_gpu = self.X1.clone()
        # I = torch.tensor(I).type(dtype).to(device=self.params['cuda'])
        # # TODO: evaluate memory vs speed for precomputing Xs, Ys, Zs
        # for t in range(nt):
        #     # update phiinv using method of characteristics
        #     if self.params['do_lddmm'] == 1 or hasattr(self,'vt0'):
        #         phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
        #         phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X1-self.vt1[t]*self.dt)
            
        #     if t == self.params['nt']-1 and (self.params['do_affine'] > 0  or (hasattr(self, 'affineA') and not torch.all(torch.eq(self.affineA,torch.tensor(np.eye(3)).type(self.params['dtype']).to(device=self.params['cuda']))) ) ): # run this if do_affine == 1 or affineA exists and isn't identity
        #         phiinv0_gpu,phiinv1_gpu = self.forwardDeformationAffineVectorized2d(self.affineA,phiinv0_gpu,phiinv1_gpu)
        
        # # deform the image
        # # TODO: do I actually need to send phiinv to gpu here?
        # if self.params['v_scale'] != 1.0:
        #     It = torch.squeeze(grid_sample(I.unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv1_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='trilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='trilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros',mode=interpmode))
        # else:
        #     It = torch.squeeze(grid_sample(I.unsqueeze(0).unsqueeze(0),torch.stack((phiinv1_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros',mode=interpmode))
        
        # del phiinv0_gpu,phiinv1_gpu
        # return It.cpu().numpy()
    
    def generateTransFormGridImg(self, dtype='torch.FloatTensor',nt=None,cpu=True):
        if nt == None:
            nt = self.params['nt']
        
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        # TODO: evaluate memory vs speed for precomputing Xs, Ys, Zs
        for t in range(nt):
            # update phiinv using method of characteristics
            if self.params['do_lddmm'] == 1 or hasattr(self,'vt0'):
                phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
                phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X1-self.vt1[t]*self.dt)
            
            if t == self.params['nt']-1 and (self.params['do_affine'] > 0  or (hasattr(self, 'affineA') and not torch.all(torch.eq(self.affineA,torch.tensor(np.eye(3)).type(self.params['dtype']).to(device=self.params['cuda']))) ) ): # run this if do_affine == 1 or affineA exists and isn't identity
                phiinv0_gpu,phiinv1_gpu = self.forwardDeformationAffineVectorized2d(self.affineA,phiinv0_gpu,phiinv1_gpu)
        if self.params['v_scale'] != 1.0:
            grid = torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv1_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='trilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='trilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0)
        else:
            grid = torch.stack((phiinv1_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0)        
        del phiinv0_gpu,phiinv1_gpu
        if cpu:
            return grid.cpu().numpy()
        return grid
    
    # deform template forward
    def forwardDeformation2d(self):
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        for t in range(self.params['nt']):
            # update phiinv using method of characteristics
            if self.params['do_lddmm'] == 1 or hasattr(self, 'vt0'):
                phiinv0_gpu = torch.squeeze(
                    grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(
                        ((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,
                         (self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),
                        padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
                phiinv1_gpu = torch.squeeze(
                    grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),
                                torch.stack(
                                    ((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,
                                     (self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),
                                padding_mode='border')) + (self.X1-self.vt1[t]*self.dt) 
            
            # do affine transforms
            if t == self.params['nt']-1 and \
                (self.params['do_affine'] > 0 or \
                    (hasattr(self, 'affineA') and not \
                        torch.all(torch.eq(self.affineA,torch.tensor(np.eye(3)).type(self.params['dtype']).to(device=self.params['cuda']))) ) ): # run this if do_affine == 1 or affineA exists and isn't identity
                if self.params['checkaffinestep'] == 1:
                    # new diffeo with old affine
                    # this doesn't match up with EAll even when vt is identity
                    phiinv0_temp,phiinv1_temp = self.forwardDeformationAffineVectorized2d(self.lastaffineA.clone(),phiinv0_gpu,phiinv1_gpu)
                    I = [None]*len(self.I)
                    for i in range(len(self.I)):
                        if self.params['v_scale'] != 1.0:
                            I[i] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv1_temp.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True))/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_temp.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))
                        else:
                            I[i] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv1_temp/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_temp/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))
                    
                    self.EMDiffeo.append( self.calculateMatchingEnergyMSEOnly2d(I) )
                    # new diffeo with new L and old T
                    phiinv0_gpu,phiinv1_gpu = self.forwardDeformationAffineR2d(self.affineA.clone(),phiinv0_gpu,phiinv1_gpu)
                    phiinv0_temp,phiinv1_temp = self.forwardDeformationAffineT2d(self.lastaffineA.clone(),phiinv0_gpu,phiinv1_gpu)
                    I = [None]*len(self.I)
                    for i in range(len(self.I)):
                        if self.params['v_scale'] != 1.0:
                            I[i] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv1_temp.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True))/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_temp.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))
                        else:
                            I[i] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv1_temp/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_temp/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))
                    
                    self.EMAffineR.append( self.calculateMatchingEnergyMSEOnly2d(I) )
                    # new everything
                    phiinv0_gpu,phiinv1_gpu = self.forwardDeformationAffineT2d(self.affineA.clone(),phiinv0_gpu,phiinv1_gpu)
                    del phiinv0_temp,phiinv1_temp,phiinv2_temp
                else:
                    phiinv0_gpu,phiinv1_gpu = self.forwardDeformationAffineVectorized2d(self.affineA.clone(),phiinv0_gpu,phiinv1_gpu)
            
            # deform the image
            for i in range(len(self.I)):
                if self.params['v_scale'] != 1.0:
                    self.It[i][t+1] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv1_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True))/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))
                else:
                    self.It[i][t+1] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))
        
        del phiinv0_gpu, phiinv1_gpu
    
    # deform template forward using affine transform
    # this could be vectorized by stacking X0, X1, X2
    def forwardDeformationAffine(self,affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu):
        affineB = torch.inverse(affineA)
        
        #Xs = affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3]
        #Ys = affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3]
        #Zs = affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3]
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2,(affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2,(affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3])
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2,(affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2,(affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3])
        phiinv2_gpu = torch.squeeze(grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2,(affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2,(affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3])
        return phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
    
    # deform template forward using affine transform vectorized
    def forwardDeformationAffineVectorized2d(self,affineA,phiinv0_gpu,phiinv1_gpu,interpmode='bilinear'):
        #affineA = affineA[[1,0,2,3],:]
        #affineA = affineA[:,[1,0,2,3]]
        affineB = torch.inverse(affineA)
        #Xs = affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]
        #Ys = affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]
        s = torch.mm(affineB[0:2,0:2],torch.stack( (torch.reshape(self.X0,(-1,)),torch.reshape(self.X1,(-1,))), dim=0)) + torch.reshape(affineB[0:2,2],(2,1)).expand(-1,self.X0.numel())
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border',mode=interpmode)) + (torch.reshape(s[0,:],(self.X0.shape)))
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border',mode=interpmode)) + (torch.reshape(s[1,:],(self.X1.shape)))
        del s
        return phiinv0_gpu, phiinv1_gpu
    
    # deform template forward using affine without translation
    def forwardDeformationAffineR2d(self,affineA,phiinv0_gpu,phiinv1_gpu):
        affineB = torch.inverse(affineA)
        #Xs = affineB[0,0]*self.X0 + affineB[0,1]*self.X1
        #Ys = affineB[1,0]*self.X0 + affineB[1,1]*self.X1
        s = torch.mm(affineB[0:2,0:2],torch.stack( (torch.reshape(self.X0,(-1,)),torch.reshape(self.X1,(-1,))), dim=0))
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[0,:],(self.X0.shape)))
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[1,:],(self.X1.shape)))
        del s
        return phiinv0_gpu, phiinv1_gpu
    
    # deform template forward using affine translation
    def forwardDeformationAffineT2d(self,affineA,phiinv0_gpu,phiinv1_gpu):
        affineB = torch.inverse(affineA)
        s = torch.stack( (torch.reshape(self.X0,(-1,)),torch.reshape(self.X1,(-1,))), dim=0) + torch.reshape(affineB[0:2,2],(2,1)).expand(-1,self.X0.numel())
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[0,:],(self.X0.shape)))
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[1,:],(self.X1.shape)))
        del s
        return phiinv0_gpu, phiinv1_gpu
    
    # compute contrast correction values
    # NOTE: does not subsample image for SGD for now
    def computeLinearContrastTransform(self,I,J,weight=1.0):
        Ibar = torch.sum(I*weight*self.M)/torch.sum(weight*self.M)
        Jbar = torch.sum(J*weight*self.M)/torch.sum(weight*self.M)
        VarI = torch.sum(((I-Ibar)*weight*self.M)**2)/torch.sum(weight*self.M)
        CovIJ = torch.sum((I-Ibar)*(J-Jbar)*weight*self.M)/torch.sum(weight*self.M)
        return Ibar, Jbar, VarI, CovIJ
    
    # contrast correction convenience function
    def runContrastCorrection(self):
        for i in self.params['cc_channels']:
            if i in self.params['we_channels'] and self.params['we'] != 0:
                if self.params['low_memory'] == 0:
                    self.ccIbar[i],self.ccJbar[i],self.ccVarI[i],self.ccCovIJ[i] = self.computeLinearContrastTransform(self.It[i][-1], self.J[i],self.W[i][0])
                else:
                    self.ccIbar[i],self.ccJbar[i],self.ccVarI[i],self.ccCovIJ[i] = self.computeLinearContrastTransform(self.applyThisTransformNT(self.I[i],nt=self.params['nt']), self.J[i],self.W[i][0])
            else:
                if self.params['low_memory'] == 0:
                    self.ccIbar[i],self.ccJbar[i],self.ccVarI[i],self.ccCovIJ[i] = self.computeLinearContrastTransform(self.It[i][-1], self.J[i])
                else:
                    self.ccIbar[i],self.ccJbar[i],self.ccVarI[i],self.ccCovIJ[i] = self.computeLinearContrastTransform(self.applyThisTransformNT(self.I[i],nt=self.params['nt']), self.J[i])
        
        return
    
    def applyContrastCorrection(self,I,i):
        #return [ ((x - self.ccIbar[i])*self.ccCovIJ[i]/self.ccVarI[i] + self.ccJbar[i]) for i,x in enumerate(I)]
        return ((I - self.ccIbar[i])*self.ccCovIJ[i]/self.ccVarI[i] + self.ccJbar[i])
    
    # compute weight estimation
    def computeWeightEstimation(self):
        for ii in range(self.params['we']):
            for i in range(len(self.I)):
                if i in self.params['we_channels']:
                    if ii == 0:
                        if self.params['low_memory'] == 0:
                            self.W[i][ii] = 1.0/np.sqrt(2.0*np.pi*self.params['sigma'][i]**2) * torch.exp(-1.0/2.0/self.params['sigma'][i]**2 * (self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])**2)
                        else:
                            self.W[i][ii] = 1.0/np.sqrt(2.0*np.pi*self.params['sigma'][i]**2) * torch.exp(-1.0/2.0/self.params['sigma'][i]**2 * (self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=self.params['nt']),i) - self.J[i])**2)
                    else:
                        self.W[i][ii] = 1.0/np.sqrt(2.0*np.pi*self.params['sigmaW'][ii]**2) * torch.exp(-1.0/2.0/self.params['sigmaW'][ii]**2 * (self.we_C[i][ii] - self.J[i])**2)
        
        for i in range(len(self.I)):
            if self.J[0].dim() == 3:
                Wsum = torch.sum(torch.stack(self.W[i],3),3)
            elif self.J[0].dim() == 2:
                Wsum = torch.sum(torch.stack(self.W[i],2),2)
            
            for ii in range(self.params['we']):
                self.W[i][ii] = self.W[i][ii] / Wsum
        
        del Wsum
        return
    
    # update weight estimation constants
    def updateWeightEstimationConstants(self):
        for i in range(len(self.I)):
            if i in self.params['we_channels']:
                for ii in range(self.params['we']):
                    self.we_C[i][ii] = torch.sum(self.W[i][ii] * self.J[i]) / torch.sum(self.W[i][ii])
        
        return
    
    # compute regularization energy for time varying velocity field in for loop to conserve memory
    def calculateRegularizationEnergyVt2d(self):
        ER = 0.0
        for t in range(self.params['nt']):
            # rfft produces a 2 channel matrix, torch does not support complex number multiplication yet
            ER += torch.sum(self.vt0[t]*irfft(rfft(self.vt0[t],2,onesided=False)*(1.0/self.Khat),2,onesided=False) + self.vt1[t]*irfft(rfft(self.vt1[t],2,onesided=False)*(1.0/self.Khat),2,onesided=False)) * 0.5 / self.params['sigmaR']**2 * self.dx[0]*self.dx[1]*self.dt
        
        return ER
        lambda1 = [None]*len(self.I)
        EM = 0
        if self.params['we'] == 0:
            for i in range(len(self.I)):
                if self.params['low_memory'] == 0:
                    lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                    EM += torch.sum(self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                else:
                    lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                    EM += torch.sum(self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                
                if self.params['optimizer'] == 'sgd':
                    lambda1[i] *= self.sgd_M
        else:
            for i in range(len(self.I)):
                if i in self.params['we_channels']:
                    for ii in range(self.params['we']):
                        if ii == 0:
                            if self.params['low_memory'] == 0:
                                lambda1[i] = -1*self.W[i][ii]*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])/self.params['sigma'][i]**2
                                EM += torch.sum(self.W[i][ii]*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                            else:
                                lambda1[i] = -1*self.W[i][ii]*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])/self.params['sigma'][i]**2
                                EM += torch.sum(self.W[i][ii]*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                            
                            if self.params['optimizer'] == 'sgd':
                                lambda1[i] *= self.sgd_M
                        else:
                            EM += torch.sum(self.W[i][ii]*self.M*( self.we_C[i][ii] - self.J[i])**2/(2.0*self.params['sigmaW'][ii]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                else:
                    if self.params['low_memory'] == 0:
                        lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                        EM += torch.sum(self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                    else:
                        lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                        EM += torch.sum(self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                    
                    if self.params['optimizer'] == 'sgd':
                        lambda1[i] *= self.sgd_M
        
        return lambda1, EM
    
    # compute matching energy
    def calculateMatchingEnergyMSE2d(self):
        lambda1 = [None]*len(self.I)
        EM = 0
        if self.params['we'] == 0:
            for i in range(len(self.I)):
                if self.params['low_memory'] == 0:
                    lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                    EM += torch.sum(self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
                else:
                    lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                    EM += torch.sum(self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
                
                if self.params['optimizer'] == 'sgd':
                    lambda1[i] *= self.sgd_M
        else:
            for i in range(len(self.I)):
                if i in self.params['we_channels']:
                    for ii in range(self.params['we']):
                        if ii == 0:
                            if self.params['low_memory'] == 0:
                                lambda1[i] = -1*self.W[i][ii]*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])/self.params['sigma'][i]**2
                                EM += torch.sum(self.W[i][ii]*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
                            else:
                                lambda1[i] = -1*self.W[i][ii]*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])/self.params['sigma'][i]**2
                                EM += torch.sum(self.W[i][ii]*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
                            
                            if self.params['optimizer'] == 'sgd':
                                lambda1[i] *= self.sgd_M
                        else:
                            EM += torch.sum(self.W[i][ii]*self.M*( self.we_C[i][ii] - self.J[i])**2/(2.0*self.params['sigmaW'][ii]**2))*self.dx[0]*self.dx[1]
                else:
                    if self.params['low_memory'] == 0:
                        lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                        EM += torch.sum(self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
                    else:
                        lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                        EM += torch.sum(self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
                    
                    if self.params['optimizer'] == 'sgd':
                        lambda1[i] *= self.sgd_M
        
        return lambda1, EM
    
    '''
    # compute matching energy
    def calculateMatchingEnergyMSE2d(self):
        lambda1 = [None]*len(self.I)
        EM = 0
        for i in range(len(self.I)):
            lambda1[i] = -1*self.M*( ((self.It[i][-1] - self.ccIbar[i])*self.ccCovIJ[i]/self.ccVarI[i] + self.ccJbar[i]) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
            EM += torch.sum(self.M*( ((self.It[i][-1] - self.ccIbar[i])*self.ccCovIJ[i]/self.ccVarI[i] + self.ccJbar[i]) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
        return lambda1, EM
    '''
    
    # compute matching energy without lambda1
    def calculateMatchingEnergyMSEOnly(self, I):
        EM = 0
        if self.params['we'] == 0:
            for i in range(len(self.I)):
                EM += torch.sum(self.M*( self.applyContrastCorrection(I[i],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
        else:
            for i in range(len(self.I)):
                if i in self.params['we_channels']:
                    for ii in range(self.params['we']):
                        if ii == 0:
                            EM += torch.sum(self.W[i][ii]*self.M*( self.applyContrastCorrection(I[i],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                        else:
                            EM += torch.sum(self.W[i][ii]*self.M*( self.we_C[i][ii] - self.J[i])**2/(2.0*self.params['sigmaW'][ii]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                else:
                    EM += torch.sum(self.M*( self.applyContrastCorrection(I[i],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
            
        return EM
    
    # compute matching energy without lambda1
    def calculateMatchingEnergyMSEOnly2d(self, I):
        EM = 0
        if self.params['we'] == 0:
            for i in range(len(self.I)):
                EM += torch.sum(self.M*( self.applyContrastCorrection(I[i],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
        else:
            for i in range(len(self.I)):
                if i in self.params['we_channels']:
                    for ii in range(self.params['we']):
                        if ii == 0:
                            EM += torch.sum(self.W[i][ii]*self.M*( self.applyContrastCorrection(I[i],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
                        else:
                            EM += torch.sum(self.W[i][ii]*self.M*( self.we_C[i][ii] - self.J[i])**2/(2.0*self.params['sigmaW'][ii]**2))*self.dx[0]*self.dx[1]
                else:
                    EM += torch.sum(self.M*( self.applyContrastCorrection(I[i],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
            
        return EM
    
    '''
    # compute matching energy without lambda1
    def calculateMatchingEnergyMSEOnly2d(self, I):
        EM = 0
        for i in range(len(self.I)):
            EM += torch.sum(self.M*( ((I[i] - self.ccIbar[i])*self.ccCovIJ[i]/self.ccVarI[i] + self.ccJbar[i]) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
        return EM
    '''
    
    # update sgd subsampling mask
    def updateSGDMask(self):
        self.sgd_maskiter += 1
        if self.sgd_maskiter == self.params['sg_holdcount']:
            self.sgd_maskiter = 0
        else:
            return
        
        if self.params['sg_mask_mode'] == 'gauss':
            self.sgd_M = base.mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda']))
        elif self.params['sg_mask_mode'] == '2gauss':
            self.sgd_M = base.mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda'])) + base.mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda']))
        elif self.params['sg_mask_mode'] == '3gauss':
            self.sgd_M = base.mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda'])) + base.mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda'])) + base.mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda']))
        elif self.params['sg_mask_mode'] == 'rand':
            self.sgd_M = self.params['sg_rand_scale']*torch.rand((self.nx[0],self.nx[1],self.nx[2])).type(self.params['dtype']).to(device=self.params['cuda'])
        elif self.params['sg_mask_mode'] == 'binrand':
            self.sgd_M = torch.round(self.params['sg_rand_scale']*torch.rand((self.nx[0],self.nx[1],self.nx[2])).type(self.params['dtype']).to(device=self.params['cuda']))
        elif self.params['sg_mask_mode'][0:5] == 'gauss' and len(self.params['sg_mask_mode']) > 5:
            self.sgd_M = base.mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda']))
            for i in range(int(self.params['sg_mask_mode'][5:])-1):
                self.sgd_M += base.mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda']))
        elif self.params['sg_mask_mode'][0:8] == 'bingauss' and len(self.params['sg_mask_mode']) > 8:
            self.sgd_M = torch.round(base.mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda'])))
            for i in range(int(self.params['sg_mask_mode'][8:])-1):
                self.sgd_M += torch.round(base.mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda'])))
            
    
    # update learning rate for gradient descent
    def updateGDLearningRate(self):
        flag = False
        if len(self.EAll) > 1:
            if self.params['optimizer'] == 'gdr':
                if self.params['checkaffinestep'] == 0 and self.params['do_affine'] == 0:
                    # energy increased
                    if self.EAll[-1] >= self.EAll[-2] or self.EAll[-1]/self.EAll[-2] > 0.99999:
                        self.GDBeta *= 0.7
                elif self.params['checkaffinestep'] == 0 and self.params['do_affine'] > 0:
                    # energy increased
                    if self.EAll[-1] >= self.EAll[-2] or self.EAll[-1]/self.EAll[-2] > 0.99999:
                        if self.params['do_lddmm'] == 1:
                            self.GDBeta *= 0.7
                        
                        self.GDBetaAffineR *= 0.7
                        self.GDBetaAffineT *= 0.7
                elif self.params['checkaffinestep'] == 1 and self.params['do_affine'] > 0:
                    # if diffeo energy increased
                    if self.ERAll[-1] + self.EMDiffeo[-1] > self.EAll[-2]:
                        self.GDBeta *= 0.7
                    
                    if self.EMAffineR[-1] > self.EMDiffeo[-1]:
                        self.GDBetaAffineR *= 0.7
                    
                    if self.EMAffineT[-1] > self.EMAffineR[-1]:
                        self.GDBetaAffineT *= 0.7
            elif self.params['optimizer'] == 'sgd' and len(self.EAll) >= self.params['sg_climbcount']:
                climbcheck = 0
                if self.params['checkaffinestep'] == 0 and self.params['do_affine'] == 0:
                    # energy increased
                    while climbcheck < self.params['sg_climbcount']:
                        if self.EAll[-1-climbcheck] >= self.EAll[-2-climbcheck] or self.EAll[-1-climbcheck]/self.EAll[-2-climbcheck] > 0.99999:
                            climbcheck += 1
                            if climbcheck == self.params['sg_climbcount']:
                                self.GDBeta *= 0.7
                                break
                        else:
                            break
                elif self.params['checkaffinestep'] == 0 and self.params['do_affine'] > 0:
                    # energy increased
                    while climbcheck < self.params['sg_climbcount']:
                        if self.EAll[-1-climbcheck] >= self.EAll[-2-climbcheck] or self.EAll[-1-climbcheck]/self.EAll[-2-climbcheck] > 0.99999:
                            climbcheck += 1
                            if climbcheck == self.params['sg_climbcount']:
                                if self.params['do_lddmm'] == 1:
                                    self.GDBeta *= 0.7

                                self.GDBetaAffineR *= 0.7
                                self.GDBetaAffineT *= 0.7
                                break
                        else:
                            break
                elif self.params['checkaffinestep'] == 1 and self.params['do_affine'] > 0:
                    # if diffeo energy increased
                    while climbcheck < self.params['sg_climbcount']:
                        if self.ERAll[-1-climbcheck] + self.EMDiffeo[-1-climbcheck] > self.EAll[-2-climbcheck]:
                            climbcheck += 1
                            if climbcheck == self.params['sg_climbcount']:
                                self.GDBeta *= 0.7
                                break
                            else:
                                break
                        
                        if self.EMAffineR[-1-climbcheck] > self.EMDiffeo[-1-climbcheck]:
                            climbcheck += 1
                            if climbcheck == self.params['sg_climbcount']:
                                self.GDBetaAffineR *= 0.7
                                break
                            else:
                                break

                        if self.EMAffineT[-1-climbcheck] > self.EMAffineR[-1-climbcheck]:
                            climbcheck += 1
                            if climbcheck == self.params['sg_climbcount']:
                                self.GDBetaAffineT *= 0.7
                                break
                            else:
                                break
            
            elif self.params['optimizer'] == 'gdw':
                # energy increased
                if self.EAll[-1] > self.EAll[-2]:
                    self.climbcount += 1
                    if self.climbcount > self.params['maxclimbcount']:
                        flag = True
                        self.GDBeta *= 0.7
                        self.climbcount = 0
                        self.vt0 = [x.to(device=self.params['cuda']) for x in self.best['vt0']]
                        self.vt1 = [x.to(device=self.params['cuda']) for x in self.best['vt1']]
                        if self.J[0].dim() > 2:
                            self.vt2 = [x.to(device=self.params['cuda']) for x in self.best['vt2']]
                        print('Reducing epsilon to ' + str((self.GDBeta*self.params['epsilon']).item()) + ' and resetting to last best point.')
                # energy decreased
                elif self.EAll[-1] < self.bestE:
                    self.climbcount = 0
                    self.GDBeta *= 1.04
                elif self.EAll[-1] < self.EAll[-2]:
                    self.climbcount = 0
        
        if self.params['savebestv']:
            if self.EAll[-1] < self.bestE:
                self.bestE = self.EAll[-1]
                # TODO: this may be too slow to keep doing on cpu. possibly clone on gpu and eat memory
                self.best['vt0'] = [x.cpu() for x in self.vt0]
                self.best['vt1'] = [x.cpu() for x in self.vt1]
                if self.J[0].dim() > 2:
                    self.best['vt2'] = [x.cpu() for x in self.vt2]
        
        return flag
    
    # compute gradient of affine transformation
    def calculateGradientA2d(self,affineA,lambda1,mode='affine'):
        self.gradA = torch.tensor(np.zeros((3,3))).type(self.params['dtype']).to(device=self.params['cuda'])
        affineB = torch.inverse(affineA)
        gi_x = [None]*len(self.I)
        gi_y = [None]*len(self.I)
        for i in range(len(self.I)):
            if self.params['low_memory'] == 0:
                if self.params['v_scale'] != 1.0:
                    gi_x[i],gi_y[i] = self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.It[i][-1],i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))
                else:
                    gi_x[i],gi_y[i] = self.torch_gradient2d(self.applyContrastCorrection(self.It[i][-1],i),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)
            else:
                if self.params['v_scale'] != 1.0:
                    gi_x[i],gi_y[i] = self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))
                else:
                    gi_x[i],gi_y[i] = self.torch_gradient2d(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)
            # TODO: can this be efficiently vectorized?
            for r in range(2):
                for c in range(3):
                    # allocating on the fly, not good
                    dA = torch.tensor(np.zeros((3,3))).type(self.params['dtype']).to(device=self.params['cuda'])
                    dA[r,c] = 1.0
                    AdAB = torch.mm(torch.mm(affineA,dA),affineB)
                    #AdABX = AdAB[0,0]*self.X0 + AdAB[0,1]*self.X1 + AdAB[0,2]
                    #AdABY = AdAB[1,0]*self.X0 + AdAB[1,1]*self.X1 + AdAB[1,2]
                    if i == 0:
                        if self.params['v_scale'] != 1.0:
                            self.gradA[r,c] = torch.sum( torch.squeeze(torch.nn.functional.interpolate(lambda1[i].unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)) * ( gi_x[i]*(AdAB[0,0]*(self.X0) + AdAB[0,1]*(self.X1) + AdAB[0,2]) + gi_y[i]*(AdAB[1,0]*(self.X0) + AdAB[1,1]*(self.X1) + AdAB[1,2]) ) ) * self.dx[0]*self.dx[1]
                        else:
                            self.gradA[r,c] = torch.sum( lambda1[i] * ( gi_x[i]*(AdAB[0,0]*(self.X0) + AdAB[0,1]*(self.X1) + AdAB[0,2]) + gi_y[i]*(AdAB[1,0]*(self.X0) + AdAB[1,1]*(self.X1) + AdAB[1,2]) ) ) * self.dx[0]*self.dx[1]
                    else:
                        if self.params['v_scale'] != 1.0:
                            self.gradA[r,c] += torch.sum( torch.squeeze(torch.nn.functional.interpolate(lambda1[i].unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)) * ( gi_x[i]*(AdAB[0,0]*(self.X0) + AdAB[0,1]*(self.X1) + AdAB[0,2]) + gi_y[i]*(AdAB[1,0]*(self.X0) + AdAB[1,1]*(self.X1) + AdAB[1,2]) ) ) * self.dx[0]*self.dx[1]
                        else:
                            self.gradA[r,c] += torch.sum( lambda1[i] * ( gi_x[i]*(AdAB[0,0]*(self.X0) + AdAB[0,1]*(self.X1) + AdAB[0,2]) + gi_y[i]*(AdAB[1,0]*(self.X0) + AdAB[1,1]*(self.X1) + AdAB[1,2]) ) ) * self.dx[0]*self.dx[1]
                    #self.gradA[r,c] = torch.sum( lambda1 * ( gi_y*(AdAB[0,0]*self.X1 + AdAB[0,1]*self.X0 + AdAB[0,2]*self.X2 + AdAB[0,3]) + gi_x*(AdAB[1,0]*self.X1 + AdAB[1,1]*self.X0 + AdAB[1,2]*self.X2 + AdAB[1,3]) + gi_z*(AdAB[2,0]*self.X1 + AdAB[2,1]*self.X0 + AdAB[2,2]*self.X2 + AdAB[2,3]) ) ) * self.dx[0]*self.dx[1]*self.dx[2]
        
        # if rigid
        if mode == 'rigid':
            self.gradA -= torch.transpose(self.gradA,0,1)
        
        # if similitude
        if mode == "sim":
            gradA_t = torch.transpose(self.gradA.clone(),0,1)
            gradA_t[0,0] = 0
            gradA_t[1,1] = 0
            self.gradA -= gradA_t
    
    # compute gradient per time step for time varying velocity field parameterization
    def calculateGradientVt2d(self,lambda1,t,phiinv0_gpu,phiinv1_gpu):
        # update phiinv using method of characteristics, note "+" because we are integrating backward
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X0+self.vt0[t]*self.dt)
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X1+self.vt1[t]*self.dt)
        
        # find the determinant of Jacobian
        if self.params['v_scale'] != 1:
            phiinv0_0,phiinv0_1 = self.torch_gradient2d(phiinv0_gpu,self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))
            phiinv1_0,phiinv1_1 = self.torch_gradient2d(phiinv1_gpu,self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))
        else:
            phiinv0_0,phiinv0_1 = self.torch_gradient2d(phiinv0_gpu,self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)
            phiinv1_0,phiinv1_1 = self.torch_gradient2d(phiinv1_gpu,self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)
        
        detjac = phiinv0_0*phiinv1_1 - phiinv0_1*phiinv1_0
        self.detjac[t] = detjac.clone()
        
        del phiinv0_0,phiinv0_1,phiinv1_0,phiinv1_1
        # deform phiinv back by affine transform if asked for
        # is this accumulating?
        #if self.params['do_affine'] == 1:
        #    phiinv0_gpu = self.affineA[0,0]*phiinv0_gpu + self.affineA[0,1]*phiinv1_gpu + self.affineA[0,2]*phiinv2_gpu + self.affineA[0,3]
        #    phiinv1_gpu = self.affineA[1,0]*phiinv0_gpu + self.affineA[1,1]*phiinv1_gpu + self.affineA[1,2]*phiinv2_gpu + self.affineA[1,3]
        #    phiinv2_gpu = self.affineA[2,0]*phiinv0_gpu + self.affineA[2,1]*phiinv1_gpu + self.affineA[2,2]*phiinv2_gpu + self.affineA[2,3]
        
        for i in range(len(self.I)):
            # find lambda_t
            #if not hasattr(self, 'affineA') or torch.all(torch.eq(self.affineA,torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda']))):
            if not hasattr(self, 'affineA'):
            #if self.params['do_affine'] == 0:
                if self.params['v_scale'] < 1.0:
                    if self.params['v_scale_smoothing'] == 1:
                        lambdat = torch.squeeze(grid_sample(torch.nn.functional.interpolate(torch.nn.functional.conv2d(lambda1[i].unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True), torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))*detjac
                    else:
                        lambdat = torch.squeeze(grid_sample(torch.nn.functional.interpolate(lambda1[i].unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True), torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))*detjac
                else:
                    lambdat = torch.squeeze(grid_sample(lambda1[i].unsqueeze(0).unsqueeze(0), torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))*detjac
            else:
                if self.params['v_scale'] < 1.0:
                    if self.params['v_scale_smoothing'] == 1:
                        lambdat = torch.squeeze(grid_sample(torch.nn.functional.interpolate(torch.nn.functional.conv2d(lambda1[i].unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True), torch.stack((((self.affineA[1,0]*(phiinv0_gpu)) + (self.affineA[1,1]*(phiinv1_gpu)) + self.affineA[1,2])/(self.nx[1]*self.dx[1]-self.dx[1])*2,((self.affineA[0,0]*(phiinv0_gpu)) + (self.affineA[0,1]*(phiinv1_gpu)) + self.affineA[0,2])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))*detjac*torch.abs(torch.det(self.affineA))
                    else:
                        lambdat = torch.squeeze(grid_sample(torch.nn.functional.interpolate(lambda1[i].unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True), torch.stack((((self.affineA[1,0]*(phiinv0_gpu)) + (self.affineA[1,1]*(phiinv1_gpu)) + self.affineA[1,2])/(self.nx[1]*self.dx[1]-self.dx[1])*2,((self.affineA[0,0]*(phiinv0_gpu)) + (self.affineA[0,1]*(phiinv1_gpu)) + self.affineA[0,2])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))*detjac*torch.abs(torch.det(self.affineA))
                else:
                    lambdat = torch.squeeze(grid_sample(lambda1[i].unsqueeze(0).unsqueeze(0), torch.stack((((self.affineA[1,0]*(phiinv0_gpu)) + (self.affineA[1,1]*(phiinv1_gpu)) + self.affineA[1,2])/(self.nx[1]*self.dx[1]-self.dx[1])*2,((self.affineA[0,0]*(phiinv0_gpu)) + (self.affineA[0,1]*(phiinv1_gpu)) + self.affineA[0,2])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))*detjac*torch.abs(torch.det(self.affineA))
            
            # get the gradient of the image at this time
            # is there a row column flip in matlab versus my torch_gradient function? yes, there is.
            if i == 0:
                if self.params['low_memory'] == 0:
                    if self.params['v_scale'] != 1.0:
                        if self.params['v_scale_smoothing'] == 1:
                            #TODO: alternatively, compute image gradient and then downsample after
                            grad_list = [x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(torch.nn.functional.conv2d(self.applyContrastCorrection(self.It[i][t],i).unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))]
                        else:
                            grad_list = [x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.It[i][t],i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))]
                    else:
                        grad_list = [x*lambdat for x in self.torch_gradient2d(self.applyContrastCorrection(self.It[i][t],i),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)]
                else:
                    if self.params['v_scale'] != 1.0:
                        if self.params['v_scale_smoothing'] == 1:
                            grad_list = [x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(torch.nn.functional.conv2d(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i).unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))]
                        else:
                            grad_list = [x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))]
                    else:
                        grad_list = [x*lambdat for x in self.torch_gradient2d(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)]
            else:
                if self.params['low_memory'] == 0:
                    if self.params['v_scale'] != 1.0:
                        if self.params['v_scale_smoothing'] == 1:
                            grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(torch.nn.functional.conv2d(self.applyContrastCorrection(self.It[i][t],i).unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))])]
                        else:
                            grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.It[i][t],i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))])]
                    else:
                        grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(self.applyContrastCorrection(self.It[i][t],i),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)])]
                else:
                    if self.params['v_scale'] != 1.0:
                        if self.params['v_scale_smoothing'] == 1:
                            grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(torch.nn.functional.conv2d(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i).unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))])]
                        else:
                            grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))])]
                    else:
                        grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)])]
        
        # smooth it
        del lambdat, detjac
        if self.params['low_memory'] > 0:
            torch.cuda.empty_cache()
            
        if self.params['optimizer'] != 'adam':
            grad_list = [irfft(rfft(x,2,onesided=False)*self.Khat,2,onesided=False) for x in grad_list]
            # add the regularization term
            grad_list[0] += self.vt0[t]/self.params['sigmaR']**2
            grad_list[1] += self.vt1[t]/self.params['sigmaR']**2
        
        return grad_list,phiinv0_gpu,phiinv1_gpu
        
        ## compute gradient per time step for time varying velocity field parameterization
        #def calculateGradientVt2d(self,lambda1,t,phiinv0_gpu,phiinv1_gpu):
        #    # update phiinv using method of characteristics, note "+" because we are integrating backward
        #    phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X0+self.vt0[t]*self.dt)
        #    phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X1+self.vt1[t]*self.dt)        
        #    
        #    # find the determinant of Jacobian
        #    phiinv0_0,phiinv0_1 = self.torch_gradient2d(phiinv0_gpu,self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)
        #    phiinv1_0,phiinv1_1 = self.torch_gradient2d(phiinv1_gpu,self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)
        #    detjac = phiinv0_0 * phiinv1_1 - phiinv0_1 * phiinv1_0
        #    self.detjac[t] = detjac.clone()
        #    '''
        #    # deform phiinv back by affine transform if asked for
        #    if self.params['do_affine'] == 1:
        #        phiinv0_gpu = self.affineA[0,0]*phiinv0_gpu + self.affineA[0,1]*phiinv1_gpu + self.affineA[0,2]
        #        phiinv1_gpu = self.affineA[1,0]*phiinv0_gpu + self.affineA[1,1]*phiinv1_gpu + self.affineA[1,2]
        #    '''
        #    for i in range(len(self.I)):
        #        # find lambda_t
        #        #if self.params['do_affine'] == 0:
        #        lambdat = torch.squeeze(grid_sample(lambda1[i].unsqueeze(0).unsqueeze(0), torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))*detjac
        #        #else:
        #        #    lambdat = torch.squeeze(grid_sample(lambda1.unsqueeze(0).unsqueeze(0), torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border'))*detjac*torch.det(self.affineA)
        #        # get the gradient of the image at this time
        #        # is there a row column flip in matlab versus my torch_gradient function? yes, there is.
        #        if i == 0:
        #            grad_list = [x*lambdat for x in self.torch_gradient2d(((self.It[i][t] - self.ccIbar[i])*self.ccCovIJ[i]/self.ccVarI[i] + self.ccJbar[i]),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)]
        #        else:
        #            grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(((self.It[i][t] - self.ccIbar[i])*self.ccCovIJ[i]/self.ccVarI[i] + self.ccJbar[i]),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)])]
        #    
        #    # smooth it
        #    grad_list = [torch.irfft(torch.rfft(x,2,onesided=False)*self.Khat,2,onesided=False) for x in grad_list]
        #    
        #    # add the regularization term
        #    grad_list[0] += self.vt0[t]/self.params['sigmaR']**2
        #    grad_list[1] += self.vt1[t]/self.params['sigmaR']**2
        #    return grad_list,phiinv0_gpu,phiinv1_gpu
    
    # update gradient
    def updateGradientVt(self,t,grad_list,iter=0):
        if self.params['optimizer'] == 'adam':
            #self.vt0[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * self.adam['m0'][t] / (torch.sqrt(self.adam['v0'][t]) + self.params['adam_epsilon'])
            #self.vt1[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * self.adam['m1'][t] / (torch.sqrt(self.adam['v1'][t]) + self.params['adam_epsilon'])
            #if self.J[0].dim() > 2:
            #    self.vt2[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * self.adam['m2'][t] / (torch.sqrt(self.adam['v2'][t]) + self.params['adam_epsilon'])
            #self.vt0[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * self.adam['m0'][t] / ((self.adam['v0'][t]) + self.params['adam_epsilon'])
            #self.vt1[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * self.adam['m1'][t] / ((self.adam['v1'][t]) + self.params['adam_epsilon'])
            #if self.J[0].dim() > 2:
            #    self.vt2[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * self.adam['m2'][t] / ((self.adam['v2'][t]) + self.params['adam_epsilon'])
            self.vt0[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * (irfft(rfft(self.adam['m0'][t] / (torch.sqrt(self.adam['v0'][t]) + self.params['adam_epsilon']),2,onesided=False)*self.Khat,2,onesided=False)) + self.vt0[t]/self.params['sigmaR']**2
            self.vt1[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * (irfft(rfft(self.adam['m1'][t] / (torch.sqrt(self.adam['v1'][t]) + self.params['adam_epsilon']),2,onesided=False)*self.Khat,2,onesided=False)) + self.vt1[t]/self.params['sigmaR']**2
            if self.J[0].dim() > 2:
                self.vt2[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * (irfft(rfft(self.adam['m2'][t] / (torch.sqrt(self.adam['v2'][t]) + self.params['adam_epsilon']),2,onesided=False)*self.Khat,2,onesided=False)) + self.vt2[t]/self.params['sigmaR']**2
        elif self.params['optimizer'] == "adadelta":
            self.vt0[t] -= irfft(rfft(torch.sqrt(self.adadelta['v0'][t] + self.params['ada_epsilon']) / torch.sqrt(self.adadelta['m0'][t] + self.params['ada_epsilon']) * grad_list[0],3,onesided=False)*self.Khat,3,onesided=False)
            self.vt1[t] -= irfft(rfft(torch.sqrt(self.adadelta['v1'][t] + self.params['ada_epsilon']) / torch.sqrt(self.adadelta['m1'][t] + self.params['ada_epsilon']) * grad_list[1],3,onesided=False)*self.Khat,3,onesided=False)
            if self.J[0].dim() > 2:
                self.vt2[t] -= irfft(rfft(torch.sqrt(self.adadelta['v2'][t] + self.params['ada_epsilon']) / torch.sqrt(self.adadelta['m2'][t] + self.params['ada_epsilon']) * grad_list[2],3,onesided=False)*self.Khat,3,onesided=False)
            self.adadelta['v0'][t] = self.params['ada_rho']*self.adadelta['v0'][t] + (1-self.params['ada_rho'])*(-1*irfft(rfft(torch.sqrt(self.adadelta['v0'][t] + self.params['ada_epsilon']) / torch.sqrt(self.adadelta['m0'][t] + self.params['ada_epsilon']) * grad_list[0],3,onesided=False)*self.Khat,3,onesided=False))**2
            self.adadelta['v1'][t] = self.params['ada_rho']*self.adadelta['v1'][t] + (1-self.params['ada_rho'])*(-1*irfft(rfft(torch.sqrt(self.adadelta['v1'][t] + self.params['ada_epsilon']) / torch.sqrt(self.adadelta['m1'][t] + self.params['ada_epsilon']) * grad_list[1],3,onesided=False)*self.Khat,3,onesided=False))**2
            if self.J[0].dim() > 2:
                self.adadelta['v2'][t] = self.params['ada_rho']*self.adadelta['v2'][t] + (1-self.params['ada_rho'])*(-1*irfft(rfft(torch.sqrt(self.adadelta['v2'][t] + self.params['ada_epsilon']) / torch.sqrt(self.adadelta['m2'][t] + self.params['ada_epsilon']) * grad_list[2],3,onesided=False)*self.Khat,3,onesided=False))**2
            #self.vt0[t] -= (self.adadelta['v0'][t] + self.params['ada_epsilon']) / (self.adadelta['m0'][t] + self.params['ada_epsilon']) * grad_list[0]
            #self.vt1[t] -= (self.adadelta['v1'][t] + self.params['ada_epsilon']) / (self.adadelta['m1'][t] + self.params['ada_epsilon']) * grad_list[1]
            #self.vt2[t] -= (self.adadelta['v2'][t] + self.params['ada_epsilon']) / (self.adadelta['m2'][t] + self.params['ada_epsilon']) * grad_list[2]
            #self.adadelta['v0'][t] = self.params['ada_rho']*self.adadelta['v0'][t] + (1-self.params['ada_rho'])*(-1*(self.adadelta['v0'][t] + self.params['ada_epsilon']) / (self.adadelta['m0'][t] + self.params['ada_epsilon']) * grad_list[0])**2
            #self.adadelta['v1'][t] = self.params['ada_rho']*self.adadelta['v1'][t] + (1-self.params['ada_rho'])*(-1*(self.adadelta['v1'][t] + self.params['ada_epsilon']) / (self.adadelta['m1'][t] + self.params['ada_epsilon']) * grad_list[1])**2
            #self.adadelta['v2'][t] = self.params['ada_rho']*self.adadelta['v2'][t] + (1-self.params['ada_rho'])*(-1*(self.adadelta['v2'][t] + self.params['ada_epsilon']) / (self.adadelta['m2'][t] + self.params['ada_epsilon']) * grad_list[2])**2
        elif self.params['optimizer'] == 'rmsprop':
            self.vt0[t] -= irfft(rfft(self.params['rms_alpha'] / torch.sqrt(self.rmsprop['m0'][t] + self.params['rms_epsilon']) * grad_list[0],3,onesided=False)*self.Khat,3,onesided=False) + self.vt0[t]/self.params['sigmaR']**2
            self.vt1[t] -= irfft(rfft(self.params['rms_alpha'] / torch.sqrt(self.rmsprop['m1'][t] + self.params['rms_epsilon']) * grad_list[1],3,onesided=False)*self.Khat,3,onesided=False) + self.vt1[t]/self.params['sigmaR']**2
            if self.J[0].dim() > 2:
                self.vt2[t] -= irfft(rfft(self.params['rms_alpha'] / torch.sqrt(self.rmsprop['m2'][t] + self.params['rms_epsilon']) * grad_list[2],3,onesided=False)*self.Khat,3,onesided=False) + self.vt2[t]/self.params['sigmaR']**2
        elif self.params['optimizer'] == 'sgdm':
            self.vt0[t] -= self.sgdm['m0'][t]
            self.vt1[t] -= self.sgdm['m1'][t]
            if self.J[0].dim() > 2:
                self.vt2[t] -= self.sgdm['m2'][t]
        else:
            self.vt0[t] -= self.params['epsilon']*self.GDBeta*grad_list[0]
            self.vt1[t] -= self.params['epsilon']*self.GDBeta*grad_list[1]
            if self.J[0].dim() > 2:
                self.vt2[t] -= self.params['epsilon']*self.GDBeta*grad_list[2]
    
    # update adam parameters
    def updateAdamLearningRate(self,t,grad_list):
        # don't normalize here, save normalization for the update step
        #self.adam['m0'][t] = [self.params['adam_beta1']*x + (1-self.params['adam_beta1'])*y for (x,y) in zip(self.adam['m0'],grad_list[0])]
        #self.adam['m1'][t] = [self.params['adam_beta1']*x + (1-self.params['adam_beta1'])*y for (x,y) in zip(self.adam['m1'],grad_list[1])]
        #self.adam['m2'][t] = [self.params['adam_beta1']*x + (1-self.params['adam_beta1'])*y for (x,y) in zip(self.adam['m2'],grad_list[2])]
        #self.adam['v0'][t] = [self.params['adam_beta2']*x + (1-self.params['adam_beta2'])*(y**2) for (x,y) in zip(self.adam['v0'],grad_list[0])]
        #self.adam['v1'][t] = [self.params['adam_beta2']*x + (1-self.params['adam_beta2'])*(y**2) for (x,y) in zip(self.adam['v1'],grad_list[1])]
        #self.adam['v2'][t] = [self.params['adam_beta2']*x + (1-self.params['adam_beta2'])*(y**2) for (x,y) in zip(self.adam['v2'],grad_list[2])]
        self.adam['m0'][t] = self.params['adam_beta1']*self.adam['m0'][t] + (1-self.params['adam_beta1'])*grad_list[0]
        self.adam['m1'][t] = self.params['adam_beta1']*self.adam['m1'][t] + (1-self.params['adam_beta1'])*grad_list[1]
        # self.adam['m2'][t] = self.params['adam_beta1']*self.adam['m2'][t] + (1-self.params['adam_beta1'])*grad_list[2]
        self.adam['v0'][t] = self.params['adam_beta2']*self.adam['v0'][t] + (1-self.params['adam_beta2'])*(grad_list[0]**2)
        self.adam['v1'][t] = self.params['adam_beta2']*self.adam['v1'][t] + (1-self.params['adam_beta2'])*(grad_list[1]**2)
        # self.adam['v2'][t] = self.params['adam_beta2']*self.adam['v2'][t] + (1-self.params['adam_beta2'])*(grad_list[2]**2)
    
    # update adadelta parameters
    def updateAdadeltaLearningRate(self,t,grad_list):
        # accumulate gradient
        self.adadelta['m0'][t] = self.params['ada_rho']*self.adadelta['m0'][t] + (1-self.params['ada_rho'])*grad_list[0]**2
        self.adadelta['m1'][t] = self.params['ada_rho']*self.adadelta['m1'][t] + (1-self.params['ada_rho'])*grad_list[1]**2
        # self.adadelta['m2'][t] = self.params['ada_rho']*self.adadelta['m2'][t] + (1-self.params['ada_rho'])*grad_list[2]**2
        
        # accumulate parameter update
        #self.adadelta['v0'][t] = self.params['ada_rho']*self.adadelta['v0'][t] + (1-self.params['ada_rho'])*self.vt0[t]**2
        #self.adadelta['v1'][t] = self.params['ada_rho']*self.adadelta['v1'][t] + (1-self.params['ada_rho'])*self.vt1[t]**2
        #self.adadelta['v2'][t] = self.params['ada_rho']*self.adadelta['v2'][t] + (1-self.params['ada_rho'])*self.vt2[t]**2
        # this is the update step
        #-1 * torch.sqrt(self.adadelta['v0'][t] + self.params['ada_epsilon']) / torch.sqrt(self.adadelta['m0'][t] + self.params['ada_epsilon']) * grad_list[0]
    
    # update rmsprop parameters
    def updateRMSPropLearningRate(self,t,grad_list):
        self.rmsprop['m0'][t] = self.params['rms_rho']*self.rmsprop['m0'][t] + (1-self.params['rms_rho'])*grad_list[0]**2
        self.rmsprop['m1'][t] = self.params['rms_rho']*self.rmsprop['m1'][t] + (1-self.params['rms_rho'])*grad_list[1]**2
        # self.rmsprop['m2'][t] = self.params['rms_rho']*self.rmsprop['m2'][t] + (1-self.params['rms_rho'])*grad_list[2]**2
    
    # update sgdm parameters
    def updateSGDMLearningRate(self,t,grad_list):
        # do I need GDBeta here?
        self.sgdm['m0'][t] = self.params['sg_gamma']*self.sgdm['m0'][t] + self.params['epsilon']*self.GDBeta*grad_list[0]
        self.sgdm['m1'][t] = self.params['sg_gamma']*self.sgdm['m1'][t] + self.params['epsilon']*self.GDBeta*grad_list[1]
        # self.sgdm['m2'][t] = self.params['sg_gamma']*self.sgdm['m2'][t] + self.params['epsilon']*self.GDBeta*grad_list[2]
    
    # convenience function for calculating and updating gradients of Vt
    def calculateAndUpdateGradientsVt(self, lambda1, iter=0):
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        if self.J[0].dim() > 2:
            phiinv2_gpu = self.X2.clone()
        
        for t in range(self.params['nt']-1,-1,-1):
            if self.J[0].dim() > 2:
                grad_list,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.calculateGradientVt(lambda1,t,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
            else:
                grad_list,phiinv0_gpu,phiinv1_gpu = self.calculateGradientVt2d(lambda1,t,phiinv0_gpu,phiinv1_gpu)
            
            if self.params['optimizer'] == 'adam':
                self.updateAdamLearningRate(t,grad_list)
            elif self.params['optimizer'] == 'adadelta':
                self.updateAdadeltaLearningRate(t,grad_list)
            elif self.params['optimizer'] == 'rmsprop':
                self.updateRMSPropLearningRate(t,grad_list)
            elif self.params['optimizer'] == 'sgdm':
                self.updateSGDMLearningRate(t,grad_list)
            
            self.updateGradientVt(t,grad_list,iter=iter)
            del grad_list
        
        del phiinv0_gpu,phiinv1_gpu
        if self.J[0].dim() > 2:
            del phiinv2_gpu
    
    # update affine matrix
    def updateAffine(self):
        # transfer to cpu for matrix exponential, takes about 20ms round trip
        gradA_cpu_numpy = self.gradA.cpu().numpy()
        e = np.zeros((4,4))
        e[0:3,0:3] = self.params['epsilonL']*self.GDBetaAffineR
        e[0:3,3] = self.params['epsilonT']*self.GDBetaAffineT
        e = torch.tensor(scipy.linalg.expm(-e * gradA_cpu_numpy)).type(self.params['dtype']).to(device=self.params['cuda'])
        self.lastaffineA = self.affineA.clone()
        self.affineA = torch.mm(self.affineA,e)
    
    # update affine matrix
    def updateAffine2d(self):
        # transfer to cpu for matrix exponential, takes about 20ms round trip
        gradA_cpu_numpy = self.gradA.cpu().numpy()
        e = np.zeros((3,3))
        e[0:2,0:2] = self.params['epsilonL']*self.GDBetaAffineR
        e[0:2,2] = self.params['epsilonT']*self.GDBetaAffineT
        e = torch.tensor(scipy.linalg.expm(-e * gradA_cpu_numpy)).type(self.params['dtype']).to(device=self.params['cuda'])
        self.lastaffineA = self.affineA.clone()
        self.affineA = torch.mm(self.affineA,e)
    
    # update epsilon after a run
    def updateEpsilonAfterRun(self):
        self.setParams('epsilonL',self.GDBetaAffineR*self.params['epsilonL']) # reduce step size, here we set it to the current size
        self.setParams('epsilonT',self.GDBetaAffineT*self.params['epsilonT']) # reduce step size, here we set it to the current size
    
    # for section alignment
    def sa(self, input, atlas, a=None, b=None, theta=None, dx=None, nx=None, niter=250, dim=2,missingslices=[],epsilonxy=0.0000025, epsilontheta=0.00000055,minepsilonxy=float(1e-20),minepsilontheta=2.0*10**-24,sigmaxy = 0.7*10, sigmatheta = np.pi/180*2*10, sigma_atlas = 1.0, sigma_target = 1.0,sigma_target_radius = 30,min_sigma_target = 1.0,sigma_atlas_radius = 34,min_sigma_atlas = 2.,norm=0,smoothing=1,optimizer='gd',beta_1=0.9,beta_2=0.999,epsilon_adam=10e-8,alpha_adam=0.001,sg_mask_mode = 'ones',sg_sigma=25):
        # equivalent of cor_sag_rot
        if dim != 2:
            target = torch.transpose(input,dim, 2).clone()
            input = torch.transpose(input,dim, 2).clone()
            template = torch.transpose(atlas,dim, 2).clone()
            if dim == 0:
                if dx == None:
                    dx = [self.dx[i] for i in [2, 1, 0]]
                else:
                    if len(dx) != 3:
                        print('ERROR: dx is not a list of length 3.')
                        return -1
                    else:
                        dx = [dx[i] for i in [2,1,0]]
                
                if nx == None:
                    nx = [self.nx[i] for i in [2, 1, 0]]
                else:
                    if len(nx) != 3:
                        print('ERROR: nx is not a list of length 3.')
                        return -1
                    else:
                        nx = [nx[i] for i in [2, 1, 0]]
            else:
                if dx == None:
                    dx = [self.dx[i] for i in [0, 2, 1]]
                else:
                    if len(dx) != 3:
                        print('ERROR: dx is not a list of length 3.')
                        return -1
                    else:
                        dx = [dx[i] for i in [0, 2, 1]]
                
                if nx == None:
                    nx = [self.nx[i] for i in [0, 2, 1]]
                else:
                    if len(nx) != 3:
                        print('ERROR: nx is not a list of length 3.')
                        return -1
                    else:
                        nx = [nx[i] for i in [0, 2, 1]]
        else:
            target = input.clone()
            template = atlas.clone()
            nx = self.nx
            dx = self.dx
        
        # allocate coordinate grid
        x0 = np.arange(nx[0])*dx[0]
        x1 = np.arange(nx[1])*dx[1]
        x2 = np.arange(nx[2])*dx[2]
        self.meanx0 = np.mean(x0)
        self.meanx1 = np.mean(x1)
        X0,X1 = np.meshgrid(x0-self.meanx0,x1-self.meanx1,indexing='ij')
        self.saX0 = torch.tensor(X0).type(self.params['dtype']).to(device=self.params['cuda'])
        self.saX1 = torch.tensor(X1).type(self.params['dtype']).to(device=self.params['cuda'])
        X,Y,_ = np.meshgrid(x0-self.meanx0,x1-self.meanx1,x2-np.mean(x2),indexing='ij')
        X = torch.tensor(X).type(self.params['dtype']).to(device=self.params['cuda'])
        Y = torch.tensor(Y).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # allocate parameters
        if a is None:
            a = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
        
        if b is None:
            b = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
        
        if theta is None:
            theta = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # allocate stuff
        # need to account for missing slices here
        factor_vector = np.ones((nx[2]-len(missingslices),))
        factor_vector[0:sigma_target_radius] = np.linspace(min_sigma_target,1.,sigma_target_radius)
        factor_vector[-sigma_target_radius:] = np.linspace(1.,min_sigma_target,sigma_target_radius)
        sigma_target_vec = torch.tensor(np.ones((nx[2]-len(missingslices),)) * sigma_target * factor_vector).type(self.params['dtype']).to(device=self.params['cuda'])
        factor_vector = np.ones((nx[2]-len(missingslices),))
        factor_vector[0:sigma_atlas_radius] = np.linspace(min_sigma_atlas,1.,sigma_atlas_radius)
        factor_vector[-sigma_atlas_radius:] = np.linspace(1.,min_sigma_atlas,sigma_atlas_radius)
        sigma_atlas_vec = torch.tensor(np.ones((nx[2]-len(missingslices),)) * sigma_atlas * factor_vector).type(self.params['dtype']).to(device=self.params['cuda'])
        
        
        # get slice z coordinates
        #TODO: remove missing slices from the data
        slicenumbers = list(range(target.shape[2]))
        for i in range(len(missingslices)):
            slicenumbers.remove(missingslices[i])
        
        # make a version of the target with missing slices removed
        #target_rem = torch.gather(target, 2, torch.Tensor(np.array(slicenumbers).astype(np.int32)).to(device=self.params['cuda'])).clone()
        target_rem = target[:,:,slicenumbers].clone()
        
        slicecoord = torch.stack([torch.tensor((x+1.0)*dx[2] - dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda']) for x in [slicenumbers[0]-2, slicenumbers[0]-1] + slicenumbers + [slicenumbers[-1]+1, slicenumbers[-1]+2]])
        shiftindices = [-2,-1,0,1,2]
        mynumerator = torch.tensor(np.zeros((len(shiftindices), slicecoord.shape[0]))).type(self.params['dtype']).to(device=self.params['cuda'])
        dz = ((torch.roll(slicecoord[2:-2],-1) - slicecoord[2:-2]) - (torch.roll(slicecoord[2:-2],1) - slicecoord[2:-2])) / 2.0
        # account for rollover
        dz[0] = dz[1]
        dz[-1] = dz[-2]
        
        # smooth?
        if smoothing == 1:
            mygauss = torch.tensor(mygaussian(1,9)).type(self.params['dtype']).to(device=self.params['cuda'])
            for i in range(input.shape[2]):
                input[:,:,i] = torch.squeeze(torch.nn.functional.conv2d(input[:,:,i].unsqueeze(0).unsqueeze(0),mygauss.unsqueeze(0).unsqueeze(0), stride=1, padding=4))
            for i in range(template.shape[2]):
                template[:,:,i] = torch.squeeze(torch.nn.functional.conv2d(template[:,:,i].unsqueeze(0).unsqueeze(0),mygauss.unsqueeze(0).unsqueeze(0), stride=1, padding=4))
        
        # downsample?
        
        # scale the image?
        if norm==1:
            (scalelist,_) = torch.topk(torch.flatten(input),100000)
            scale = scalelist[99999]
            print(scale)
            #template = template / scale
            target = target/scale
            input = input/scale
            #template = (template - torch.min(template)) / torch.max((template - torch.min(template)))
            #template = (template - torch.min(template)) / torch.max((template - torch.min(template)))
        else:
            scale = 1.0
        
        # optimizer
        if optimizer=="adam":
            best_E = torch.tensor(1e15).type(self.params['dtype']).to(device=self.params['cuda'])
            m_a = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
            m_b = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
            m_theta = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
            v_a = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
            v_b = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
            v_theta = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
            meshind_x = torch.linspace(0,target_rem.shape[0]-1,target_rem.shape[0]).type(self.params['dtype']).to(device=self.params['cuda'])
            meshind_y = torch.linspace(0,target_rem.shape[1]-1,target_rem.shape[1]).type(self.params['dtype']).to(device=self.params['cuda'])
            meshX,meshY = torch.meshgrid(meshind_x,meshind_y)

        
        # sgd mask
        M = torch.ones(target_rem.shape).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # loop
        iter = 1
        E = torch.tensor(1e15).type(self.params['dtype']).to(device=self.params['cuda'])
        while (iter < niter):
            # interpolate stack
            k = 0
            for i in range(nx[2]):
                if i in missingslices:
                    continue
                
                # maybe don't store TX and TY
                TX = torch.cos(theta[i])*self.saX1 + -1*torch.sin(theta[i])*self.saX0 + a[i]
                TY = torch.sin(theta[i])*self.saX1 + torch.cos(theta[i])*self.saX0 + b[i]
                #target[:,:,i] = torch.squeeze(grid_sample(torch.squeeze(torch.transpose(input.clone(),dim, 2)[:,:,i]/scale).unsqueeze(0).unsqueeze(0),torch.stack( ( (TX)/(nx[1]*dx[1]-dx[1])*2, (TY)/(nx[0]*dx[0]-dx[0])*2 ) ,dim=2).unsqueeze(0),padding_mode='border'))
                target[:,:,i] = torch.squeeze(grid_sample(torch.squeeze(input[:,:,i]).unsqueeze(0).unsqueeze(0),torch.stack( ( (TX)/(nx[1]*dx[1]-dx[1])*2, (TY)/(nx[0]*dx[0]-dx[0])*2 ) ,dim=2).unsqueeze(0),padding_mode='zeros'))
                target_rem[:,:,k] = target[:,:,i].clone()
                k += 1
            
            # centered difference approximate
            #DYTI = ( torch.cat( (target[1:,:,:], torch.zeros(1,nx[1],nx[2])), dim=0) - torch.cat( ( torch.zeros(1,nx[1],nx[2]), target[:-1,:,:]), dim=0) ) / dx[1] / 2.0
            #DXTI = ( torch.cat( (target[:,1:,:], torch.zeros(nx[1],1,nx[2])), dim=0) - torch.cat( ( torch.zeros(nx[1],1,nx[2]), target[:,:-1,:]), dim=0) ) / dx[0] / 2.0
            DXTI = (torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),-1,dims=1)[:,:,2:-2] - torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),1,dims=1)[:,:,2:-2]) / dx[0] / 2.0
            DYTI = (torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),-1,dims=0)[:,:,2:-2] - torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),1,dims=0)[:,:,2:-2]) / dx[1] / 2.0
            #DXTI[:,0,:] = target[:,1,:]
            #DXTI[:,-1,:] = -1*target[:,-2,:]
            #DYTI[0,:,:] = target[1,:,:]
            #DYTI[-1,:,:] = -1*target[-2,:,:]
            
            # 2nd derivative
            mynumerator *= 0.0 # hack to set to 0, not sure if actually faster
            for i in range(len(shiftindices)):
                myshiftind = list(shiftindices)
                myshiftind.remove(shiftindices[i])
                mynumerator[i,:] = 2.0 * torch.roll(slicecoord,myshiftind[0]) * torch.roll(slicecoord,myshiftind[1]) + 2.0 * torch.roll(slicecoord,myshiftind[0]) * torch.roll(slicecoord,myshiftind[2]) + 2.0 * torch.roll(slicecoord,myshiftind[0]) * torch.roll(slicecoord,myshiftind[3]) + 2.0 * torch.roll(slicecoord,myshiftind[1]) * torch.roll(slicecoord,myshiftind[2]) + 2.0 * torch.roll(slicecoord,myshiftind[1]) * torch.roll(slicecoord,myshiftind[3]) + 2.0 * torch.roll(slicecoord,myshiftind[2]) * torch.roll(slicecoord,myshiftind[3]) - 6.0 * slicecoord * torch.roll(slicecoord,myshiftind[0]) - 6.0 * slicecoord * torch.roll(slicecoord,myshiftind[1]) - 6.0 * slicecoord * torch.roll(slicecoord,myshiftind[2]) - 6.0 * slicecoord * torch.roll(slicecoord,myshiftind[3]) + 12.0 * slicecoord**2
            
            # this array is mirrored left-right for some reason
            # correct torch.roll(target) to account for zero padding
            LZTI = (torch.reshape( mynumerator[3,:] / ( (torch.roll(slicecoord,1) - slicecoord) * (torch.roll(slicecoord,1) - torch.roll(slicecoord,-1)) * (torch.roll(slicecoord,1) - torch.roll(slicecoord,2)) * (torch.roll(slicecoord,1) - torch.roll(slicecoord,-2)) ), [1,1,slicecoord.shape[0] ] ).expand([target_rem.shape[0], target_rem.shape[1], -1] ) * torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),1,dims=2))[:,:,2:-2] \
                + (torch.reshape( mynumerator[4,:] / ( (torch.roll(slicecoord,2) - slicecoord) * (torch.roll(slicecoord,2) - torch.roll(slicecoord,-1)) * (torch.roll(slicecoord,2) - torch.roll(slicecoord,-2)) * (torch.roll(slicecoord,2) - torch.roll(slicecoord,1)) ), [1,1,slicecoord.shape[0]]).expand( [target_rem.shape[0],target_rem.shape[1], -1] ) * torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),2,dims=2))[:,:,2:-2] \
                + (torch.reshape( mynumerator[1,:] / ( (torch.roll(slicecoord,-1) - slicecoord) * (torch.roll(slicecoord,-1) - torch.roll(slicecoord,1)) * (torch.roll(slicecoord,-1) - torch.roll(slicecoord,-2)) * (torch.roll(slicecoord,-1) - torch.roll(slicecoord,2)) ), [1,1,slicecoord.shape[0]]).expand( [target_rem.shape[0],target_rem.shape[1],-1] ) * torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),-1,dims=2))[:,:,2:-2] \
                + (torch.reshape( mynumerator[0,:] / ( (torch.roll(slicecoord,-2) - slicecoord) * (torch.roll(slicecoord,-2) - torch.roll(slicecoord,1)) * (torch.roll(slicecoord,-2) - torch.roll(slicecoord,2)) * (torch.roll(slicecoord,-2) - torch.roll(slicecoord,-1)) ), [1,1,slicecoord.shape[0]]).expand( [target_rem.shape[0],target_rem.shape[1],-1] ) * torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),-2,dims=2))[:,:,2:-2] \
                + (torch.reshape( mynumerator[2,:] / ( (slicecoord - torch.roll(slicecoord,1)) * (slicecoord - torch.roll(slicecoord,2)) * (slicecoord - torch.roll(slicecoord,-1)) * (slicecoord - torch.roll(slicecoord,-2)) ), [1,1,slicecoord.shape[0]]).expand( [target_rem.shape[0],target_rem.shape[1],-1] ) * torch.nn.functional.pad(target_rem,(2,2),'constant',0) )[:,:,2:-2]
            
            # update sgd mask
            if optimizer=="adam" and sg_mask_mode=="gauss":
                center_x = (torch.rand(1)*target_rem.shape[0]).type(self.params['dtype']).to(device=self.params['cuda'])
                center_y = (torch.rand(1)*target_rem.shape[1]).type(self.params['dtype']).to(device=self.params['cuda'])
                M = mygaussian_torch_selectcenter_meshgrid(meshX,meshY,sg_sigma,center_x,center_y)
                # normalize mask
                #M = M / (1/(2*sg_sigma**2))
                # make mask 3d
                M = M.unsqueeze(2).expand(-1,-1,target_rem.shape[2])
                #print(torch.max(M))
            elif optimizer=="adam" and sg_mask_mode=="2gauss":
                M = mygaussian_torch_selectcenter_meshgrid(meshX,meshY,sg_sigma,(torch.rand(1)*target_rem.shape[0]).type(self.params['dtype']).to(device=self.params['cuda']),(torch.rand(1)*target_rem.shape[1]).type(self.params['dtype']).to(device=self.params['cuda'])) + mygaussian_torch_selectcenter_meshgrid(meshX,meshY,sg_sigma,(torch.rand(1)*target_rem.shape[0]).type(self.params['dtype']).to(device=self.params['cuda']),(torch.rand(1)*target_rem.shape[1]).type(self.params['dtype']).to(device=self.params['cuda']))
                M = M.unsqueeze(2).expand(-1,-1,target_rem.shape[2])
            elif optimizer=="adam" and sg_mask_mode=="i2gauss":
                for i in range(M.shape[2]):
                    M[:,:,i] =  mygaussian_torch_selectcenter_meshgrid(meshX,meshY,sg_sigma,(torch.rand(1)*target_rem.shape[0]).type(self.params['dtype']).to(device=self.params['cuda']),(torch.rand(1)*target_rem.shape[1]).type(self.params['dtype']).to(device=self.params['cuda'])) + mygaussian_torch_selectcenter_meshgrid(meshX,meshY,sg_sigma,(torch.rand(1)*target_rem.shape[0]).type(self.params['dtype']).to(device=self.params['cuda']),(torch.rand(1)*target_rem.shape[1]).type(self.params['dtype']).to(device=self.params['cuda']))
            elif optimizer=="adam" and sg_mask_mode=="igauss":
                for i in range(M.shape[2]):
                    M[:,:,i] = mygaussian_torch_selectcenter_meshgrid(meshX,meshY,sg_sigma,(torch.rand(1)*target_rem.shape[0]).type(self.params['dtype']).to(device=self.params['cuda']),(torch.rand(1)*target_rem.shape[1]).type(self.params['dtype']).to(device=self.params['cuda']))
            elif optimizer=="adam" and sg_mask_mode=="rand":
                M = torch.rand(M.shape)
            
            # compute energies
            last_E = E.clone()
            Eimtarget = torch.sum(1.0 / torch.squeeze(sigma_target_vec)**2 * torch.squeeze(torch.sum(torch.sum(-LZTI * target_rem,dim=0),dim=0)) ) * np.prod(dx)/2.0
            Eimatlas = torch.sum(1.0 / sigma_atlas_vec**2 * torch.squeeze(torch.sum(torch.sum( (target_rem - template[:,:,slicenumbers])**2 , dim=0),dim=0))) * np.prod(dx)/2.0
            Eregxy = torch.sum((a[slicenumbers]**2 + b[slicenumbers]**2)*dz)/2.0/sigmaxy**2/slicecoord[2:-2].shape[0]
            Eregtheta = torch.sum((theta[slicenumbers]**2)*dz)/2.0/sigmatheta**2/slicecoord[2:-2].shape[0]
            E = Eimtarget + Eimatlas + Eregxy + Eregtheta
            
            # adjust optimization parameters
            # locked to gdr for now
            if optimizer == "gd":
                if E < last_E and iter > 1:
                    if np.mod(iter,2) == 0:
                        epsilonxy *= 1.04
                    else:
                        epsilontheta *= 1.04
                
                if E > last_E and iter > 1:
                    if np.mod(iter,2) == 0:
                        epsilonxy = epsilonxy/1.5
                        #print('reducing epsilonxy to ' + str(epsilonxy))
                        if epsilonxy < minepsilonxy and epsilontheta < minepsilontheta:
                            break
                    else:
                        epsilontheta = epsilontheta/1.5
                        #print('reducing epsilontheta to ' + str(epsilontheta))
                        if epsilontheta < minepsilontheta and epsilontheta < minepsilontheta:
                            break
            elif optimizer=="adam":
                if E < best_E:
                    best_E = E.clone()
                    best_a = a.clone()
                    best_b = b.clone()
                    best_theta = theta.clone()
            
            # print energies
            if iter > 1:
                start_time = end_time
            
            end_time = time.time()
            if iter > 1:
                print("iter: " + str(iter) + ", E= {:.3f}, Eim_t= {:.3f}, Eim_a= {:.3f}, ER_xy= {:.3f}, ER_t= {:.4f}, ep_xy= {:.4f}, ep_t= {:.4f}, time= {:.2f}.".format(E.item(),Eimtarget.item(),Eimatlas.item(),Eregxy.item(),Eregtheta.item(), epsilonxy, epsilontheta,end_time-start_time))
            else:
                print("iter: " + str(iter) + ", E= {:.3f}, Eim_t= {:.3f}, Eim_a= {:.3f}, ER_xy= {:.3f}, ER_t= {:.4f}, ep_xy= {:.4f}, ep_t= {:.4f}.".format(E.item(),Eimtarget.item(),Eimatlas.item(),Eregxy.item(),Eregtheta.item(), epsilonxy, epsilontheta))
            #print('iter: ' + str(iter) +', cost: '+ str(E) +', im_t = ' +str(Eimtarget)+ ', im_a = ' + str(Eimatlas) + ', regxy = ' +str(Eregxy)+ ', regt = ' +str(Eregtheta)+ ', ep = ' +str(epsilonxy)+ ', ' +str(epsilontheta))
            
            # compute gradients
            if np.mod(iter,2) == 1:
                gradx = 1/sigma_target_vec**2 * torch.squeeze(torch.sum(torch.sum(-2*M*LZTI*DXTI,dim=0),dim=0) * dx[0] * dx[1])
                grady = 1/sigma_target_vec**2 * torch.squeeze(torch.sum(torch.sum(-2*M*LZTI*DYTI,dim=0),dim=0) * dx[0] * dx[1])
            
            #diffatlas = torch.zeros(DXTI.shape)
            diffatlas = target_rem - template[:,:,slicenumbers]
            # set missing slices to zero
            #for i in missingslices:
            #    diffatlas[:,:,i] = torch.zeros((nx[0],nx[1],1))
            
            if np.mod(iter,2) == 1:
                gradx = gradx + 1/sigma_atlas_vec**2 * torch.squeeze(torch.sum(torch.sum(2*M*DXTI*diffatlas,dim=0),dim=0) * dx[0] * dx[1])
                grady = grady + 1/sigma_atlas_vec**2 * torch.squeeze(torch.sum(torch.sum(2*M*DYTI*diffatlas,dim=0),dim=0) * dx[0] * dx[1])
                gradx += torch.squeeze(a[slicenumbers])/sigmaxy**2
                grady += torch.squeeze(b[slicenumbers])/sigmaxy**2
            
            if np.mod(iter,2) == 0:
                gradtheta = 1/sigma_target_vec**2 * torch.squeeze(torch.sum(torch.sum( -2*M*LZTI*(DXTI*-1*X[:,:,slicenumbers] + DYTI*Y[:,:,slicenumbers]),dim=0),dim=0) * dx[0] * dx[1])
                gradtheta = gradtheta + 1/sigma_atlas_vec**2 * torch.squeeze(torch.sum(torch.sum( 2*M* (DXTI*-1*X[:,:,slicenumbers] + DYTI*Y[:,:,slicenumbers]) * diffatlas,dim=0),dim=0) * dx[0] * dx[1])
                gradtheta += torch.squeeze(theta[slicenumbers])/sigmatheta**2
            
            # update parameters
            # don't need to calculate gradtheta and gradxy every iteration
            #TODO: separate normalization into update step only
            if np.mod(iter,2) == 1:
                if optimizer=="gd":
                    a[slicenumbers] = a[slicenumbers]-epsilonxy*gradx
                    b[slicenumbers] = b[slicenumbers]-epsilonxy*grady
                elif optimizer=="adam":
                    m_a[slicenumbers] = (beta_1*m_a[slicenumbers] + (1-beta_1)*gradx) / (1-beta_1**(iter+1))
                    m_b[slicenumbers] = (beta_1*m_b[slicenumbers] + (1-beta_1)*grady) / (1-beta_1**(iter+1))
                    v_a[slicenumbers] = (beta_2*v_a[slicenumbers] + (1-beta_2)*(gradx**2)) / (1-beta_2**(iter+1))
                    v_b[slicenumbers] = (beta_2*v_b[slicenumbers] + (1-beta_2)*(grady**2)) / (1-beta_2**(iter+1))
                    a[slicenumbers] = a[slicenumbers] - alpha_adam * m_a[slicenumbers] / (torch.sqrt(v_a[slicenumbers]) + epsilon_adam)
                    b[slicenumbers] = b[slicenumbers] - alpha_adam * m_b[slicenumbers] / (torch.sqrt(v_b[slicenumbers]) + epsilon_adam)
            else:
                if optimizer=="gd":
                    theta[slicenumbers] = theta[slicenumbers]-epsilontheta*gradtheta
                elif optimizer=="adam":
                    m_theta[slicenumbers] = (beta_1*m_theta[slicenumbers] + (1-beta_1)*gradtheta) / (1-beta_1**(iter+1))
                    v_theta[slicenumbers] = (beta_2*v_theta[slicenumbers] + (1-beta_2)*(gradtheta**2)) / (1-beta_2**(iter+1))
                    theta[slicenumbers] = theta[slicenumbers] - alpha_adam * m_theta[slicenumbers] / (torch.sqrt(v_theta[slicenumbers]) + epsilon_adam)
            
            iter += 1
        
        if optimizer != "adam":
            best_a = a.clone()
            best_b = b.clone()
            best_theta = theta.clone()
        
        return best_a, best_b, best_theta, target*scale, epsilonxy, epsilontheta
    
    # apply stack alignment transform
    def applySA(self, input, a, b, theta, dx=None, nx=None, dim=2):
        if dim != 2:
            target = torch.transpose(input,dim, 2).clone()
            out = torch.transpose(input,dim, 2).clone()
            if dim == 0:
                if dx == None:
                    dx = [self.dx[i] for i in [2, 1, 0]]
                else:
                    if len(dx) != 3:
                        print('ERROR: dx is not a list of length 3.')
                        return -1
                    else:
                        dx = [dx[i] for i in [2,1,0]]
                
                if nx == None:
                    nx = [self.nx[i] for i in [2, 1, 0]]
                else:
                    if len(nx) != 3:
                        print('ERROR: nx is not a list of length 3.')
                        return -1
                    else:
                        nx = [nx[i] for i in [2, 1, 0]]
            else:
                if dx == None:
                    dx = [self.dx[i] for i in [0, 2, 1]]
                else:
                    if len(dx) != 3:
                        print('ERROR: dx is not a list of length 3.')
                        return -1
                    else:
                        dx = [dx[i] for i in [0, 2, 1]]
                
                if nx == None:
                    nx = [self.nx[i] for i in [0, 2, 1]]
                else:
                    if len(nx) != 3:
                        print('ERROR: nx is not a list of length 3.')
                        return -1
                    else:
                        nx = [nx[i] for i in [0, 2, 1]]
        else:
            target = input.clone()
            out = input.clone()
            if nx == None:
                nx = self.nx
            elif len(nx) != 3:
                print('ERROR: nx is not a list of length 3.')
                return -1
            
            if dx == None:
                dx = self.dx
            elif len(dx) != 3:
                print('ERROR: dx is not a list of length 3.')
                return -1
        
        # allocate coordinate grid
        x0 = np.arange(nx[0])*dx[0]
        x1 = np.arange(nx[1])*dx[1]
        self.meanx0 = np.mean(x0)
        self.meanx1 = np.mean(x1)
        X0,X1 = np.meshgrid(x0-self.meanx0,x1-self.meanx1,indexing='ij')
        self.saX0 = torch.tensor(X0).type(self.params['dtype']).to(device=self.params['cuda'])
        self.saX1 = torch.tensor(X1).type(self.params['dtype']).to(device=self.params['cuda'])
        
        for i in range(nx[2]):
            # maybe don't store TX and TY
            TX = torch.cos(theta[i])*self.saX1 + -1*torch.sin(theta[i])*self.saX0 + a[i]
            TY = torch.sin(theta[i])*self.saX1 + torch.cos(theta[i])*self.saX0 + b[i]
            #target[:,:,i] = torch.squeeze(grid_sample(torch.squeeze(torch.transpose(input.clone(),dim, 2)[:,:,i]/scale).unsqueeze(0).unsqueeze(0),torch.stack( ( (TX)/(nx[1]*dx[1]-dx[1])*2, (TY)/(nx[0]*dx[0]-dx[0])*2 ) ,dim=2).unsqueeze(0),padding_mode='border'))
            out[:,:,i] = torch.squeeze(grid_sample(torch.squeeze(target[:,:,i]).unsqueeze(0).unsqueeze(0),torch.stack( ( (TX)/(nx[1]*dx[1]-dx[1])*2, (TY)/(nx[0]*dx[0]-dx[0])*2 ) ,dim=2).unsqueeze(0),padding_mode='zeros'))
        
        return out
    
    # main loop
    def registration(self):
        historyErr = np.zeros(100)
        shape = len(self.params["template"])
        
        for it in range(self.params['niter']):

            if self.params['low_memory'] < 1:
                self.forwardDeformation2d()
            
            if self.params['cc'] == 1:
                self.runContrastCorrection()
            
            # update weight estimation
            if self.params['we'] > 0 and np.mod(it,self.params['nMstep']) == 0:
                self.computeWeightEstimation()
            
            if self.params['do_lddmm'] == 1:
                ER = self.calculateRegularizationEnergyVt2d()
            else:
                ER = torch.tensor(0.0).type(self.params['dtype'])
            
            if self.params['optimizer'] == 'sgd' or self.params['optimizer'] == 'adam' or self.params['optimizer'] == 'rmsprop':
                self.updateSGDMask()
            
            lambda1,EM = self.calculateMatchingEnergyMSE2d()
                
            # save variables
            E = ER+EM
            self.EMAll.append(EM)
            self.ERAll.append(ER)
            self.EAll.append(E.item())
            if self.params['checkaffinestep']:
                self.EMAffineT.append(EM)
            if it == 0 and self.params['savebestv']:
                self.bestE = E.clone()
            
            ERR = EM.item() / (shape * shape) 
            
            
            verbose = int(self.params['verbose'])
            if verbose > 0 and it % verbose == 0:
                end_time = time.time()
                total_time = end_time-self.params["last_time"]
                if it > 0:
                    if self.params['checkaffinestep'] == 1 and self.params['do_affine'] > 0:
                        print("iter: " + str(it) + ", E= {:.4f}, ER= {:.3f}, EM= {:.3f}, epd= {:.3f}, del_Ev= {:.4f}, del_El= {:.4f}, del_Et= {:.4f}, time= {:.2f}s.".format(E.item(),ER.item(),ERR,(self.GDBeta*self.params['epsilon']).item(),self.ERAll[-1] + self.EMDiffeo[-1] - self.EAll[-2], self.EMAffineR[-1] - self.EMDiffeo[-1], self.EMAffineT[-1] - self.EMAffineR[-1],total_time))
                    else:
                        print("iter: " + str(it) + ", E= {:.4f}, ER= {:.3f}, EM= {:.3f}, epd= {:.3f}, time= {:.2f}s.".format(E.item(),ER.item(),ERR,(self.GDBeta*self.params['epsilon']).item(),total_time))
                    self.params["last_time"] = time.time()
                else:
                    print("iter: " + str(it) + ", E = {:.4f}, ER = {:.4f}, EM = {:.4f}, epd = {:.6f}.".format(E.item(),ER.item(),ERR,(self.GDBeta*self.params['epsilon']).item()))
                    
            self.params["total_step"] += 1
            
            if self.params["target_step"] > 0 and self.params["target_step"] < self.params["total_step"]:
                print('Early termination: Target Step: %d reached.' % int(self.params["target_step"]))
                break
            if self.params["target_err"] > 0 and EM.item() < self.params["target_err"] and self.params["total_step"] > 500:
                print('Early termination: Target Err: %.4f reached.' % float(ERR))
                break
            # lastErr = np.max(historyErr)
            lastErr = historyErr[it % len(historyErr + 1)]
            if abs(lastErr -  ERR) < self.params["target_err_skip"] and it > len(historyErr):
                print('Early termination: Target Err Skip: %.4f reached. %.2f' % (float(ERR), self.params["target_err_skip"]))
                break
            historyErr[it % len(historyErr)] = ERR
            
            if it == self.params['niter']-1 or (
                (self.params['do_lddmm'] == 0 or self.GDBeta < self.params['minbeta']) and (
                    self.params['do_affine']==0 or (
                        self.GDBetaAffineR < self.params['minbeta'] and 
                        self.GDBetaAffineT < self.params['minbeta']))) or \
                            self.EAll[-1]/self.EAll[self.params['energy_fraction_from']] <= self.params['energy_fraction']:
                if ((self.params['do_lddmm'] == 0 or 
                     self.GDBeta < self.params['minbeta']) and 
                    (self.params['do_affine']==0 or (
                         self.GDBetaAffineR < self.params['minbeta'] and 
                         self.GDBetaAffineT < self.params['minbeta']))):
                    print('Early termination: Energy change threshold reached.')
                elif self.EAll[-1]/self.EAll[self.params['energy_fraction_from']] <= self.params['energy_fraction']:
                    print('Early termination: Minimum fraction of initial energy reached.')
                
                print('Total elapsed runtime: {:.2f} seconds.'.format(total_time))
                break
            
            del E, ER, EM
            
            # update step sizes
            if self.params['we'] == 0 or (self.params['we'] > 0 and np.mod(it,self.params['nMstep']) != 0):
                updateflag = self.updateGDLearningRate()
                # if asked for, recompute images
                if updateflag:
                    if self.params['low_memory'] < 1:
                        self.forwardDeformation2d()
                    
                    lambda1,EM = self.calculateMatchingEnergyMSE2d()
                   
            
            # calculate affine gradient
            if self.params['do_affine'] == 1:
                self.calculateGradientA2d(self.affineA,lambda1)
            elif self.params['do_affine'] == 2:
                self.calculateGradientA2d(self.affineA,lambda1,mode='rigid')
            elif self.params['do_affine'] == 3:
                self.calculateGradientA2d(self.affineA,lambda1,mode='sim')
            
            if self.params['low_memory'] > 0:
                torch.cuda.empty_cache()
            
            # calculate and update gradients
            if self.params['do_lddmm'] == 1:
                self.calculateAndUpdateGradientsVt(lambda1,iter=it)
            
            del lambda1
            
            # update affine
            if self.params['do_affine'] > 0:
                self.updateAffine2d()
            
            # update weight estimation
            if self.params['we'] > 0 and np.mod(it,self.params['nMstep']) == 0:
                #self.computeWeightEstimation()
                self.updateWeightEstimationConstants()
            
    
    # reset all transforms
    def resetTransforms(self):
        if hasattr(self, 'vt0'): # we never reset lddmm variables
            self.vt0 = []
            self.vt1 = []
            self.vt2 = []
            for i in range(self.params['nt']):
                self.vt0.append(torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.vt1.append(torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.vt2.append(torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).to(device=self.params['cuda']))
        
        if hasattr(self,'affineA'): # we never automatically reset affine variables
            self.affineA = torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda'])
            self.lastaffineA = torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda'])
            self.gradA = torch.tensor(np.zeros((4,4))).type(self.params['dtype']).to(device=self.params['cuda'])
        
        if hasattr(self,'ccIbar'): # we never reset cc variables
            self.ccIbar = []
            self.ccJbar = []
            self.ccVarI = []
            self.ccCovIJ = []
            for i in range(len(self.I)):
                self.ccIbar.append(0.0)
                self.ccJbar.append(0.0)
                self.ccVarI.append(1.0)
                self.ccCovIJ.append(1.0)
        
        # weight estimation variables
        if hasattr(self,'W'): # if number of channels changed, reset everything
            self.W = [[] for i in range(len(self.I))]
            self.we_C = [[] for i in range(len(self.I))]
            for i in range(self.params['we']):
                if i == 0: # first index is the matching channel, the rest is artifacts
                    for ii in self.params['we_channels']: # allocate space only for the desired channels
                        self.W[ii].append(torch.tensor(0.9*np.ones((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).to(device=self.params['cuda']))
                        self.we_C[ii].append(torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda']))
                else:
                    for ii in self.params['we_channels']:
                        self.W[ii].append(torch.tensor(0.1*np.ones((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).to(device=self.params['cuda']))
                        self.we_C[ii].append(torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda']))
        
        # optimizer update variables
        self.GDBeta = torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda'])
        self.GDBetaAffineR = float(1.0)
        self.GDBetaAffineT = float(1.0)
        return
    
    # delete transforms to save memory (mainly for running slice alignment using same object)
    def delete(self):
        if hasattr(self,'vt0'):
            del self.vt0
        if hasattr(self,'vt1'):
            del self.vt1
        if hasattr(self,'vt2'):    
            del self.vt2
        if hasattr(self,'affineA'):
            del self.affineA
        if hasattr(self,'gradA'):
            del self.gradA
        if hasattr(self,'It'):
            del self.It
        if hasattr(self,'W'):
            del self.W
        if hasattr(self,'we_C'):
            del self.we_C
        if hasattr(self,'X0'):
            del self.X0
        if hasattr(self,'X1'):
            del self.X1
        if hasattr(self,'X2'):
            del self.X2
        if hasattr(self,'Khat'):
            del self.Khat
        if hasattr(self,'saX0'):
            del self.saX0
        if hasattr(self,'saX1'):
            del self.saX1

        self.initializer_flags['lddmm'] = 1
        self.initializer_flags['affine'] = 1
        self.initializer_flags['cc'] = 1
        self.initializer_flags['we'] = 1
        torch.cuda.empty_cache()
        return
    
    # parse input transforms
    def parseInputVTransforms(self,vt0,vt1):
        varlist = [vt0,vt1]
        namelist = ['vt0','vt1']
        for i in range(len(varlist)):
            if varlist[i] is not None:
                if not isinstance(varlist[i],list):
                    print('ERROR: input \'' + str(namelist[i]) + '\' must be a list.')
                    return -1
                else:
                    if len(varlist[i]) != len(self.vt0):
                        print('ERROR: input \'' + str(namelist[i]) + '\' must be a list of length ' + str(len(self.vt0)) + ', length ' + str(len(varlist[i])) + ' was received.')
                        return -1
                    else:
                        for ii in range(len(varlist[i])):
                            if not isinstance(varlist[i][ii],(np.ndarray,torch.Tensor)):
                                print('ERROR: input \'' + str(namelist[i]) + '\' must be a list of numpy.ndarray or torch.Tensor.')
                                return -1
                            elif not varlist[i][ii].shape == self.vt0[ii].shape:
                                print('ERROR: input \'' + str(namelist[i]) + '\' must be a list of numpy.ndarray or torch.Tensor of shapes ' + str(list(self.vt0[ii].shape)) + ', shape ' + str(list(varlist[i][ii].shape)) + ' was received.')
                                return -1
                
                if i == 0:
                    self.vt0 = torch.Tensor(varlist[i]).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('Custom vt0 assigned.')
                elif i == 1:
                    self.vt1 = torch.Tensor(varlist[i]).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('Custom vt1 assigned.')
            
        return 1
    
    # parse input affine transforms
    def parseInputATransforms(self,affineA):
        if affineA is not None:
            if not isinstance(affineA,(np.ndarray, torch.Tensor)):
                print('ERROR: input affineA must be of type numpy.ndarray or torch.Tensor.')
                return -1
            else:
                if not affineA.shape == self.affineA.shape:
                    print('ERROR: input affineA must be of shape ' + str(list(self.affineA.shape)) + ', received shape ' + str(list(affineA.shape)) + '.')
                    return -1
            
            self.affineA = torch.Tensor(affineA).type(self.params['dtype']).to(device=self.params['cuda'])
            print('Custom affineA assigned.')
        
        return 1
    
    # save transforms to numpy arrays
    def outputTransforms(self):
        if hasattr(self,'affineA') and hasattr(self,'vt0'):
            return [x.cpu().numpy() for x in self.vt0], [x.cpu().numpy() for x in self.vt1], self.affineA.cpu().numpy()
        elif hasattr(self,'affineA'):
            return self.affineA.cpu().numpy()
        elif hasattr(self,'vt0'):
            return [x.cpu().numpy() for x in self.vt0], [x.cpu().numpy() for x in self.vt1]
        else:
            print('ERROR: no LDDMM or linear transforms to output.')
    
    # output deformed template
    def outputDeformedTemplate(self):
        if self.params['low_memory'] == 0:
            return [x[-1].cpu().numpy() for x in self.It]
        else:
            return [(self.applyThisTransformNT(x)).cpu().numpy() for x in self.I]

    # save files to disk
    def saveOutputs(self, save_template=False):
        if save_template:
            for i in range(len(self.I)):
                outimg = nib.AnalyzeImage(self.It[i][-1].to('cpu').numpy(),None)
                outimg.header['pixdim'][1:4] = self.dx
                nib.save(outimg,self.params['outdir'] + 'deformed_template_ch' + str(i) + '.img')
    
    # load transforms from numpy arrays into object
    def loadTransforms(self,vt0=None, vt1=None, affineA=None):
        # check parameters
        flag = self._checkParameters()
        if flag==-1:
            print('ERROR: parameters did not check out.')
            return
        
        if self.initializer_flags['load'] == 1:
            # load images
            flag = self._load(self.params['template'],self.params['target'],self.params['costmask'])
            if flag==-1:
                print('ERROR: images did not load.')
                return
            
        # initialize initialize
        self.initializeVariables2d()
        
        varlist = [vt0,vt1]
        namelist = ['vt0','vt1']
        for i in range(len(varlist)):
            if varlist[i] is not None:
                if not isinstance(varlist[i],list):
                    print('ERROR: input \'' + str(namelist[i]) + '\' must be a list.')
                    return -1
                else:
                    if len(varlist[i]) != len(self.vt0):
                        print('ERROR: input \'' + str(namelist[i]) + '\' must be a list of length ' + str(len(self.vt0)) + ', length ' + str(len(varlist[i])) + ' was received.')
                        return -1
                    else:
                        for ii in range(len(varlist[i])):
                            if not isinstance(varlist[i][ii],(np.ndarray,torch.Tensor)):
                                print('ERROR: input \'' + str(namelist[i]) + '\' must be a list of numpy.ndarray or torch.Tensor.')
                                return -1
                            elif not varlist[i][ii].shape == self.vt0[ii].shape:
                                print('ERROR: input \'' + str(namelist[i]) + '\' must be a list of numpy.ndarray or torch.Tensor of shapes ' + str(list(self.vt0[ii].shape)) + ', shape ' + str(list(varlist[i][ii].shape)) + ' was received.')
                                return -1
                
                if i == 0:
                    self.vt0 = torch.Tensor(varlist[i]).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('Custom vt0 assigned.')
                elif i == 1:
                    self.vt1 = torch.Tensor(varlist[i]).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('Custom vt1 assigned.')
        
        if affineA is not None:
            if not isinstance(affineA,(np.ndarray, torch.Tensor)):
                print('ERROR: input affineA must be of type numpy.ndarray or torch.Tensor.')
                return -1
            else:
                if not affineA.shape == self.affineA.shape:
                    print('ERROR: input affineA must be of shape ' + str(list(self.affineA.shape)) + ', received shape ' + str(list(affineA.shape)) + '.')
                    return -1
            
            self.affineA = torch.Tensor(affineA).type(self.params['dtype']).to(device=self.params['cuda'])
            print('Custom affineA assigned.')
        
        return 1
            
    
    # convenience function
    def run(self, restart=True, vt0=None, vt1=None, affineA=None, save_template=False):
        # check parameters
        
        if self.willUpdate is not None:
            print("Params Update: %s" % (str(self.willUpdate)))
            self.willUpdate = None
        
        flag = self._checkParameters()
        if flag==-1:
            print('ERROR: parameters did not check out.')
            return
        
        if self.initializer_flags['load'] == 1:
            # load images
            flag = self._load(self.params['template'],self.params['target'],self.params['costmask'])
            if flag==-1:
                print('ERROR: images did not load.')
                return
            
        self.initializeVariables2d()
        
        # check for initializing transforms
        flag = self.parseInputVTransforms(vt0,vt1)
        if flag == -1:
            print('ERROR: problem with input velocity fields.')
            return

        flag = self.parseInputATransforms(affineA)
        if flag == -1:
            print('ERROR: problem with input linear transforms.')
            return

        # initialize stuff for gradient function
        self._allocateGradientDivisors()
        
        # initialize kernels
        if self.params['do_lddmm'] == 1:
            self.initializeKernels2d()
            
        # main loop
        self.registration()
        
        # update epsilon
        if self.params['update_epsilon'] == 1:
            self.updateEpsilonAfterRun()
        
        # save outputs
        self.saveOutputs(save_template=save_template)
        

def loadLDDMM2D_np(ldm: LDDMM2D, folder):
    if not os.path.exists(folder):
        return None
    
    device = ldm.params.get('cuda', 'cpu')
    dtyp = ldm.params["dtype"]
    ldm.params['cuda'] = device   
    
    x0 = np.load(folder + "/x0.npy")
    ldm.X0 = torch.tensor(x0).type(dtyp).to(device=device)
    
    x1 = np.load(folder + "/x1.npy")
    ldm.X1 = torch.tensor(x1).type(dtyp).to(device=device)
    
    nx = np.load(folder + "/nx.npy")
    ldm.nx = np.array(nx)
    
    dx = np.load(folder + "/dx.npy")
    ldm.dx = np.array(dx)
    
    ldm.dt = np.load(folder + "/dt.npy")
    
    ldm.params['nt'] = np.load(folder + "/nt.npy")
    
    ldm.vt0 = []
    vt0 = np.load(folder + "/vt0.npy")
    for v in vt0:
        ldm.vt0.append(torch.tensor(v).type(dtyp).to(device=device))
        
    ldm.vt1 = []
    vt1 = np.load(folder + "/vt1.npy")
    for v in vt1:
        ldm.vt1.append(torch.tensor(v).type(dtyp).to(device=device))
        
    afA = np.load(folder + "/affineA.npy")
    ldm.affineA = torch.tensor(afA).type(dtyp).to(device=device)
    return ldm
    
def saveLDDMM2D_np(ldm: LDDMM2D, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    x0 = ldm.X0.clone().detach().cpu().numpy()
    np.save(folder + "/x0.npy", x0)

    x1 = ldm.X1.clone().detach().cpu().numpy()
    np.save(folder + "/x1.npy", x1)
    
    np.save(folder + "/nx.npy", ldm.nx)
    np.save(folder + "/dx.npy", ldm.dx)
    np.save(folder + "/dt.npy", ldm.dt)
    np.save(folder + "/nt.npy", np.array(ldm.params['nt']))
    
    
    vts = []
    for v in ldm.vt0:
        vts.append(v.clone().detach().cpu().numpy())
    np.save(folder + "/vt0.npy", np.array(vts))
    
    vts = []
    for v in ldm.vt1:
        vts.append(v.clone().detach().cpu().numpy())
    np.save(folder + "/vt1.npy", np.array(vts))
    
    afA = ldm.affineA.clone().detach().cpu().numpy()
    np.save(folder + "/affineA.npy", afA)
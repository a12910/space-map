import torch
import numpy as np
import scipy.linalg
import time
import sys
import os
import distutils.version
import nibabel as nib
import spacemap

def mygaussian(sigma=1,size=5):
    ind = np.linspace(-np.floor(size/2.0),np.floor(size/2.0),size)
    X,Y = np.meshgrid(ind,ind,indexing='xy')
    out_mat = np.exp(-(X**2 + Y**2) / (2*sigma**2))
    out_mat = out_mat / np.sum(out_mat)
    return out_mat

def mygaussian_torch_selectcenter(sigma=1,center_x=0,center_y=0,size_x=100,size_y=100):
    ind_x = torch.linspace(0,size_x-1,size_x)
    ind_y = torch.linspace(0,size_y-1,size_y)
    X,Y = torch.meshgrid(ind_x,ind_y)
    out_mat = torch.exp(-((X-center_x)**2 + (Y-center_y)**2) / (2*sigma**2))
    #out_mat = out_mat / torch.sum(out_mat)
    return out_mat

def mygaussian_torch_selectcenter_meshgrid(X,Y,sigma=1,center_x=0,center_y=0):
    out_mat = torch.exp(-((X-center_x)**2 + (Y-center_y)**2) / (2*sigma**2))
    #out_mat = out_mat / torch.sum(out_mat)
    return out_mat

def mygaussian3d(sigma=1,size=5):
    ind = np.linspace(-np.floor(size/2.0),np.floor(size/2.0),size)
    X,Y,Z = np.meshgrid(ind,ind,ind,indexing='xy')
    out_mat = np.exp(-(X**2 + Y**2 + Z**2) / (2*sigma**2))
    out_mat = out_mat / np.sum(out_mat)
    return out_mat

def mygaussian_3d_torch_selectcenter_meshgrid(X,Y,Z,sigma=1,center_x=0,center_y=0,center_z=0):
    out_mat = torch.exp(-((X-center_x)**2 + (Y-center_y)**2 + (Z-center_z)**2) / (2*sigma**2))
    #out_mat = out_mat / torch.sum(out_mat)
    return out_mat

def grid_sample(*args,**kwargs):
    if distutils.version.LooseVersion(torch.__version__) < distutils.version.LooseVersion("1.3.0"):
        return torch.nn.functional.grid_sample(*args,**kwargs)
    else:
        return torch.nn.functional.grid_sample(*args,**kwargs,align_corners=True)

def _fillGrid(grid):
    if isinstance(grid, np.ndarray):
        grid = torch.tensor(grid).type(torch.FloatTensor)
    else:
        grid = grid.clone()
    if len(grid.shape) == 3:
        grid = grid.unsqueeze(0)
    return grid

def irfft(mat,dim,onesided=False):
    if distutils.version.LooseVersion(torch.__version__) < distutils.version.LooseVersion("1.8.0"):
        return torch.irfft(mat,dim,onesided=onesided)
    else:
        return torch.fft.irfftn(mat,s=mat.shape)

def rfft(mat,dim,onesided=False):
    if distutils.version.LooseVersion(torch.__version__) < distutils.version.LooseVersion("1.8.0"):
        return torch.rfft(mat,dim,onesided=onesided)
    else:
        return torch.fft.fftn(mat)
    
    
class LDDMMBase:
    def __init__(self,template=None,target=None,costmask=None,outdir='./',gpu_number=0,
                 a=5.0,p=2,niter=100,epsilon=5e-3,epsilonL=1.0e-7,epsilonT=2.0e-5,sigma=2.0,sigmaR=1.0,
                 nt=5,do_lddmm=1,do_affine=0,checkaffinestep=0,optimizer='gd',
                 sg_mask_mode='ones',sg_rand_scale=1.0,sg_sigma=1.0,sg_climbcount=1,sg_holdcount=1,sg_gamma=0.9,
                 adam_alpha=0.1,adam_beta1=0.9,adam_beta2=0.999,adam_epsilon=1e-8,ada_rho=0.95,ada_epsilon=1e-6,
                 rms_rho=0.9,rms_epsilon=1e-8,rms_alpha=0.001,maxclimbcount=3,savebestv=False, rmlambda=1.0,
                 minenergychange = 0.000001,minbeta=1e-4,dtype='float',im_norm_ms=0,
                 slice_alignment=0,energy_fraction=0.02,energy_fraction_from=0,
                 cc=0,cc_channels=[],we=0,we_channels=[],sigmaW=1.0,nMstep=5,
                 dx=None,low_memory=0,update_epsilon=0,verbose=100,v_scale=1.0,
                 v_scale_smoothing=0,target_err=30,target_step=8000,show_init=True,target_err_skip=0.001):
        self.params = {}
        self.params['gpu_number'] = gpu_number
        self.params['a'] = float(a)
        self.params['p'] = float(p)
        self.params['niter'] = niter
        self.params['epsilon'] = float(epsilon)
        self.params['epsilonL'] = float(epsilonL)
        self.params['epsilonT'] = float(epsilonT)
        self.params["target_err"] = int(target_err)
        self.params["target_step"] = int(target_step)
        self.params['rmlambda'] = float(rmlambda)
        if isinstance(sigma,(int,float)):
            self.params['sigma'] = float(sigma)
        else:
            self.params['sigma'] = [float(x) for x in sigma]
        self.params['sigmaR'] = float(sigmaR)
        self.params['nt'] = nt
        self.params['orig_nt'] = nt
        self.params['template'] = template
        self.params['target'] = target
        self.params['costmask'] = costmask
        self.params['outdir'] = outdir
        self.params['do_lddmm'] = do_lddmm
        self.params['do_affine'] = do_affine
        self.params['checkaffinestep'] = checkaffinestep
        self.params['optimizer'] = optimizer
        self.params['sg_sigma'] = float(sg_sigma)
        self.params['sg_mask_mode'] = sg_mask_mode
        self.params['sg_rand_scale'] = float(sg_rand_scale)
        self.params['sg_climbcount'] = int(sg_climbcount)
        self.params['sg_holdcount'] = int(sg_holdcount)
        self.params['sg_gamma'] = float(sg_gamma)
        self.params['adam_alpha'] = float(adam_alpha)
        self.params['adam_beta1'] = float(adam_beta1)
        self.params['adam_beta2'] = float(adam_beta2)
        self.params['adam_epsilon'] = float(adam_epsilon)
        self.params['ada_rho'] = float(ada_rho)
        self.params['ada_epsilon'] = float(ada_epsilon)
        self.params['rms_rho'] = float(rms_rho)
        self.params['rms_epsilon'] = float(rms_epsilon)
        self.params['rms_alpha'] = float(rms_alpha)
        self.params['maxclimbcount'] = maxclimbcount
        self.params['savebestv'] = savebestv
        self.params['minbeta'] = minbeta
        self.params['minenergychange'] = minenergychange
        self.params['im_norm_ms'] = im_norm_ms
        self.params['slice_alignment'] = slice_alignment
        self.params['energy_fraction'] = energy_fraction
        self.params['energy_fraction_from'] = energy_fraction_from
        self.params['cc'] = cc
        self.params['cc_channels'] = cc_channels
        self.params['we'] = we
        self.params['we_channels'] = we_channels
        self.params['sigmaW'] = sigmaW
        self.params['nMstep'] = nMstep
        self.params['v_scale'] = float(v_scale)
        self.params['v_scale_smoothing'] = int(v_scale_smoothing)
        self.params['dx'] = dx
        dtype_dict = {}
        dtype_dict['float'] = 'torch.FloatTensor'
        dtype_dict['double'] = 'torch.DoubleTensor'
        self.params['dtype'] = dtype_dict[dtype]
        self.params['low_memory'] = low_memory
        self.params['update_epsilon'] = float(update_epsilon)
        self.params['verbose'] = float(verbose)
        self.params["total_step"] = 0
        self.params["last_err"] = 1e6
        self.params["last_time"] = time.time()
        self.params["target_err_skip"] = float(target_err_skip)
        optimizer_dict = {}
        optimizer_dict['gd'] = 'gradient descent'
        optimizer_dict['gdr'] = 'gradient descent with reducing epsilon'
        optimizer_dict['gdw'] = 'gradient descent with delayed reducing epsilon'
        optimizer_dict['adam'] = 'adaptive moment estimation (UNDER CONSTRUCTION)'
        optimizer_dict['adadelta'] = 'adadelta (UNDER CONSTRUCTION)'
        optimizer_dict['rmsprop'] = 'root mean square propagation (UNDER CONSTRUCTION)'
        optimizer_dict['sgd'] = 'stochastic gradient descent'
        optimizer_dict['sgdm'] = 'stochastic gradient descent with momentum (UNDER CONSTRUCTION)'
        if show_init:
            print('\nCurrent parameters:')
            print('>    a               = ' + str(a) + ' (smoothing kernel, a*(pixel_size))')
            print('>    p               = ' + str(p) + ' (smoothing kernel power, p*2)')
            print('>    niter           = ' + str(niter) + ' (number of iterations)')
            print('>    epsilon         = ' + str(epsilon) + ' (gradient descent step size)')
            print('>    epsilonL        = ' + str(epsilonL) + ' (gradient descent step size, affine)')
            print('>    epsilonT        = ' + str(epsilonT) + ' (gradient descent step size, translation)')
            print('>    minbeta         = ' + str(minbeta) + ' (smallest multiple of epsilon)')
            print('>    sigma           = ' + str(sigma) + ' (matching term coefficient (0.5/sigma**2))')
            print('>    sigmaR          = ' + str(sigmaR)+ ' (regularization term coefficient (0.5/sigmaR**2))')
            print('>    nt              = ' + str(nt) + ' (number of time steps in velocity field)')
            print('>    do_lddmm        = ' + str(do_lddmm) + ' (perform LDDMM step, 0 = no, 1 = yes)')
            print('>    do_affine       = ' + str(do_affine) + ' (interleave linear registration: 0 = no, 1 = affine, 2 = rigid, 3 = rigid + scale)')
            print('>    checkaffinestep = ' + str(checkaffinestep) + ' (evaluate linear matching energy: 0 = no, 1 = yes)')
            print('>    im_norm_ms      = ' + str(im_norm_ms) + ' (normalize image by mean and std: 0 = no, 1 = yes)')
            print('>    gpu_number      = ' + str(gpu_number) + ' (index of CUDA_VISIBLE_DEVICES to use)')
            print('>    dtype           = ' + str(dtype) + ' (bit depth, \'float\' or \'double\')')
            print('>    energy_fraction = ' + str(energy_fraction) + ' (fraction of initial energy at which to stop)')
            print('>    cc              = ' + str(cc) + ' (contrast correction: 0 = no, 1 = yes)')
            print('>    cc_channels     = ' + str(cc_channels) + ' (image channels to run contrast correction (0-indexed))')
            print('>    we              = ' + str(we) + ' (weight estimation: 0 = no, 2+ = yes)')
            print('>    we_channels     = ' + str(we_channels) + ' (image channels to run weight estimation (0-indexed))')
            print('>    sigmaW          = ' + str(sigmaW) + ' (coefficient for each weight estimation class)')
            print('>    nMstep          = ' + str(nMstep) + ' (update weight estimation every nMstep steps)')
            print('>    v_scale         = ' + str(v_scale) + ' (parameter scaling factor)')
            print('>    target_err         = ' + str(target_err) + ' (parameter target err)')
            print('>    target_step         = ' + str(target_step) + ' (parameter target step)')
            if v_scale < 1.0:
                print('>    v_scale_smooth  = ' + str(v_scale_smoothing) + ' (smoothing before interpolation for v-scaling: 0 = no, 1 = yes)')
            print('>    low_memory      = ' + str(low_memory) + ' (low memory mode: 0 = no, 1 = yes)')
            print('>    update_epsilon  = ' + str(update_epsilon) + ' (update optimization step size between runs: 0 = no, 1 = yes)')
            print('>    outdir          = ' + str(outdir) + ' (output directory name)')
            if optimizer in optimizer_dict:
                print('>    optimizer       = ' + str(optimizer_dict[optimizer]) + ' (optimizer type)')
                if optimizer == 'adam':
                    print('>    +adam_alpha     = ' + str(adam_alpha) + ' (learning rate)')
                    print('>    +adam_beta1     = ' + str(adam_beta1) + ' (decay rate 1)')
                    print('>    +adam_beta2     = ' + str(adam_beta2) + ' (decay rate 2)')
                    print('>    +adam_epsilon   = ' + str(adam_epsilon) + ' (epsilon)')
                    print('>    +sg_sigma       = ' + str(sg_sigma) + ' (subsampler sigma (for gaussian mode))')
                    print('>    +sg_mask_mode   = ' + str(sg_mask_mode) + ' (subsampler scheme)')
                    print('>    +sg_climbcount  = ' + str(sg_climbcount) + ' (# of times energy is allowed to increase)')
                    print('>    +sg_holdcount   = ' + str(sg_holdcount) + ' (# of iterations per random mask)')
                    print('>    +sg_rand_scale  = ' + str(sg_rand_scale) + ' (scale for non-gauss sg masking)')
                elif optimizer == "adadelta":
                    print('>    +ada_rho        = ' + str(ada_rho) + ' (decay rate)')
                    print('>    +ada_epsilon    = ' + str(ada_epsilon) + ' (epsilon)')
                elif optimizer == "rmsprop":
                    print('>    +rms_rho        = ' + str(rms_rho) + ' (decay rate)')
                    print('>    +rms_epsilon    = ' + str(rms_epsilon) + ' (epsilon)')
                    print('>    +rms_alpha      = ' + str(rms_alpha) + ' (learning rate)')
                    print('>    +sg_sigma       = ' + str(sg_sigma) + ' (subsampler sigma (for gaussian mode))')
                    print('>    +sg_mask_mode   = ' + str(sg_mask_mode) + ' (subsampler scheme)')
                    print('>    +sg_climbcount  = ' + str(sg_climbcount) + ' (# of times energy is allowed to increase)')
                    print('>    +sg_holdcount   = ' + str(sg_holdcount) + ' (# of iterations per random mask)')
                    print('>    +sg_rand_scale  = ' + str(sg_rand_scale) + ' (scale for non-gauss sg masking)')
                elif optimizer == 'sgd' or optimizer == 'sgdm':
                    print('>    +sg_sigma       = ' + str(sg_sigma) + ' (subsampler sigma (for gaussian mode))')
                    print('>    +sg_mask_mode   = ' + str(sg_mask_mode) + ' (subsampler scheme)')
                    print('>    +sg_climbcount  = ' + str(sg_climbcount) + ' (# of times energy is allowed to increase)')
                    print('>    +sg_holdcount   = ' + str(sg_holdcount) + ' (# of iterations per random mask)')
                    print('>    +sg_rand_scale  = ' + str(sg_rand_scale) + ' (scale for non-gauss sg masking)')
                    if optimizer == 'sgdm':
                        print('>    +sg_gamma       = ' + str(sg_gamma) + ' (fraction of paste updates)')
        
            else:
                print('WARNING: optimizer \'' + str(optimizer) + '\' not recognized. Setting to basic gradient descent with reducing step size.')
                self.params['optimizer'] = 'gdr'
        
            print('\n')
            if template is None:
                print('WARNING: template file name is not set. Use LDDMM.setParams(\'template\',filename\/array).\n')
            elif isinstance(template,np.ndarray):
                print('>    template        = numpy.ndarray\n')
            elif isinstance(template,list) and isinstance(template[0],np.ndarray):
                myprintstring = '>    template        = [numpy.ndarray'
                for i in range(len(template)-1):
                    myprintstring = myprintstring + ', numpy.ndarray'
                
                myprintstring = myprintstring + ']\n'
                print(myprintstring)
            else:
                print('>    template        = ' + str(template) + '\n')
            
            if target is None:
                print('WARNING: target file name is not set. Use LDDMM.setParams(\'target\',filename\/array).\n')
            elif isinstance(target,np.ndarray):
                print('>    target          = numpy.ndarray\n')
            elif isinstance(target,list) and isinstance(target[0],np.ndarray):
                myprintstring = '>    target          = [numpy.ndarray'
                for i in range(len(target)-1):
                    myprintstring = myprintstring + ', numpy.ndarray'
                
                myprintstring = myprintstring + ']\n'
                print(myprintstring)
            else:
                print('>    target          = ' + str(target) + '\n')
            
            if isinstance(costmask,np.ndarray):
                print('>    costmask        = numpy.ndarray (costmask file name or numpy.ndarray)')
            else:
                print('>    costmask        = ' + str(costmask) + ' (costmask file name or numpy.ndarray)')
        
        self.initializer_flags = {}
        self.initializer_flags['load'] = 1
        self.initializer_flags['v_scale'] = 0
        if self.params['do_lddmm'] == 1:
            self.initializer_flags['lddmm'] = 1
        else:
            self.initializer_flags['lddmm'] = 0
        
        if self.params['do_affine'] > 0:
            self.initializer_flags['affine'] = 1
        else:
            self.initializer_flags['affine'] = 0
        
        if self.params['cc'] != 0:
            self.initializer_flags['cc'] = 1
        else:
            self.initializer_flags['cc'] = 0
        
        if self.params['we'] >= 2:
            self.initializer_flags['we'] = 1
        else:
            self.initializer_flags['we'] = 0
            
        self.willUpdate = None
        
    def tensor(self, t):
        if isinstance(t, np.ndarray):
            t = torch.tensor(t)
        elif isinstance(t, torch.Tensor):
            t = t.clone()
        return t.to(device=self.params['cuda'])
    
    def tensor_ncp(self, t):
        if isinstance(t, np.ndarray):
            t = torch.tensor(t)
        return t.to(device=self.params['cuda'])
    
    # manual edit parameter
    def setParams(self,parameter_name,parameter_value):
        if self.willUpdate is None:
            self.willUpdate = {}
        if parameter_name in self.params:
            # print('Parameter \'' + str(parameter_name) + '\' changed to \'' + str(parameter_value) + '\'.')
            if parameter_name == 'template' or parameter_name == 'target' or parameter_name == 'costmask':
                self.initializer_flags['load'] = 1
            elif parameter_name == 'do_lddmm' and parameter_value == 1 and self.params['do_lddmm'] == 0:
                self.initializer_flags['lddmm'] = 1
                print('WARNING: LDDMM state has changed. Variables will be initialized.')
            elif parameter_name == 'do_affine' and parameter_value > 0 and self.params['do_affine'] != parameter_value:
                self.initializer_flags['affine'] = 1
                print('WARNING: Affine state has changed. Variables will be initialized.')
            elif (parameter_name == 'cc' and parameter_value != 0 and self.params['cc'] != parameter_value) or (parameter_name == 'cc_channels' and parameter_value != self.params['cc_channels']):
                self.initializer_flags['cc'] = 1
                print('WARNING: Contrast correction state has changed. Variables will be initialized.')
            elif parameter_name == 'we' and parameter_value >= 2 and self.params['we'] != parameter_value or (parameter_name == 'we_channels' and parameter_value != self.params['we_channels']):
                self.initializer_flags['we'] = 1
                print('WARNING: Weight estimation state has changed. Variables will be initialized.')
            elif parameter_name == 'v_scale' and self.params['do_lddmm'] == 1 and hasattr(self,'vt0'):
                self.initializer_flags['v_scale'] = 1
                print('WARNING: Parameter sparsity has changed. Variables will be initialized.')
            
            self.willUpdate[parameter_name] = parameter_value
            self.params[parameter_name] = parameter_value
        else:
            print('Parameter \'' + str(parameter_name) + '\' is not a valid parameter.')
        
        return
    
        # image loader
    def loadImage(self, filename,im_norm_ms=0):
        fname, fext = os.path.splitext(filename)
        if fext == '.img' or fext == '.hdr':
            img_struct = nib.load(fname + '.img')
            spacing = img_struct.header['pixdim'][1:4]
            size = img_struct.header['dim'][1:4]
            image = np.squeeze(img_struct.get_data().astype(np.float32))
            if im_norm_ms == 1:
                if np.std(image) != 0:
                    image = torch.tensor((image - np.mean(image)) / np.std(image)).type(self.params['dtype']).to(device=self.params['cuda'])

                else:
                    image = torch.tensor((image - np.mean(image)) ).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('WARNING: stdev of image is zero, not rescaling.')
            else:
                image = torch.tensor(image).type(self.params['dtype']).to(device=self.params['cuda'])
            return (image, spacing, size)
        elif fext == '.nii':
            img_struct = nib.load(fname + '.nii')
            spacing = img_struct.header['pixdim'][1:4]
            size = img_struct.header['dim'][1:4]
            image = np.squeeze(img_struct.get_data().astype(np.float32))
            if im_norm_ms == 1:
                if np.std(image) != 0:
                    image = torch.tensor((image - np.mean(image)) / np.std(image)).type(self.params['dtype']).to(device=self.params['cuda'])

                else:
                    image = torch.tensor((image - np.mean(image)) ).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('WARNING: stdev of image is zero, not rescaling.')
            else:
                image = torch.tensor(image).type(self.params['dtype']).to(device=self.params['cuda'])
            return (image, spacing, size)
        else:
            print('File format not supported.\n')
            return (-1,-1,-1)
    
    # helper function to check parameters before running registration
    def _checkParameters(self):
        flag = 1
        if self.params['gpu_number'] is not None and not isinstance(self.params['gpu_number'], (int, float)):
            flag = -1
            print('ERROR: gpu_number must be None or a number.')
        else:
            if self.params['gpu_number'] is None:
                self.params['cuda'] = 'cpu'
            elif self.params['gpu_number'] == -1:
                self.params['cuda'] = 'mps'
            else:
                self.params['cuda'] = 'cuda:' + str(self.params['gpu_number'])
        
        number_list = ['a','p','niter','epsilon','sigmaR','nt','do_lddmm','do_affine','epsilonL','epsilonT','im_norm_ms','slice_alignment','energy_fraction','energy_fraction_from','cc','we','nMstep','low_memory','update_epsilon','v_scale','adam_alpha','adam_beta1','adam_beta2','adam_epsilon','ada_rho','ada_epsilon','rms_rho','rms_alpha','rms_epsilon','sg_sigma','sg_climbcount','sg_rand_scale','sg_holdcount','sg_gamma','v_scale_smoothing','verbose', 'rmlambda']
        string_list = ['outdir','optimizer']
        stringornone_list = ['costmask'] # or array, actually
        stringorlist_list = ['template','target'] # or array, actually
        numberorlist_list = ['sigma','cc_channels','we_channels','sigmaW']
        noneorarrayorlist_list = ['dx']
        for i in range(len(number_list)):
            if not isinstance(self.params[number_list[i]], (int, float)):
                flag = -1
                print('ERROR: ' + number_list[i] + ' must be a number.')
        
        for i in range(len(string_list)):
            if not isinstance(self.params[string_list[i]], str):
                flag = -1
                print('ERROR: ' + string_list[i] + ' must be a string.')
        
        for i in range(len(stringornone_list)):
            if not isinstance(self.params[stringornone_list[i]], (str,np.ndarray)) and self.params[stringornone_list[i]] is not None:
                flag = -1
                print('ERROR: ' + stringornone_list[i] + ' must be a string or None.')
        
        for i in range(len(stringorlist_list)):
            if not isinstance(self.params[stringorlist_list[i]], str) and not isinstance(self.params[stringorlist_list[i]], list) and not isinstance(self.params[stringorlist_list[i]], np.ndarray):
                flag = -1
                print('ERROR: ' + stringorlist_list[i] + ' must be a string or an np.ndarray or a list of these.')
            elif isinstance(self.params[stringorlist_list[i]], (str,np.ndarray)):
                self.params[stringorlist_list[i]] = [self.params[stringorlist_list[i]]]
        
        for i in range(len(numberorlist_list)):
            if not isinstance(self.params[numberorlist_list[i]], (int,float)) and not isinstance(self.params[numberorlist_list[i]], list):
                flag = -1
                print('ERROR: ' + numberorlist_list[i] + ' must be a number or a list of numbers.')
            elif isinstance(self.params[numberorlist_list[i]], (int,float)):
                self.params[numberorlist_list[i]] = [self.params[numberorlist_list[i]]]
        
        for i in range(len(noneorarrayorlist_list)):
            if self.params[noneorarrayorlist_list[i]] is not None and not isinstance(self.params[noneorarrayorlist_list[i]], list) and not isinstance(self.params[noneorarrayorlist_list[i]], np.ndarray):
                flag = -1
                print('ERROR: ' + noneorarrayorlist_list[i] + ' must be None or a list or a np.ndarray.')
            elif isinstance(self.params[noneorarrayorlist_list[i]], str):
                self.params[noneorarrayorlist_list[i]] = [self.params[noneorarrayorlist_list[i]]]
            elif isinstance(self.params[noneorarrayorlist_list[i]], np.ndarray):
                self.params[noneorarrayorlist_list[i]] = np.ndarray.tolist(self.params[noneorarrayorlist_list[i]])
        
        # check channel length
        channel_check_list = ['sigma','template','target']
        channels = [len(self.params[x]) for x in channel_check_list]
        channel_set = list(set(channels))
        if len(channel_set) > 2 or (len(channel_set) == 2 and 1 not in channel_set):
            print('ERROR: number of channels is not the same between sigma, template, and target.')
            flag = -1
        elif len(self.params['template']) != len(self.params['target']):
            print('ERROR: number of channels is not the same between template and target.')
            flag = -1
        elif len(self.params['sigma']) > 1 and len(self.params['sigma']) != len(self.params['template']):
            print('ERROR: sigma does not have channels of size 1 or # of template channels.')
            flag = -1
        elif (len(channel_set) == 2 and 1 in channel_set):
            channel_set.remove(1)
            for i in range(len(channel_check_list)):
                if channels[i] == 1:
                    self.params[channel_check_list[i]] = self.params[channel_check_list[i]]*channel_set[0]
            
            print('WARNING: one or more of sigma, template, and target has length 1 while another does not.')
        
        # check contrast correction channels
        if isinstance(self.params['cc_channels'],(int,float)):
            self.params['cc_channels'] = [int(self.params['cc_channels'])]
        elif isinstance(self.params['cc_channels'],list):
            if len(self.params['cc_channels']) == 0:
                self.params['cc_channels'] = list(range(max(channel_set)))
            
            if max(self.params['cc_channels']) > max(channel_set):
                print('ERROR: one or more of the contrast correction channels is greater than the number of image channels.')
                flag = -1
        
        # check weight estimation channels
        if isinstance(self.params['we_channels'],(int,float)):
            self.params['we_channels'] = [int(self.params['we_channels'])]
        elif isinstance(self.params['we_channels'],list):
            if len(self.params['we_channels']) == 0:
                self.params['we_channels'] = list(range(max(channel_set)))
            
            if max(self.params['we_channels']) > max(channel_set):
                print('ERROR: one or more of the weight estimation channels is greater than the number of image channels.')
                flag = -1
        
        # check weight estimation sigmas
        if len(self.params['sigmaW']) == 1:
            self.params['sigmaW'] = self.params['sigmaW']*int(self.params['we'])
        elif len(self.params['sigmaW']) != int(self.params['we']):
            print('ERROR: length of weight estimation sigma list must be either 1 or equal to parameter \'we\'.')
            flag = -1
        
        # optimizer flags
        if self.params['optimizer'] == 'gdw':
            self.params['savebestv'] = True
        
        # set timesteps to 1 if doing affine only
        if self.params['do_affine'] > 0 and self.params['do_lddmm'] == 0 and not hasattr(self,'vt0'):
            if self.params['nt'] != 1:
                print('WARNING: nt set to 1 because settings indicate affine registration only.')
                self.params['nt'] = 1
        elif self.params['do_affine'] == 0 and self.params['do_lddmm'] == 0:
            flag = -1
            print('ERROR: both linear and LDDMM registration are turned off. Exiting.')
        elif self.params['do_lddmm'] == 1:
            if self.params['nt'] == 1 and self.params['orig_nt'] == 1:
                print('WARNING: parameter \'nt\' is currently set to 1. You might have just finished linear registration. For LDDMM, set to a higher value.')
            elif self.params['nt'] == 1 and self.params['orig_nt'] != 1:
                self.params['nt'] = self.params['orig_nt']
                print('WARNING: parameter \'nt\' was set to 1 and has been automatically reverted to your initial value of ' + str(self.params['orig_nt']) + '.')
        
        return flag
    
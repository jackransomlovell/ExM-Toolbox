import SimpleITK as sitk

class sitkTile:
    # 1. estimate transformation between input volumes
    # 2. warp one volume with the transformation
    def __init__(self):
        self.elastix = sitk.ElastixImageFilter()
        self.transformix = sitk.TransformixImageFilter()
        self.otsu = sitk.OtsuThresholdImageFilter()
        self.dilate = sitk.BinaryDilateImageFilter()
        self.thrsh = sitk.BinaryThresholdImageFilter()
        self.gauss = sitk.SmoothingRecursiveGaussianImageFilter()
        self.minmax = sitk.MinimumMaximumImageFilter()
        self.parameter_map = None
        self.transform_type = None
    
    def setResolution(self, resolution):
        # xyz-order
        self.resolution = resolution
        
    def convertSitkImage(self, vol_np, res_np):
        vol = sitk.GetImageFromArray(vol_np)
        vol.SetSpacing(res_np)
        return vol
    
    #### Setup Tform, Otsu, Kernel
    def setTransformType(self, transform_type, num_iteration = -1):
        self.transform_type = transform_type
        self.parameter_map = self.createParameterMap(transform_type, num_iteration)
        self.elastix.SetParameterMap(self.parameter_map)
        
    def updateParameterMap(self, parameter_map=None):
        if parameter_map is not None:
            self.parameter_map = parameter_map
        self.elastix.SetParameterMap(self.parameter_map)

    def getParameterMap(self):
        return self.parameter_map

    def readTransformMap(self, filename):
        return sitk.ReadParameterFile(filename)

    def writeTransformMap(self, filename, transform_map):
        return sitk.WriteParameterFile(transform_map, filename)
    
    def createParameterMap(self, transform_type = None, num_iteration = -1):
        if transform_type is None:
            transform_type = self.transform_type
        if len(transform_type) == 1:
            parameter_map = sitk.GetDefaultParameterMap(transform_type[0])
            parameter_map['NumberOfSamplesForExactGradient'] = ['5000']
            if num_iteration > 0:
                parameter_map['MaximumNumberOfIterations'] = [str(num_iteration)]
            else:
                parameter_map['MaximumNumberOfIterations'] = ['5000']
            parameter_map['MaximumNumberOfSamplingAttempts'] = ['100']
            parameter_map['FinalBSplineInterpolationOrder'] = ['1']            
        else:
            parameter_map = sitk.VectorOfParameterMap()
            for trans in transform_type:
                parameter_map.append(self.createParameterMap(trans, num_iteration))
        return parameter_map
    def getMinMax(self, image):
        self.minmax.Execute(image)
        return self.minmax.GetMaximum(),self.minmax.GetMinimum()
        
    def setThrshValue(self, upper_val, lower_val = 175, outside_val = 0, inside_val = 1):
        self.thrsh.SetLowerThreshold(lower_val)
        self.thrsh.SetUpperThreshold(upper_val)
        self.thrsh.SetOutsideValue(outside_val)
        self.thrsh.SetInsideValue(inside_val)
        
    def setOtsuValues(self, inside_val = 0, outside_val = 1):
        self.otsu.SetInsideValue(inside_val)
        self.otsu.SetOutsideValue(outside_val)
        
    def setGausSigma(self, sigma):
        self.gauss.SetSigma(sigma)
        
    def setKernelType(self, kernel = sitk.sitkBox):
        self.dilate.SetKernelType(kernel)
    
    def setKernelRadius(self, radius):
        self.dilate.SetKernelRadius(radius)

    ### Run thrsh
    def computeThresh(self, image, thrsh = None, res= None):
        if thrsh:
            self.thrsh.SetLowerThreshold(thrsh)
        if res is None:
            res = self.resolution
        #image = self.convertSitkImage(image, res)
        out_img = self.thrsh.Execute(image)
        #out = sitk.GetArrayFromImage(out_img)
        return out_img
    
    
    ### Run Otsu
    def computeOtsuThresh(self, image, inside_val = None, outside_val = None, res= None):
        if inside_val:
            self.otsu.SetInsideValue(inside_val)
        if outside_val:
            self.otsu.SetOutsideValue(outside_val)
        if res is None:
            res = self.resolution
        #image = self.convertSitkImage(image, res)
        out_img = self.otsu.Execute(image)
        #out = sitk.GetArrayFromImage(out_img)
        return out_img
    
    #run gaussian
    
    def computeGaussianFilter(self, image, sigma = None, res = None):
        if sigma:
            self.gauss.SetSigma(sigma)
        if res is None:
            res = self.resolution
        #image = self.convertSitkImage(image, res)
        out_img = self.dilate.Execute(image)
        #out = sitk.GetArrayFromImage(out_img)
        return out_img
    
    ### Run kernel
    def computeDilateFilter(self, image, kernel_type = None, kernel_radius = None, res = None):
        if kernel_type:
            self.dilate.SetKernelType(sitk.kernel_type)
        if kernel_radius:
            self.dilate.SetKernelRadius(kernel_radius)
        if res is None:
            res = self.resolution
        #image = self.convertSitkImage(image, res)
        out_img = self.dilate.Execute(image)
        out = sitk.GetArrayFromImage(out_img)
        return out_img

    #### Estimate and warp with transformation
    
    def computeTransformMap(self, vol_fix, vol_move, res_fix=None, res_move=None, mask_fix=None, mask_move=None):
        # work with mask correctly
        # https://github.com/SuperElastix/SimpleElastix/issues/198
        # not enough samples in the mask
        self.elastix.SetParameter("ImageSampler", "RandomSparseMask")
        #self.elastix.SetLogToConsole(False)
        self.elastix.SetLogToConsole(True)
        if res_fix is None:
            res_fix = self.resolution
        if res_move is None:
            res_move = self.resolution
        # 2. load volume
        # print('vol-fix shape:', vol_fix.shape)
        vol_fix = self.convertSitkImage(vol_fix, res_fix)
        self.elastix.SetFixedImage(vol_fix)
        if mask_fix is not None:
            mask_fix = self.convertSitkImage(mask_fix, res_fix)
            mask_fix.CopyInformation(vol_fix)
            self.elastix.SetFixedMask(mask_fix)

        # print('vol-move shape:', vol_move.shape)
        vol_move = self.convertSitkImage(vol_move, res_move)
        self.elastix.SetMovingImage(vol_move)
        if mask_move is not None:
            mask_move = self.convertSitkImage(mask_move, res_move)
            mask_move.CopyInformation(vol_move)
            self.elastix.SetMovingMask(mask_move)
            
        # 3. compute transformation
        self.elastix.Execute()

        # 4. output transformation parameter
        return self.elastix.GetTransformParameterMap()[0]
        
    def warpVolume(self, vol_move, transform_map, res_move=None):
        self.transformix.SetLogToConsole(True)
        if res_move is None:
            res_move = self.resolution
        self.transformix.SetTransformParameterMap(transform_map)
        self.transformix.SetMovingImage(self.convertSitkImage(vol_move, res_move))
        self.transformix.Execute()
        out = sitk.GetArrayFromImage(self.transformix.GetResultImage())
        return out

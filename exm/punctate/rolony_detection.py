class rolony_detection:
    def __init__(self):
        self.assignment = None
        self.masks = None
        self.thresholds = None
        self.channel_names = None
    
    def rolony_detection(self, h5_path, channel_names, thresholds, truncate = False, round_ind = None):
        
        self.channel_names = channel_names
        self.num_channels = len(self.channel_names)
        self.thresholds = thresholds
        
        if self.thresholds.ndim > 1:
            assert round_ind is not None, "must specify round_ind for multiple rounds of thresholds"        
        
        with h5py.File(h5_path, "r+") as f:
            #get image shape for mask
            image = f[channel_names[0]]
            if truncate:
                image = truncate(np.asarray(image))
            else:
                image = np.asarray(image)
            z,y,x = image.shape
            masks = np.zeros((self.num_channels,z,y,x))
            
            #compute masks for each channel
            for c in tqdm(range(self.num_channels)):
                    
                image = f[channel_names[c]]
                if truncate:
                    image = truncate(np.asarray(image))
                else:
                    image = np.asarray(image)
                
                #thresh_image = np.zeros(image.shape, 'double')
                #sigma = (block_size - 1) / 6.0
                #gaussian_filter(image, sigma, output=thresh_image, mode='reflect', cval=0)
                #masks[c,:,:,:] = image >= (thresh_image+50)
                if self.thresholds.ndim > 1:
                    masks[c,:,:,:] = image > thresholds[round_ind,c]
                else:
                    masks[c,:,:,:] = image > thresholds[c]
            
            #save masks in h5 and write as attribute
            masks = np.asarray(masks)
            self.masks = masks
            if "masks" in f.keys():
                del f["masks"]
            dset = f.create_dataset("masks", masks.shape, dtype='uint16') 
            dset[...] = masks
            
            #compute max projection of masks
            masks = np.sum(masks,axis = 0) > 0 
            labels = label(masks, connectivity=1)
            regions = regionprops(labels)
            
            #record intensities, coords, and channels of centroids in assignment
            assignment = np.zeros((len(regions),8))
            for c in range(self.num_channels):
                if truncate:
                    chunk = truncate(f[channel_names[c]])
                else: 
                    chunk = f[channel_names[c]]
                for ind,region in tqdm(enumerate(regions)):
                    z0, y0, x0 = region.centroid
                    assignment[ind,c] = chunk[int(z0),int(y0),int(x0)]
                
            for ind,region in tqdm(enumerate(regions)):
                max_index = np.argmax(assignment[ind,:])
                z0, y0, x0 = region.centroid
                assignment[ind,4:8] = [z0,y0,x0,max_index]
            
            assignment = np.asarray(assignment)
            self.assignment = assignment
            
            if "assignment" in f.keys():
                del f["assignment"]
            dset = f.create_dataset("assignment", assignment.shape, dtype='uint16')  
            dset[...] = assignment
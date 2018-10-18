import torch.utils.data as data
from PIL import Image
import os
import numpy as np
from torchvision.datasets.folder import find_classes,make_dataset
from torchvision.datasets.folder import DatasetFolder,ImageFolder
from torchvision.datasets.folder import default_loader,IMG_EXTENSIONS

class ImageAugmentationDataset(ImageFolder):
    '''
        Pre_transform is applied to the source image firstly.
        Then, degrade_transform is applied to get degraded image (such as blurred image).
        Finally, both the degraded imgs and the pre-transformed images are uniformly post-transformed.
        Usually, the post-transforms are ToTensor() and Normalize()

        Degrade transform should have two arguments as input:
            1: input image to be degraded
            2: Boolean input indicates if the degrading parameters should be returned.
    '''
    def __init__(self, root, 
                pre_transform=None, 
                degrade_transform=None,
                post_transform=None,
                loader=default_loader):
        super(self.__class__, self).__init__(root, 
                            transform=None,
                            target_transform=None,
                            loader=loader)
        self.imgs = self.samples
        self.pre_transform = pre_transform
        self.degrade_transform = degrade_transform
        self.post_transform = post_transform        


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.pre_transform:
            sample = self.pre_transform(sample)
        degraded_sample = sample.copy()
        if self.degrade_transform:
            degraded_sample = self.degrade_transform(degraded_sample)

        if self.post_transform:
            sample = self.post_transform(sample)
            degraded_sample = self.post_transform(degraded_sample)

        return degraded_sample,sample



class BatchQueue(object):
    ## TODO: modify for torch api
    def __init__(self, max_length=50, y_max=1.5, y_min=0.8, 
        img_size=128,
        channels=3):
        '''
            if max length is 50, then there are at most 25 real samples and 25 fake
            sample in the queue.
        '''
        self.real=[]
        self.fake=[]
        self.max_length=max_length//2
        self.real_generator=None
        self.y_max = y_max
        self.y_min = y_min
        self.width = img_size
        self.height = img_size
        self.channels = channels
        self.input_shape = (self.height, self.width, self.channels)
        self.real_generator=None
        self.base_generator=None
        self.faking_model=None

    def init_with_generator(self,real_generator, base_generator, faking_model):
        '''
        real will be marked as 1, 
        fake samples generated by base_generator will be marked as 0
        '''
        self.real_generator=real_generator
        self.base_generator=base_generator
        self.faking_model=faking_model
        for i in range(self.max_length):
            real = get_image_batch(real_generator)
            fake = faking_model.predict(get_image_batch(base_generator))
            self.push_real(real)
            self.push_fake(fake)
        return self

    def init_with_data(self,real,fake):
        self.push_real(real)
        self.push_fake(fake)

    def reinit(self):
        '''
        init the queue again after parameters are loaded by restore()
        '''
        return self.init(self.real_generator,self.base_generator,self.faking_model)

    def push(self,real,fake):
        self.push_real(real)
        self.push_fake(fake)

    def push_real(self,realimg):
        return self._push(self.real,realimg)

    def push_fake(self,fakeimg):
        return self._push(self.fake,fakeimg)    

    def _push(self, queue, imgs):
        shape = imgs.shape
        # batch of images is pushing in
        batch_size = shape[0]
        if shape[1:] != self.input_shape:
            print('BatchQueue[Warning]:input',shape,'inconsistent with',self.input_shape)
            #return False
        for i in range(batch_size):
            self.__push(queue,imgs[i:i+1,:,:,:])
        return True

    def __push(self, queue, img):
        if len(queue)< self.max_length:
            queue.insert(0,img)
        else:
            queue.pop()
            self._push(queue,img)
 
    def get(self,queue):
        '''
        input: List[np.vector]
        make an np.ndarray with size of: all x columns
        '''
        l=len(queue)
        dat=np.concatenate(queue,0)
        return dat,l

    def get_real(self):
        r,_ = self.get(self.real)
        return r
    def get_fake(self):
        r,_ = self.get(self.fake)
        return r

    def get_batch(self):
        '''
        get a batch with batch size of self.max_length*2
        '''
        realbatch, realbatchsize = self.get(self.real)
        fakebatch, fakebatchsize = self.get(self.fake)
        x=np.concatenate([realbatch,fakebatch],0)
        ones=np.random.uniform(low=self.y_min,high=self.y_max,size=realbatchsize)
        zeros=np.random.uniform(low=-self.y_max,high=-self.y_min,size=fakebatchsize)
        y=np.concatenate([ones,zeros],0)
        return x,y

    def get_real_fake(self):
        r,_ = self.get(self.real)
        f,_ = self.get(self.fake)
        return r,f

    def new_sample(self):
        '''
        returns one img or imgs according to generator's batch size configuration
        '''
        imgs = get_image_batch(self.real_generator) # ranges -1,1
        return imgs, np.random.uniform(low=self.y_min,high=self.y_max,size=imgs.shape[0])

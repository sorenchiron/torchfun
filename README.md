# TorchFun (originally Torchure)

##for those who expect more convenient features from torch
<!-- ### -- for those who suffer from torch -->

### Functionality

* flatten(tensor)
* imshow(tensor_batch)
* load

## Install TorchFun

```bash
pip install torchfun
```

## API

### Flatten (Module)
used to reshape outputs

Usage:

    flat = Flatten()
    out = flat(x)


### flatten (function)

Usage:

    out = flatten(x)


### imshow (function)

only deal with torch channel-first image batch,

Arguements:

* x: input data cube, torch tensor or numpy array.
* title: add title to plot. (Default None)
    * title can be string, or any string-able object.
* auto_close: (default True) 
    * Close the pyplot session afterwards. 
    * Clean the environment just like you had never used matplotlib here.
    * if set to False, the plot will remain in the memory for further drawings.

Usage:

imshow(batch)
imshow(batch,title=[a,b,c])
imshow(batch,title='title')
imshow(batch,auto_close=False) 

### load (function)
Arguements:
* arbitrary arguemnts named : `a` and `b`
Load weight `a` into model `b`, or load model `b` using weight `a`
The order of the arguments doesn't matter.
Example:

    >load('weights.pts',model)

or

    >load(model,'weights.pts')

or

    >f = open('weight.pts')
    >load(f,model)

or

    >load(model,f)

Return value:
* None
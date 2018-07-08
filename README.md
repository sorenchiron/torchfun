# TorchFun (originally Torchure)

## for those who expect more convenient features from torch
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
```python
    flat = Flatten()
    out = flat(x)
```

----------------
### flatten (function)

Usage:
```python
    out = flatten(x)
```

----------------
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
```python
    imshow(batch)
    imshow(batch,title=[a,b,c])
    imshow(batch,title='title')
    imshow(batch,auto_close=False) 
```
Warnings:
```text
    TorchFun:imshow:Warning, you are using WebAgg backend for Matplotlib. 
    Please consider windowed display SDKs such as TkAgg backend and GTK* backends.
```
This means your matplotlib is using web-browser for figure display. We __strongly__ recommend you to use window-based native display because browser-based backends are fragile and tend to crash. You can change the display mamanger for matplotlib each time you execute your script by:
```python
import matplotlib
matplotlib.use('TkAgg') # or GTK GTKAgg
```
or permanantly by editing: `site-packages/matplotlib/mpl-data/matplotlibrc` and change backend to `TkAgg`

A full list of available backends can be found at:
```python
import matplotlib
matplotlib.rcsetup.all_backends
```
and, the TCL/TK GUI library for `tkinter` can be downloaded [here](https://www.tcl.tk/).

----------------

### load(a,b) (function)
Arguements:
* arbitrary arguemnts named : `a` and `b`
Load weight `a` into model `b`, or load model `b` using weight `a`
The order of the arguments doesn't matter.
Example:

```python
    >load('weights.pts',model)
```

or
```python
    >load(model,'weights.pts')
```
or
```python
    >f = open('weight.pts')
    >load(f,model)
```
or
```python
    >load(model,f)
```
Return value:
* None


----------------
### save(a,b) (function)
Arguements:
* arbitrary arguemnts named : `a` and `b`
save weight `a` into target `b`, or save model `b` into target `a`
The order of the arguments doesn't matter.

Example:

```python
    >save('weights.pts',model)
```

or
```python
    >save(model,'weights.pts')
```
or
```python
    >f = open('weight.pts')
    >save(f,model)
```
or
```python
    >save(model,f)
```
or
```python
    >save('weights.pts',state_dict)
```
or
```python
    >save(state_dict,'weights.pts')
```
Return value: None


----------------
### count_parameters(model_or_dict) (function)

Arguements:
    * model_or_dict: model or state dictionary
Return: parameter amount in python-int
        Returns 0 if datatype not understood
Usage:
```python
    count_parameters(model)
    count_parameters(state_dict)
    count_parameters(weight_tensor)
    count_parameters(numpy_array)
```
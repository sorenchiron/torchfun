# TorchFun 

More convenient features based on PyTorch (originally Torchure)

## Preface
<!-- ### -- for those who suffer from torch -->
`TorchFun` project was initiated long ago and was published in 2018-6-13. 

TorchFun is motivated by the one author loves, who sometimes encounter inconvenience using PyTorch at her work. The author published the project as a python package `TorchFun` on PyPi, so that his beloved could access it whenever and wheresoever needed.

The purposed of TorchFun is to provide functions which are important and convenient, but absent in PyTorch, like some layers and visualization utils.

This project has been undergoing in secret so that the author could credit TorchFun to his beloved as a little supprise of help, one day when this project is well-enriched or gets famous.

Interestingly, The original project name given by the author, is `Torchure`. That is because he was always multi-tasking trivial affairs in school and got scorched, and when he was learning this new framework in hope to help his beloved, he found plenty of issues/missing-functionalities. He felt that this was totally a torture. So, this project was named "Torchure" to satirize the lame PyTorch (you can still found Torchure in PyPi). And, the author hoped, by developing Torchure, his beloved could feel ease even when encountering the crappy-parts of PyTorch.

This history-of-project was appended recently, because his adorable little beloved wants a supprise immediately, or she will keep rolling on the floor.

To Eggy c71b05bf46d8772e4488335085a2e7fd.

## Latest Documentation:

Please visit: https://sorenchiron.github.io/torchfun

### Functionality
* Flatten
* flatten()
* Subpixel
* subpixel()
* imshow()
* load()
* save()
* count_parameters()
* Packsearch
* packsearch()
* hash_parameters()
* force_exist()
* whereis()

## Install TorchFun
installl
```bash
pip install torchfun
```
update
```bash
pip install -U torchfun
```


## API

### Flatten [Module]
used to reshape outputs

Usage:
```python
    flat = Flatten()
    out = flat(x)
```

----------------
### flatten(x) [Function]

Usage:
```python
    out = flatten(x)
```


----------------
### subpixel(x,out_channels) [Function]

Unfold channel/depth dimensions to enlarge the feature map

Notice: 

    Output size is deducted. 
    The size of the unfold square is automatically determined

e.g. :

    images: 100x16x16x9.  9=3x3 square
    subpixel-out: 100x48x48x1

Arguement:

    out_channels, channel number of output feature map

----------------
### Subpixel Layer [Module]

Same functionality as subpixel(x), but with Module interface.

    s = Subpixel(out_channels=1)
    out = s(x)

----------------
### imshow(x,title=None,auto_close=True) [Function]

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

### load(a,b) [Function]
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
### save(a,b) [Function]
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
### count_parameters(model_or_dict_or_param) [Function]

Count parameter numer of a module/state_dict/layer/tensor.
    This function can also print the occupied memory of parameters in MBs
    
Arguements:
* model_or_dict_or_param: model or state dictionary or parameters()

Return: parameter amount in python-int
        Returns 0 if datatype not understood

Usage:
```python
    count_parameters(model)
    count_parameters(state_dict) #all params 
    count_parameters(model.parameters()) #only trainable params
    count_parameters(weight_tensor)
    count_parameters(numpy_array)
```


-----------------
### Packsearch [Module]

This is a very useful thing you definitly have been dreaming of.

You can now use packsearch to query names inside any package! 

Given an module object as input:

       > p = Packsearch(torch) 
    or > p = Packsarch(numpy) whatever

the instance `p` provide `p.search()` method. So that you can 
search everything inside this package

    > p.search('maxpoo')

output:

    Packsearch: 35 results found:
    -------------results start-------------
    0        torch.nn.AdaptiveMaxPool1d
    1        torch.nn.AdaptiveMaxPool2d
    2        torch.nn.AdaptiveMaxPool3d
    3        torch.nn.FractionalMaxPool2d
    4        torch.nn.MaxPool1d
    5        torch.nn.MaxPool2d
    ...

-----------------
### packsearch(module,keyword) [Function]
or packsearch(keyword,module)

Given an module object, and search pattern string as input:

    > packsearch(torch,'maxpoo')

or

    > packsearch('maxpoo',torch)
    
    output:
        Packsearch: 35 results found:
        -------------results start-------------
        0        torch.nn.AdaptiveMaxPool1d
        1        torch.nn.AdaptiveMaxPool2d
        2        torch.nn.AdaptiveMaxPool3d
        3        torch.nn.FractionalMaxPool2d
        4        torch.nn.MaxPool1d
        5        torch.nn.MaxPool2d
        ... 

you can search for everything inside any package

------------------
### hash_parameters(module_or_statdict_or_param)  [Function]

return the summary of all variables.

This is used to detect chaotic changes of weights.
You can check the sum_parameters before and after some operations, to know
if there is any change made to the params.

I use this function to verify gradient behaviours.

By default, This only hash the trainable parameters!

arguements:

* module_or_statdict_or_param: torch.nn.module, 
                    or model.state_dict(), 
                    or model.parameters().
* use_sum: return the sum instead of mean value of all params.

Usage demo:

```python
model = MyNet()
print(hash_parameters(model)) # see params
train_one_step(model)
print(hash_parameters(model)) # see if params are updated
print(hash_parameters(model.state_dict())) # see if trainable+un-trainable params are updated
```

-----------------
### force_exist(dirname,verbose=True) [Function]

force a series of hierachical directories to exist.

`force_exist` can automatically create directory with any depth.

Arguements:

* dirname: path of the desired directory
* verbose: print every directory creation. default True.

Usage:

    force_exist('a/b/c/d/e/f')
    force_exist('a/b/c/d/e/f',verbose=False)


------------------
### sort_args(args_or_types,types_or_args) [Function]

This is a very interesting function.
It is used to support __arbitrary-arguments-ordering__ in TorchFun.

Input:
    The function takes a list of types, and a list of arguments.

Returns:
    a list of arguments, with the same order as the types-list.

Of course, `sort_args` supports arbitrary-arguments-ordering by itself.


-------------------
### whereis(module_or_string) [Function]

find the source file location of a module
arguments:

*    module_or_string: target module object, or it's string path like `torch.nn`
*    open_gui: open the folder with default window-manager.

returns:

*    module file name, or None
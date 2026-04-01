# **torch-PoL**

This small library implements an efficient Pyramid of Laplacian (PoL) Datatype in pure Pytorch.

This datatype is meant to be used as a replcaement for tensors that need to be optimized.
For instance instead of optimizing a voxelgrid or and image we can optimize an underlying PoL.
It usually converges much faster and is less noisy.

This library supports 1D,2D and 3D data with batch and channel dimension.

## **Usage**

A full test script is provided in `test.py`. To run it, first install the requirements:
```
pip install -r requirements.txt
python test.py
```
You can also run the test with `--dim {1,2 or 3}` to run only a certain dimension.

### **Initialization**

To create a PoL you can call:
```
from torchpol import PoL
pol = PoL(
        init_tensor,
        levels=-1, 
        reprojection_interval=16,
        learning_rate=3e-3,
        weight_decay=0.0,
        padding_mode="reflect",
    )
```

Explanation of the inputs:

- `init_tensor` - Required: Tensor used to initialize the PoL, format must be NCW (1D signal), NCHW (2D signal) or NCTHW (3D signal). The size of the tensor will be used to define the PoL shape. The PoL is automatically transferred to the same device as the init tensor.
- `levels=-1` - The number of desired levels in the PoL. Default is -1 and compute the maximum level that can be used automatically. Using levels=1 is equivalent to optimizing directly the signal (an Image for instance in 2D)
- `reprojection_interval=16` - Because we optimize the levels of a PoL directly, there is no guarantee that, after an optimization step, these levels are indeed the PoL of the reconstructed signal.
- `learning_rate=3e-3` - Learning rate of the PoL parameters.
- `weight_decay=0.0` - Weight decay to apply to the parameters of the PoL. `/!\` If using this parameter it is recommended to use the **AdamW** optimizer instead of Adam. 0.05 seems to work well to regularize. The weight decay is not applied to the lowest level.
- `padding_mode="reflect"` - Padding mode to be used during up and down sampling the PoL
- `graphed=True`- Wether to use a torch.cuda.graph for the inverse laplacian, improves performance, Default is True. Note profiling might not work with this mode in pytorch 2.0.1 (pytorch 2.1.0 nightly seem to work)

As you can see the `learning_rate` and `weight_decay` need to be provided when building the object. This simplifies the handling of the weight_decay. To provide the parameters of the PoL to an optimizer simply do:

```
pol = PoL(...)
params = pol.parameters()
optimizer = torch.optim.AdamW(params, betas=(0.9, 0.99),fused=True)

# Note that params is a list of dictionaries and if combined with other parameters they would need to be converted to a list and concatenated with `+`
# optimizer = torch.optim.AdamW(params+list(model.parameters()), betas=(0.9, 0.99),fused=True)
```


### **Obtaining the reconstruction**

To obtain the reconstruction of a signal simply call the forward function:

```
recon = pol()
```

Note that in order to make things as efficient as possible the PoL object holds an internal representation of the reconstruction such that the reconstruction is reconstructed on the first call to `pol()` but the stored version is used for subsequent ones.
This allows using the PoL several time during a single iteration without recomputing the recontruction from the parameters several time and also simplifies the computation graph.



### **End of iteration**

At the end of the iteration we want to perform two task with the PoL:
- Clean the internal representation of the reconstruction
- Reproject the representation to a valid PoL (periodically, see `reprojection_interval` input)

Both these tasks are taken care of by the PoL `end_iter_callback` that needs to be called after the `optimizer.step()`:
```
pol.end_iter_callback(i)
```



### **A typical training loop**

Here is a typical setup involving the PoL:

```
pol = PoL(torch.zeros(1,3,512,512)) #Create 2D PoL
pol = pol.cuda()

params = pol.parameters()

optimizer = torch.optim.AdamW(params, betas=(0.9, 0.99),fused=True)

for i in tqdm(range(STEPS)):

    optimizer.zero_grad(set_to_none=True)
    
    recon = pol()

    ... Perform computation using the recon and obtain a loss ...

    loss.backward()
    optimizer.step()
    
    pol.end_iter_callback(i)
```

The same structure can be found in `test.py` where we optimize the PoL using only 1pc or 0.1pc of the signal. This files provides example for 1D, 2D and 3D data.

### **Obtaining other view of the reconstruction or representation**

Visualization routine and usages are provided in `visualize_{1,2,3}d` int `test.py`.

#### **Levels of the PoL**
The parameters of the PoL are the levels themselves. They can be accessed easily using:
```
levels = pol.levels
```
Each level as a different size.

![Levels of the PoL](doc/decomposed.png)

#### **MipMap levels of the PoL**
Here again each level has a different size but is composed with the previous levels leading to a mipmap like representation:
```
mipmaps = pol.mip_lvls()
```
A list of tensor is returned. This function is differentiable.

![MipMaps](doc/mipmaps.png)

#### **Composed levels of the PoL**
This time the function returns a Tensor with one more dimension than the PoL, which (in the 2D case) can be trilinearly interpolated.
Each of the mipmap levels are upsampled to the highest resolution so that we obtain a stack of tensors.
```
composed = pol.composed_lvls()
```
A single tensor is returned. This function is differentiable.

![Composed levels](doc/composed_lvls.png)


## **Performance**

This representation should be highly performant allowing up to 500it/s on a signle A100 for [3,512,768] 2D or [1,128,128,128] 3D data using the maximum number of levels.
The test script should allow for such performances when profiling is disabled.

The number of levels plays a big role in the performance as each level adds some cpu overhead (which is dominating quickly as resolutioin decreases). It might be beneficial to use a smaller number of levels if performance is critical.

To be very efficient both scriptin and torch.cuda.graphs are used reducing the overhead to a minimum. With this for a [3,512,768] image both the forward and backward take about 160μs (0.160ms).

The library uses separable transposed convolutions for 2D and 3D upsampling.

In `test.py` one can see that flags are setup this way for optimal performance on A100:
```
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
```
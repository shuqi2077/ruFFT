//! Device-only spectral layout and linear adjoint helpers.
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{backtrace::BackTrace, calculate_ruda_count_elemwise, prelude::*};
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_contiguous_dtype};

fn invalid(reason: &str) -> LaunchError {
    LaunchError::Unknown { reason: reason.into(), backtrace: BackTrace::capture() }
}
trait HostFloat: Float + RudaElement { fn scalar(value: f64) -> Self; }
impl HostFloat for f32 { fn scalar(value: f64) -> Self { value as f32 } }
impl HostFloat for f64 { fn scalar(value: f64) -> Self { value } }
fn check<R: Runtime>(x: &RudaTensor<R>, shape: &Shape) -> Result<usize, LaunchError> {
    if !matches!(x.dtype, DType::F32 | DType::F64) || x.qparams.is_some()
        || shape.len() != x.meta.num_dims() {
        return Err(invalid("spectral layout requires same-rank F32/F64 tensors"));
    }
    shape.iter().try_fold(1usize, |n, &s| n.checked_mul(s))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or_else(|| invalid("spectral layout exceeds U32 indexing"))
}
/// Leading-coordinate crop or trailing zero padding on any axes. No signal
/// values leave the execution queue. The same operation is the padding adjoint.
pub fn resize<R: Runtime>(input: &RudaTensor<R>, shape: Shape) -> Result<RudaTensor<R>, LaunchError> {
    let count = check(input, &shape)?;
    let out = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), shape, input.dtype);
    if count > 0 {
        let d = RudaDim::new(input.client.properties(), count);
        match input.dtype {
            DType::F32 => resize_kernel::launch::<f32,R>(&input.client, calculate_ruda_count_elemwise(&input.client,count,d),d,
                input.clone().into_tensor_arg(),out.clone().into_tensor_arg()),
            DType::F64 => resize_kernel::launch::<f64,R>(&input.client, calculate_ruda_count_elemwise(&input.client,count,d),d,
                input.clone().into_tensor_arg(),out.clone().into_tensor_arg()),
            _ => unreachable!(),
        }
    }
    Ok(out)
}
pub fn zeros<R:Runtime>(like:&RudaTensor<R>,shape:Shape)->Result<RudaTensor<R>,LaunchError> {
    let count=check(like,&shape)?;
    let output=empty_device_contiguous_dtype(like.client.clone(),like.device.clone(),shape,like.dtype);
    if count>0 {
        let d=RudaDim::new(like.client.properties(),count);
        let grid=calculate_ruda_count_elemwise(&like.client,count,d);
        match like.dtype {
            DType::F32=>zero_kernel::launch::<f32,R>(&like.client,grid,d,output.clone().into_array_arg()),
            DType::F64=>zero_kernel::launch::<f64,R>(&like.client,grid,d,output.clone().into_array_arg()),
            _=>unreachable!(),
        }
    }
    Ok(output)
}
/// Scale, conjugate, or weight the interior bins of a one-sided spectrum.
/// `interior_axis=None` scales every element; otherwise DC and the even-length
/// Nyquist bin retain weight one and all other bins receive `interior_weight`.
pub fn scale<R: Runtime>(input: &RudaTensor<R>, factor: f64,
    interior_axis: Option<(usize,usize,f64)>) -> Result<RudaTensor<R>, LaunchError> {
    let count=check(input,input.meta.shape())?;
    let (axis,n,weight)=interior_axis.unwrap_or((0,0,1.0));
    if interior_axis.is_some() && (axis>=input.meta.num_dims() || n==0) {
        return Err(invalid("invalid one-sided spectral axis/length"));
    }
    let out=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),input.meta.shape().clone(),input.dtype);
    if count>0 {
        let d=RudaDim::new(input.client.properties(),count);
        match input.dtype {
            DType::F32 => scale_kernel::launch::<f32,R>(&input.client,calculate_ruda_count_elemwise(&input.client,count,d),d,
                input.clone().into_tensor_arg(),out.clone().into_array_arg(),<f32 as HostFloat>::scalar(factor),
                <f32 as HostFloat>::scalar(weight),axis,n,interior_axis.is_some()),
            DType::F64 => scale_kernel::launch::<f64,R>(&input.client,calculate_ruda_count_elemwise(&input.client,count,d),d,
                input.clone().into_tensor_arg(),out.clone().into_array_arg(),<f64 as HostFloat>::scalar(factor),
                <f64 as HostFloat>::scalar(weight),axis,n,interior_axis.is_some()),
            _=>unreachable!(),
        }
    }
    Ok(out)
}
/// Enforce the real-only DC and even-length Nyquist bins immediately after a
/// real transform, before complex transforms on the other axes.
pub fn real_spectrum_imaginary<R:Runtime>(input:&RudaTensor<R>,axis:usize,n:usize)
    ->Result<RudaTensor<R>,LaunchError> {
    let count=check(input,input.meta.shape())?;
    if axis>=input.meta.num_dims() || n==0 {return Err(invalid("invalid real spectrum boundary axis"));}
    let output=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),input.meta.shape().clone(),input.dtype);
    if count>0 {
        let dim=RudaDim::new(input.client.properties(),count);
        let grid=calculate_ruda_count_elemwise(&input.client,count,dim);
        match input.dtype {
            DType::F32=>boundary_kernel::launch::<f32,R>(&input.client,grid,dim,
                input.clone().into_tensor_arg(),output.clone().into_array_arg(),axis,n),
            DType::F64=>boundary_kernel::launch::<f64,R>(&input.client,grid,dim,
                input.clone().into_tensor_arg(),output.clone().into_array_arg(),axis,n),
            _=>unreachable!(),
        }
    }
    Ok(output)
}
/// Equivalent to the ordered sequence of cyclic rolls used by fftshift.
/// Repeated axes are deliberately retained rather than de-duplicated.
pub fn shift<R: Runtime>(input:&RudaTensor<R>,axes:&[usize],inverse:bool)
    -> Result<RudaTensor<R>,LaunchError> {
    check(input,input.meta.shape())?;
    let mut value=input.clone();
    for &axis in axes {
        if axis>=input.meta.num_dims() { return Err(invalid("spectral shift axis out of range")); }
        let count=value.meta.num_elements();
        let n=value.meta.shape()[axis];
        let out=empty_device_contiguous_dtype(value.client.clone(),value.device.clone(),value.meta.shape().clone(),value.dtype);
        if count>0 {
            let d=RudaDim::new(value.client.properties(),count);
            let source_shift=if inverse { n/2 } else { n-n/2 };
            match value.dtype {
                DType::F32=>shift_kernel::launch::<f32,R>(&value.client,calculate_ruda_count_elemwise(&value.client,count,d),d,
                    value.clone().into_tensor_arg(),out.clone().into_tensor_arg(),axis,source_shift),
                DType::F64=>shift_kernel::launch::<f64,R>(&value.client,calculate_ruda_count_elemwise(&value.client,count,d),d,
                    value.clone().into_tensor_arg(),out.clone().into_tensor_arg(),axis,source_shift),
                _=>unreachable!(),
            }
        }
        value=out;
    }
    Ok(value)
}
pub fn fftshift<R:Runtime>(input:&RudaTensor<R>,axes:&[usize])->Result<RudaTensor<R>,LaunchError> {
    shift(input,axes,false)
}
pub fn ifftshift<R:Runtime>(input:&RudaTensor<R>,axes:&[usize])->Result<RudaTensor<R>,LaunchError> {
    shift(input,axes,true)
}
pub fn fftfreq<R:Runtime>(like:&RudaTensor<R>,n:usize,distance:f64)->Result<RudaTensor<R>,LaunchError> {
    frequency_tensor(like,n,distance,false)
}
pub fn rfftfreq<R:Runtime>(like:&RudaTensor<R>,n:usize,distance:f64)->Result<RudaTensor<R>,LaunchError> {
    frequency_tensor(like,n,distance,true)
}
fn frequency_tensor<R:Runtime>(like:&RudaTensor<R>,n:usize,distance:f64,onesided:bool)
    ->Result<RudaTensor<R>,LaunchError> {
    if n>u32::MAX as usize || !matches!(like.dtype,DType::F32|DType::F64) {
        return Err(invalid("frequency tensor requires U32 length and F32/F64 dtype"));
    }
    let output=empty_device_contiguous_dtype(like.client.clone(),like.device.clone(),
        Shape::from([if onesided {n/2+1} else {n}]),like.dtype);
    frequencies(&output,n,distance,onesided)?;
    Ok(output)
}
/// Fill an allocated F32/F64 vector with the canonical FFT frequency layout.
pub fn frequencies<R:Runtime>(output:&RudaTensor<R>,n:usize,distance:f64,onesided:bool)
    -> Result<(),LaunchError> {
    check(output,output.meta.shape())?;
    if output.meta.num_dims()!=1 || output.meta.num_elements()!=if onesided { n/2+1 } else { n } {
        return Err(invalid("invalid FFT frequency output shape/length"));
    }
    let count=output.meta.num_elements();
    if count==0 {return Ok(());}
    let dim=RudaDim::new(output.client.properties(),count);
    let grid=calculate_ruda_count_elemwise(&output.client,count,dim);
    match output.dtype {
        DType::F32=>frequency_kernel::launch::<f32,R>(&output.client,grid,dim,output.clone().into_array_arg(),
            <f32 as HostFloat>::scalar(1.0/(n as f64*distance)),n,onesided),
        DType::F64=>frequency_kernel::launch::<f64,R>(&output.client,grid,dim,output.clone().into_array_arg(),
            <f64 as HostFloat>::scalar(1.0/(n as f64*distance)),n,onesided),
        _=>unreachable!(),
    }
    Ok(())
}

#[ruda(launch)]
fn resize_kernel<F:Float>(input:&Tensor<F>,output:&mut Tensor<F>) {
    let pos=ABSOLUTE_POS;
    if pos>=output.len() { terminate!(); }
    let mut rem=pos;
    let mut offset=0usize;
    let mut valid=true;
    let mut axis=output.rank();
    while axis>0 {
        axis-=1;
        let c=rem%output.shape(axis); rem/=output.shape(axis);
        if c>=input.shape(axis) { valid=false; }
        offset+=c*input.stride(axis);
    }
    output[pos]=if valid { input[offset] } else { F::new(0.0) };
}
#[ruda(launch)]
fn zero_kernel<F:Float>(output:&mut Array<F>) {
    let pos=ABSOLUTE_POS;
    if pos<output.len() { output[pos]=F::new(0.0); }
}
#[ruda(launch)]
fn boundary_kernel<F:Float>(input:&Tensor<F>,output:&mut Array<F>,#[comptime] axis:usize,#[comptime] n:usize) {
    let pos=ABSOLUTE_POS;if pos>=output.len() {terminate!();}
    let mut rem=pos;let mut offset=0usize;let mut bin=0usize;let mut dim=input.rank();
    while dim>0 {
        dim-=1;let c=rem%input.shape(dim);rem/=input.shape(dim);
        offset+=c*input.stride(dim);if dim==axis {bin=c;}
    }
    output[pos]=if bin==0 || (n%2==0 && bin==n/2) {F::new(0.0)} else {input[offset]};
}
#[ruda(launch)]
fn scale_kernel<F:Float + RudaElement>(input:&Tensor<F>,output:&mut Array<F>,factor:F,weight:F,
    #[comptime] axis:usize,#[comptime] n:usize,#[comptime] interior:bool) {
    let pos=ABSOLUTE_POS;
    if pos>=output.len() { terminate!(); }
    let mut rem=pos; let mut offset=0usize; let mut bin=0usize; let mut dim=input.rank();
    while dim>0 {
        dim-=1;
        let c=rem%input.shape(dim); rem/=input.shape(dim);
        offset+=c*input.stride(dim);
        if dim==axis { bin=c; }
    }
    let mut multiplier=factor;
    if interior && bin>0 && bin<=(n-1)/2 { multiplier*=weight; }
    output[pos]=input[offset]*multiplier;
}
#[ruda(launch)]
fn shift_kernel<F:Float>(input:&Tensor<F>,output:&mut Tensor<F>,#[comptime] axis:usize,#[comptime] shift:usize) {
    let pos=ABSOLUTE_POS;
    if pos>=output.len() { terminate!(); }
    let mut rem=pos; let mut offset=0usize; let mut dim=input.rank();
    while dim>0 {
        dim-=1;
        let mut c=rem%input.shape(dim); rem/=input.shape(dim);
        if dim==axis { c=(c+shift)%input.shape(dim); }
        offset+=c*input.stride(dim);
    }
    output[pos]=input[offset];
}
#[ruda(launch)]
fn frequency_kernel<F:Float + RudaElement>(output:&mut Array<F>,step:F,#[comptime] n:usize,#[comptime] onesided:bool) {
    let k=ABSOLUTE_POS;
    if k>=output.len() { terminate!(); }
    let frequency=if onesided || k<(n+1)/2 { F::cast_from(k) } else { -F::cast_from(n-k) };
    output[k]=frequency*step;
}

use crate::{FftMode, RealFftPlan, double::DoubleFftPlan};
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{backtrace::BackTrace, calculate_ruda_count_elemwise, prelude::*};
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_contiguous_dtype};

trait HostFloat: Float + RudaElement {
    fn scalar(value: f64) -> Self;
}
impl HostFloat for f32 {
    fn scalar(value: f64) -> Self {
        value as f32
    }
}
impl HostFloat for f64 {
    fn scalar(value: f64) -> Self {
        value
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FftNormalization {
    None,
    Backward,
    Forward,
    Ortho,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NdFftKind {
    ComplexForward,
    ComplexInverse,
    RealForward,
    RealInverse,
}
fn invalid(s: &str) -> LaunchError {
    LaunchError::Unknown {
        reason: s.into(),
        backtrace: BackTrace::capture(),
    }
}
fn allocate<R: Runtime>(like: &RudaTensor<R>, shape: Shape) -> RudaTensor<R> {
    empty_device_contiguous_dtype(like.client.clone(), like.device.clone(), shape, like.dtype)
}
fn validate<R: Runtime>(t: &RudaTensor<R>) -> Result<(), LaunchError> {
    if !matches!(t.dtype, DType::F32 | DType::F64) || t.qparams.is_some() || t.meta.num_dims() == 0
    {
        return Err(invalid("FFT requires non-scalar unquantized F32/F64"));
    }
    let count = t
        .meta
        .shape()
        .iter()
        .try_fold(1usize, |n, &s| n.checked_mul(s))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or_else(|| invalid("FFT element count exceeds device indexing"))?;
    if count != 0 {
        let span = t
            .meta
            .shape()
            .iter()
            .zip(t.meta.strides().iter())
            .try_fold(0usize, |n, (&s, &st)| {
                (s - 1).checked_mul(st).and_then(|k| n.checked_add(k))
            })
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(if t.dtype == DType::F64 { 8 } else { 4 }))
            .ok_or_else(|| invalid("FFT tensor span overflow"))?;
        let available = t
            .handle
            .size()
            .checked_sub(t.handle.offset_start.unwrap_or(0))
            .and_then(|n| n.checked_sub(t.handle.offset_end.unwrap_or(0)))
            .ok_or_else(|| invalid("FFT allocation offsets invalid"))?;
        if span as u64 > available {
            return Err(invalid("FFT layout exceeds backing allocation"));
        }
    }
    Ok(())
}

enum Axis<R: Runtime> {
    Single {
        plan: RealFftPlan<R>,
        half: Option<[RudaTensor<R>; 4]>,
    },
    Double(DoubleFftPlan<R>),
}
struct Stage<R: Runtime> {
    input_shape: Shape,
    axis: usize,
    length: usize,
    real: bool,
    plan: Axis<R>,
    output: [RudaTensor<R>; 2],
}
/// Reusable exact-length multi-axis plan. Axes are unique and ordered; lengths
/// virtually pad or truncate each corresponding dimension. Arbitrary valid
/// positive-stride input layouts are accepted; outputs are contiguous.
/// The last listed axis is the Hermitian-compressed axis for real transforms.
/// Workspace/tables are owned by the plan and queue-bound. Returned outputs
/// are independent of subsequent calls; the plan is not concurrently usable.
pub struct NdFftPlan<R: Runtime> {
    shape: Shape,
    dtype: DType,
    kind: NdFftKind,
    axes: Vec<usize>,
    lengths: Vec<usize>,
    stages: Vec<Stage<R>>,
    scale: f64,
}
impl<R: Runtime> NdFftPlan<R> {
    pub fn new(
        like: &RudaTensor<R>,
        axes: &[usize],
        lengths: &[usize],
        kind: NdFftKind,
        norm: FftNormalization,
    ) -> Result<Self, LaunchError> {
        validate(like)?;
        if axes.is_empty()
            || axes.len() != lengths.len()
            || lengths.contains(&0)
            || lengths.iter().any(|&n| n > u32::MAX as usize)
            || axes.iter().any(|&a| a >= like.meta.num_dims())
            || axes.iter().enumerate().any(|(i, a)| axes[..i].contains(a))
        {
            return Err(invalid(
                "nonempty unique valid axes and matching positive lengths required",
            ));
        }
        let product = lengths
            .iter()
            .try_fold(1usize, |n, &s| n.checked_mul(s))
            .ok_or_else(|| invalid("multidimensional transform volume overflow"))?;
        let inverse = matches!(kind, NdFftKind::ComplexInverse | NdFftKind::RealInverse);
        let real = matches!(kind, NdFftKind::RealForward | NdFftKind::RealInverse);
        let last = axes.len() - 1;
        let scale = match norm {
            FftNormalization::None => 1.0,
            FftNormalization::Backward => {
                if inverse {
                    1.0 / product as f64
                } else {
                    1.0
                }
            }
            FftNormalization::Forward => {
                if inverse {
                    1.0
                } else {
                    1.0 / product as f64
                }
            }
            FftNormalization::Ortho => 1.0 / (product as f64).sqrt(),
        };
        let mut sequence: Vec<usize> = if real && !inverse {
            std::iter::once(last).chain(0..last).collect()
        } else if real && inverse {
            (0..last).rev().chain(std::iter::once(last)).collect()
        } else {
            (0..axes.len()).collect()
        };
        // Every complex stage is unnormalized. Existing IRFFT normalizes its
        // own axis, so compensate once at the final requested normalization.
        let scale = scale
            * if kind == NdFftKind::RealInverse {
                lengths[last] as f64
            } else {
                1.0
            };
        let mut model = like.clone();
        if kind == NdFftKind::RealInverse {
            let mut padded_shape = like.meta.shape().clone();
            padded_shape[axes[last]] = lengths[last] / 2 + 1;
            if &padded_shape != like.meta.shape() {
                model = allocate(like, padded_shape);
            }
        }
        let mut stages = Vec::with_capacity(sequence.len());
        for index in sequence.drain(..) {
            let axis = axes[index];
            let length = lengths[index];
            let is_real = real && index == last;
            let mut output_shape = model.meta.shape().clone();
            output_shape[axis] = if is_real && !inverse {
                length / 2 + 1
            } else {
                length
            };
            output_shape
                .iter()
                .try_fold(1usize, |n, &s| n.checked_mul(s))
                .filter(|&n| n <= u32::MAX as usize)
                .ok_or_else(|| invalid("FFT stage output exceeds device indexing"))?;
            let output = [
                allocate(&model, output_shape.clone()),
                allocate(&model, output_shape),
            ];
            let plan = if like.dtype == DType::F64 {
                Axis::Double(DoubleFftPlan::new(
                    &model,
                    axis,
                    length,
                    if inverse {
                        FftMode::Inverse
                    } else {
                        FftMode::Forward
                    },
                )?)
            } else {
                let mut half_shape = model.meta.shape().clone();
                half_shape[axis] = length / 2 + 1;
                Axis::Single {
                    plan: RealFftPlan::new(
                        model.client.clone(),
                        length,
                        if is_real && inverse {
                            FftMode::Inverse
                        } else {
                            FftMode::Forward
                        },
                    )?,
                    half: if is_real {
                        None
                    } else {
                        Some(std::array::from_fn(|_| {
                            allocate(&model, half_shape.clone())
                        }))
                    },
                }
            };
            stages.push(Stage {
                input_shape: model.meta.shape().clone(),
                axis,
                length,
                real: is_real,
                plan,
                output,
            });
            model = stages.last().unwrap().output[0].clone();
        }
        Ok(Self {
            shape: like.meta.shape().clone(),
            dtype: like.dtype,
            kind,
            axes: axes.to_vec(),
            lengths: lengths.to_vec(),
            stages,
            scale,
        })
    }
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }
    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }
    pub fn output_shape(&self) -> &Shape {
        self.stages.last().unwrap().output[0].meta.shape()
    }
    /// Retained GPU storage, including axis tables, scratch and stage buffers.
    /// F32 Bluestein scratch is allocated lazily; this number updates after use.
    /// Double execution additionally has transient axis output allocations.
    pub fn retained_bytes(&self) -> usize {
        self.stages
            .iter()
            .map(|s| {
                s.output
                    .iter()
                    .map(|t| {
                        t.meta.shape().iter().product::<usize>()
                            * if self.dtype == DType::F64 { 8 } else { 4 }
                    })
                    .sum::<usize>()
                    + match &s.plan {
                        Axis::Double(p) => p.retained_bytes(),
                        Axis::Single { plan, half } => {
                            plan.retained_bytes()
                                + half
                                    .iter()
                                    .flatten()
                                    .map(|t| t.meta.shape().iter().product::<usize>() * 4)
                                    .sum::<usize>()
                        }
                    }
            })
            .sum()
    }
    /// Release lazily allocated F32 Bluestein scratch. Tables and fixed stage
    /// buffers remain; queued consumers are protected by runtime ordering.
    pub fn clear_lazy_workspace(&mut self) {
        for stage in &mut self.stages {
            if let Axis::Single { plan, .. } = &mut stage.plan {
                plan.clear_workspace();
            }
        }
    }
    pub fn execute_complex(
        &mut self,
        real: &RudaTensor<R>,
        imag: &RudaTensor<R>,
    ) -> Result<(RudaTensor<R>, RudaTensor<R>), LaunchError> {
        if !matches!(
            self.kind,
            NdFftKind::ComplexForward | NdFftKind::ComplexInverse
        ) {
            return Err(invalid("execute_complex requires a complex plan"));
        }
        self.run(real, Some(imag))
    }
    pub fn forward_real(
        &mut self,
        signal: &RudaTensor<R>,
    ) -> Result<(RudaTensor<R>, RudaTensor<R>), LaunchError> {
        if self.kind != NdFftKind::RealForward {
            return Err(invalid("forward_real requires RealForward plan"));
        }
        self.run(signal, None)
    }
    pub fn inverse_real(
        &mut self,
        real: &RudaTensor<R>,
        imag: &RudaTensor<R>,
    ) -> Result<RudaTensor<R>, LaunchError> {
        if self.kind != NdFftKind::RealInverse {
            return Err(invalid("inverse_real requires RealInverse plan"));
        }
        Ok(self.run(real, Some(imag))?.0)
    }
    fn run(
        &mut self,
        real: &RudaTensor<R>,
        imag: Option<&RudaTensor<R>>,
    ) -> Result<(RudaTensor<R>, RudaTensor<R>), LaunchError> {
        let queue = &self.stages[0].output[0].client;
        for t in std::iter::once(real).chain(imag) {
            validate(t)?;
            if t.dtype != self.dtype
                || t.meta.shape() != &self.shape
                || !t.client.same_execution_queue(queue)
            {
                return Err(invalid("FFT input differs from plan dtype, shape or queue"));
            }
        }
        let inverse = matches!(
            self.kind,
            NdFftKind::ComplexInverse | NdFftKind::RealInverse
        );
        let mut re = real.clone();
        let mut im = imag.cloned();
        if self.kind == NdFftKind::RealInverse && re.meta.shape() != &self.stages[0].input_shape {
            re = crate::spectral_layout::resize(&re, self.stages[0].input_shape.clone())?;
            im = Some(crate::spectral_layout::resize(im.as_ref().unwrap(), self.stages[0].input_shape.clone())?);
        }
        for stage in &mut self.stages {
            if re.meta.shape() != &stage.input_shape {
                return Err(invalid("FFT stage shape mismatch"));
            }
            match &mut stage.plan {
                Axis::Double(plan) => {
                    let pair = if stage.real {
                        plan.run(
                            &re,
                            im.as_ref(),
                            inverse,
                            if inverse {
                                stage.length
                            } else {
                                stage.length / 2 + 1
                            },
                            inverse,
                        )?
                    } else {
                        plan.execute(
                            &re,
                            im.as_ref()
                                .ok_or_else(|| invalid("complex stage missing imaginary input"))?,
                            false,
                        )?
                    };
                    copy::<f64, R>(&pair.0, &stage.output[0], 1.0);
                    if stage.real && !inverse {
                        let imag = crate::spectral_layout::real_spectrum_imaginary(&pair.1,stage.axis,stage.length)?;
                        copy::<f64, R>(&imag, &stage.output[1], 1.0);
                    } else {
                        copy::<f64, R>(&pair.1, &stage.output[1], 1.0);
                    }
                }
                Axis::Single { plan, half } => {
                    if stage.real && !inverse {
                        plan.forward(
                            re.clone().binding(),
                            stage.output[0].clone().binding(),
                            stage.output[1].clone().binding(),
                            stage.axis,
                            re.meta.shape()[stage.axis].min(stage.length),
                        )?;
                    } else if stage.real {
                        plan.inverse(
                            re.clone().binding(),
                            im.as_ref().unwrap().clone().binding(),
                            stage.output[0].clone().binding(),
                            stage.axis,
                            stage.length / 2 + 1,
                        )?;
                        zero::<f32, R>(&stage.output[1]);
                    } else {
                        let half = half.as_ref().unwrap();
                        let imag = im
                            .as_ref()
                            .ok_or_else(|| invalid("complex stage missing imaginary input"))?;
                        plan.forward(
                            re.clone().binding(),
                            half[0].clone().binding(),
                            half[1].clone().binding(),
                            stage.axis,
                            re.meta.shape()[stage.axis].min(stage.length),
                        )?;
                        plan.forward(
                            imag.clone().binding(),
                            half[2].clone().binding(),
                            half[3].clone().binding(),
                            stage.axis,
                            imag.meta.shape()[stage.axis].min(stage.length),
                        )?;
                        let count = stage.output[0].meta.shape().iter().product::<usize>();
                        let d = RudaDim::new(re.client.properties(), count.max(1));
                        if count > 0 {
                            combine::launch::<R>(
                                &re.client,
                                calculate_ruda_count_elemwise(&re.client, count, d),
                                d,
                                half[0].clone().into_tensor_arg(),
                                half[1].clone().into_tensor_arg(),
                                half[2].clone().into_tensor_arg(),
                                half[3].clone().into_tensor_arg(),
                                stage.output[0].clone().into_tensor_arg(),
                                stage.output[1].clone().into_tensor_arg(),
                                stage.axis,
                                stage.length,
                                inverse,
                            );
                        }
                    }
                }
            }
            re = stage.output[0].clone();
            im = Some(stage.output[1].clone());
        }
        let out_re = allocate(&re, re.meta.shape().clone());
        let out_im = allocate(&re, re.meta.shape().clone());
        match self.dtype {
            DType::F32 => {
                copy::<f32, R>(&re, &out_re, self.scale);
                copy::<f32, R>(im.as_ref().unwrap(), &out_im, self.scale);
            }
            DType::F64 => {
                copy::<f64, R>(&re, &out_re, self.scale);
                copy::<f64, R>(im.as_ref().unwrap(), &out_im, self.scale);
            }
            _ => unreachable!(),
        }
        Ok((out_re, out_im))
    }
}

pub fn fft2<R: Runtime>(
    real: &RudaTensor<R>,
    imag: &RudaTensor<R>,
    axes: [usize; 2],
    lengths: [usize; 2],
    inverse: bool,
    norm: FftNormalization,
) -> Result<(RudaTensor<R>, RudaTensor<R>), LaunchError> {
    NdFftPlan::new(
        real,
        &axes,
        &lengths,
        if inverse {
            NdFftKind::ComplexInverse
        } else {
            NdFftKind::ComplexForward
        },
        norm,
    )?
    .execute_complex(real, imag)
}
pub fn fft3<R: Runtime>(
    real: &RudaTensor<R>,
    imag: &RudaTensor<R>,
    axes: [usize; 3],
    lengths: [usize; 3],
    inverse: bool,
    norm: FftNormalization,
) -> Result<(RudaTensor<R>, RudaTensor<R>), LaunchError> {
    NdFftPlan::new(
        real,
        &axes,
        &lengths,
        if inverse {
            NdFftKind::ComplexInverse
        } else {
            NdFftKind::ComplexForward
        },
        norm,
    )?
    .execute_complex(real, imag)
}
pub fn rfftn<R: Runtime>(
    signal: &RudaTensor<R>,
    axes: &[usize],
    lengths: &[usize],
    norm: FftNormalization,
) -> Result<(RudaTensor<R>, RudaTensor<R>), LaunchError> {
    NdFftPlan::new(signal, axes, lengths, NdFftKind::RealForward, norm)?.forward_real(signal)
}
pub fn irfftn<R: Runtime>(
    real: &RudaTensor<R>,
    imag: &RudaTensor<R>,
    axes: &[usize],
    lengths: &[usize],
    norm: FftNormalization,
) -> Result<RudaTensor<R>, LaunchError> {
    NdFftPlan::new(real, axes, lengths, NdFftKind::RealInverse, norm)?.inverse_real(real, imag)
}

/// Adjoint normalization used by Hermitian and inverse-Hermitian transforms.
pub fn reverse_normalization(norm: FftNormalization) -> FftNormalization {
    match norm {
        FftNormalization::Backward => FftNormalization::Forward,
        FftNormalization::Forward => FftNormalization::Backward,
        FftNormalization::None => FftNormalization::None,
        FftNormalization::Ortho => FftNormalization::Ortho,
    }
}

/// N-dimensional FFT of a Hermitian-symmetric signal. The final listed axis
/// is supplied one-sided; explicit lengths distinguish odd and even signals.
pub fn hfftn<R: Runtime>(real: &RudaTensor<R>, imag: &RudaTensor<R>,
    axes: &[usize], lengths: &[usize], norm: FftNormalization)
    -> Result<RudaTensor<R>, LaunchError> {
    let conjugated = crate::spectral_layout::scale(imag, -1.0, None)?;
    irfftn(real, &conjugated, axes, lengths, reverse_normalization(norm))
}

/// Inverse Hermitian transform, returning the compressed final signal axis.
pub fn ihfftn<R: Runtime>(signal: &RudaTensor<R>, axes: &[usize], lengths: &[usize],
    norm: FftNormalization) -> Result<(RudaTensor<R>, RudaTensor<R>), LaunchError> {
    let (real, imag) = rfftn(signal, axes, lengths, reverse_normalization(norm))?;
    Ok((real, crate::spectral_layout::scale(&imag, -1.0, None)?))
}

pub fn hfft2<R: Runtime>(real: &RudaTensor<R>, imag: &RudaTensor<R>, axes: [usize;2],
    lengths: [usize;2], norm: FftNormalization) -> Result<RudaTensor<R>,LaunchError> {
    hfftn(real,imag,&axes,&lengths,norm)
}
pub fn ihfft2<R: Runtime>(signal: &RudaTensor<R>, axes: [usize;2], lengths: [usize;2],
    norm: FftNormalization) -> Result<(RudaTensor<R>,RudaTensor<R>),LaunchError> {
    ihfftn(signal,&axes,&lengths,norm)
}

pub fn hfft<R: Runtime>(real: &RudaTensor<R>, imag: &RudaTensor<R>, axis: usize,
    length: usize, norm: FftNormalization) -> Result<RudaTensor<R>,LaunchError> {
    hfftn(real,imag,&[axis],&[length],norm)
}
pub fn ihfft<R: Runtime>(signal: &RudaTensor<R>, axis: usize, length: usize,
    norm: FftNormalization) -> Result<(RudaTensor<R>,RudaTensor<R>),LaunchError> {
    ihfftn(signal,&[axis],&[length],norm)
}

fn copy<F: HostFloat, R: Runtime>(input: &RudaTensor<R>, output: &RudaTensor<R>, scale: f64) {
    let count = output.meta.shape().iter().product::<usize>();
    if count == 0 {
        return;
    }
    let d = RudaDim::new(input.client.properties(), count);
    copy_scaled::launch::<F, R>(
        &input.client,
        calculate_ruda_count_elemwise(&input.client, count, d),
        d,
        input.clone().into_tensor_arg(),
        output.clone().into_array_arg(),
        F::scalar(scale),
    );
}
fn zero<F: Float + RudaElement, R: Runtime>(output: &RudaTensor<R>) {
    let count = output.meta.shape().iter().product::<usize>();
    if count == 0 {
        return;
    }
    let d = RudaDim::new(output.client.properties(), count);
    zero_array::launch::<F, R>(
        &output.client,
        calculate_ruda_count_elemwise(&output.client, count, d),
        d,
        output.clone().into_array_arg(),
    );
}
#[ruda(launch)]
fn copy_scaled<F: Float + RudaElement>(input: &Tensor<F>, output: &mut Array<F>, scale: F) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }
    let mut rem = pos;
    let mut offset = 0usize;
    let mut dim = input.rank();
    while dim > 0 {
        dim -= 1;
        let coord = rem % input.shape(dim);
        rem /= input.shape(dim);
        offset += coord * input.stride(dim);
    }
    output[pos] = input[offset] * scale;
}
#[ruda(launch)]
fn zero_array<F: Float>(out: &mut Array<F>) {
    let pos = ABSOLUTE_POS;
    if pos < out.len() {
        out[pos] = F::new(0.0);
    }
}
#[ruda(launch)]
fn combine(
    xr: &Tensor<f32>,
    xi: &Tensor<f32>,
    yr: &Tensor<f32>,
    yi: &Tensor<f32>,
    real: &mut Tensor<f32>,
    imag: &mut Tensor<f32>,
    #[comptime] axis: usize,
    #[comptime] n: usize,
    #[comptime] inverse: bool,
) {
    let pos = ABSOLUTE_POS;
    if pos >= real.len() {
        terminate!();
    }
    let mut rem = pos;
    let mut offset = 0usize;
    let mut k = 0usize;
    let mut dim = real.rank();
    while dim > 0 {
        dim -= 1;
        let coord = rem % real.shape(dim);
        rem /= real.shape(dim);
        if dim == axis {
            k = coord;
        } else {
            offset += coord * xr.stride(dim);
        }
    }
    if inverse && k != 0 {
        k = n - k;
    }
    let reflected = k > n / 2;
    let bin = if reflected { n - k } else { k };
    offset += bin * xr.stride(axis);
    let mut a = xi[offset];
    let mut b = yi[offset];
    if reflected {
        a = -a;
        b = -b;
    }
    real[pos] = xr[offset] - b;
    imag[pos] = a + yr[offset];
}

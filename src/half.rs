use crate::FftMode;
use ruda_core::tensor::{DType, Metadata, Shape, data::TensorData};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{backtrace::BackTrace, calculate_ruda_count_elemwise, prelude::*};
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_contiguous_dtype};

fn invalid(reason: &str) -> LaunchError {
    LaunchError::Unknown { reason: reason.into(), backtrace: BackTrace::capture() }
}

pub struct HalfFftPlan<R: Runtime> {
    shape: Shape,
    axis: usize,
    length: usize,
    rows: usize,
    mode: FftMode,
    work: [RudaTensor<R>; 2],
    twiddle: RudaTensor<R>,
}

impl<R: Runtime> HalfFftPlan<R> {
    pub fn new(like: &RudaTensor<R>, axis: usize, length: usize, mode: FftMode) -> Result<Self, LaunchError> {
        crate::multidim::validate(like)?;
        if like.dtype != DType::F16 || axis >= like.meta.num_dims() || !length.is_power_of_two() {
            return Err(invalid("half FFT requires F16 storage, a valid axis and a power-of-two length"));
        }
        let rows = like.meta.shape().iter().enumerate().filter(|(i, _)| *i != axis)
            .try_fold(1usize, |n, (_, &s)| n.checked_mul(s)).ok_or_else(|| invalid("half FFT batch overflow"))?;
        rows.checked_mul(length).filter(|&n| n <= u32::MAX as usize)
            .ok_or_else(|| invalid("half FFT workspace exceeds U32 indexing"))?;
        if length > u32::MAX as usize { return Err(invalid("half FFT length exceeds U32 indexing")); }
        let work = std::array::from_fn(|_| empty_device_contiguous_dtype(like.client.clone(), like.device.clone(),
            [rows, length].into(), DType::F16));
        let table_length = length.max(2);
        let mut values = Vec::with_capacity(table_length);
        for k in 0..table_length / 2 {
            let phase = -2.0 * std::f64::consts::PI * k as f64 / length as f64;
            let (imag, real) = phase.sin_cos();
            values.push(real as f32); values.push(if k == 0 {0.0} else {imag as f32});
        }
        let data = TensorData::new(values, [table_length]);
        let twiddle = RudaTensor::new(like.client.clone(), like.client.create_from_slice(&data.bytes),
            Metadata::new([table_length], [1]), like.device.clone(), DType::F32);
        Ok(Self { shape: like.meta.shape().clone(), axis, length, rows, mode, work, twiddle })
    }

    pub fn retained_bytes(&self) -> usize {
        self.rows * self.length * 4 + self.length.max(2) * 4
    }

    pub(crate) fn run(&mut self, real: &RudaTensor<R>, imag: Option<&RudaTensor<R>>,
        out_real: &RudaTensor<R>, out_imag: &RudaTensor<R>, real_transform: bool,
    ) -> Result<(), LaunchError> {
        let inverse = self.mode == FftMode::Inverse;
        for input in std::iter::once(real).chain(imag) {
            crate::multidim::validate(input)?;
            if input.dtype != DType::F16 || input.meta.shape() != &self.shape
                || !input.client.same_execution_queue(&self.work[0].client) {
                return Err(invalid("half FFT input differs from plan dtype, shape or queue"));
            }
        }
        if (!real_transform || inverse) && imag.is_none() { return Err(invalid("half FFT requires imaginary input")); }
        let mut shape = self.shape.clone();
        shape[self.axis] = if real_transform && !inverse {self.length / 2 + 1} else {self.length};
        for output in [out_real, out_imag] {
            crate::multidim::validate(output)?;
            if output.dtype != DType::F16 || output.meta.shape() != &shape
                || !output.client.same_execution_queue(&real.client) {
                return Err(invalid("half FFT output differs from plan dtype, shape or queue"));
            }
        }
        if self.rows == 0 { return Ok(()); }
        let total = self.rows * self.length;
        let dim = RudaDim::new(real.client.properties(), total);
        prepare::launch::<R>(&real.client, calculate_ruda_count_elemwise(&real.client, total, dim), dim,
            real.clone().into_tensor_arg(), imag.unwrap_or(real).clone().into_tensor_arg(),
            self.work[0].clone().into_array_arg(), self.work[1].clone().into_array_arg(), self.length, self.axis,
            self.length.trailing_zeros(), imag.is_some(), real_transform && inverse);
        let mut half = 1usize;
        while half < self.length {
            butterfly::launch::<R>(&real.client, calculate_ruda_count_elemwise(&real.client, total / 2, dim), dim,
                self.work[0].clone().into_array_arg(), self.work[1].clone().into_array_arg(), self.twiddle.clone().into_array_arg(),
                self.length, half, inverse);
            half *= 2;
        }
        let count = out_real.meta.num_elements();
        finish::launch::<R>(&real.client, calculate_ruda_count_elemwise(&real.client, count, dim), dim,
            self.work[0].clone().into_array_arg(), self.work[1].clone().into_array_arg(),
            out_real.clone().into_tensor_arg(), out_imag.clone().into_tensor_arg(), self.length, self.axis, real_transform, inverse);
        Ok(())
    }
}

#[ruda]
fn reverse(mut value: usize, #[comptime] bits: u32) -> usize {
    let mut result = 0usize;
    for _ in 0..bits { result = (result << 1usize) | (value & 1usize); value >>= 1usize; }
    result
}

#[ruda(launch)]
fn prepare(real: &Tensor<f16>, imag: &Tensor<f16>, wr: &mut Array<f16>, wi: &mut Array<f16>,
    #[comptime] n: usize, #[comptime] axis: usize, #[comptime] bits: u32,
    #[comptime] has_imag: bool, #[comptime] hermitian: bool,
) {
    let pos = ABSOLUTE_POS; if pos >= wr.len() { terminate!(); }
    let row = pos / n; let k = pos % n;
    let bin = if hermitian && k > n / 2 {n - k} else {k};
    let mut xr = f16::new(0.0); let mut xi = f16::new(0.0);
    if bin < real.shape(axis) {
        let mut rem = row; let mut ro = bin * real.stride(axis); let mut io = bin * imag.stride(axis);
        let mut d = real.rank();
        while d > 0 {
            d -= 1;
            if d != axis {
                let c = rem % real.shape(d); rem /= real.shape(d);
                ro += c * real.stride(d); io += c * imag.stride(d);
            }
        }
        xr = real[ro];
        if has_imag && (!hermitian || (bin != 0 && !(n % 2 == 0 && bin == n / 2))) {
            xi = imag[io]; if hermitian && k > n / 2 { xi = -xi; }
        }
    }
    let destination = row * n + reverse(k, bits);
    wr[destination] = xr; wi[destination] = xi;
}

#[ruda(launch)]
fn butterfly(real: &mut Array<f16>, imag: &mut Array<f16>, twiddle: &Array<f32>,
    #[comptime] n: usize, #[comptime] half: usize, #[comptime] inverse: bool,
) {
    let pos = ABSOLUTE_POS; if pos >= real.len() / 2 { terminate!(); }
    let row = pos / (n / 2); let pair = pos % (n / 2); let k = pair % half;
    let a = row * n + pair / half * (2 * half) + k; let b = a + half;
    let t = k * (n / (2 * half)) * 2;
    let wr = twiddle[t]; let mut wi = twiddle[t + 1]; if inverse { wi = -wi; }
    let ar = f32::cast_from(real[a]); let ai = f32::cast_from(imag[a]);
    let br = f32::cast_from(real[b]); let bi = f32::cast_from(imag[b]);
    let xr = wr * br - wi * bi; let xi = wr * bi + wi * br;
    real[a] = f16::cast_from(ar + xr); imag[a] = f16::cast_from(ai + xi);
    real[b] = f16::cast_from(ar - xr); imag[b] = f16::cast_from(ai - xi);
}

#[ruda(launch)]
fn finish(real: &Array<f16>, imag: &Array<f16>, out_real: &mut Tensor<f16>, out_imag: &mut Tensor<f16>,
    #[comptime] n: usize, #[comptime] axis: usize, #[comptime] real_transform: bool, #[comptime] inverse: bool,
) {
    let pos = ABSOLUTE_POS; if pos >= out_real.len() { terminate!(); }
    let mut rem = pos; let mut row = 0usize; let mut stride = 1usize; let mut k = 0usize;
    let mut ro = 0usize; let mut io = 0usize; let mut d = out_real.rank();
    while d > 0 {
        d -= 1; let c = rem % out_real.shape(d); rem /= out_real.shape(d);
        ro += c * out_real.stride(d); io += c * out_imag.stride(d);
        if d == axis { k = c; } else { row += c * stride; stride *= out_real.shape(d); }
    }
    out_real[ro] = real[row * n + k];
    let mut i = imag[row * n + k];
    if real_transform && (inverse || k == 0 || (n % 2 == 0 && k == n / 2)) { i = f16::new(0.0); }
    out_imag[io] = i;
}

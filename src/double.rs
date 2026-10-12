use crate::FftMode;
use ruda_core::tensor::{DType, Metadata, Shape, data::TensorData};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::backtrace::BackTrace;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_contiguous_dtype};

fn invalid(reason: &str) -> LaunchError {
    LaunchError::Unknown {
        reason: reason.into(),
        backtrace: BackTrace::capture(),
    }
}

fn table<R: Runtime>(like: &RudaTensor<R>, values: Vec<f64>) -> RudaTensor<R> {
    let len = values.len();
    let data = TensorData::new(values, [len]);
    RudaTensor::new(
        like.client.clone(),
        like.client.create_from_slice(&data.bytes),
        Metadata::new(Shape::from([len]), [1]),
        like.device.clone(),
        DType::F64,
    )
}

fn coefficient_fft(real: &mut [f64], imag: &mut [f64]) {
    let n = real.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = if bits == 0 {
            0
        } else {
            i.reverse_bits() >> (usize::BITS - bits)
        };
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }
    let mut width = 2;
    while width <= n {
        let half = width / 2;
        for base in (0..n).step_by(width) {
            for k in 0..half {
                let phase = -2.0 * std::f64::consts::PI * k as f64 / width as f64;
                let (wi, wr) = phase.sin_cos();
                let a = base + k;
                let b = a + half;
                let tr = wr * real[b] - wi * imag[b];
                let ti = wr * imag[b] + wi * real[b];
                let ar = real[a];
                let ai = imag[a];
                real[a] = ar + tr;
                imag[a] = ai + ti;
                real[b] = ar - tr;
                imag[b] = ai - ti;
            }
        }
        if width == n {
            break;
        }
        width *= 2;
    }
}

pub struct DoubleFftPlan<R: Runtime> {
    shape: Shape,
    axis: usize,
    length: usize,
    work_length: usize,
    rows: usize,
    mode: FftMode,
    work: [RudaTensor<R>; 4],
    twiddle: [RudaTensor<R>; 2],
    chirp: [RudaTensor<R>; 2],
    spectrum: [RudaTensor<R>; 2],
}

impl<R: Runtime> DoubleFftPlan<R> {
    pub fn new(
        like: &RudaTensor<R>,
        axis: usize,
        length: usize,
        mode: FftMode,
    ) -> Result<Self, LaunchError> {
        if like.dtype != DType::F64
            || like.qparams.is_some()
            || axis >= like.meta.num_dims()
            || length == 0
        {
            return Err(invalid(
                "double FFT requires unquantized F64, a valid axis and positive length",
            ));
        }
        let rows = like
            .meta
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .try_fold(1usize, |size, (_, &extent)| size.checked_mul(extent))
            .ok_or_else(|| invalid("FFT batch size overflow"))?;
        let work_length = if length.is_power_of_two() {
            length
        } else {
            length
                .checked_mul(2)
                .and_then(|n| n.checked_sub(1))
                .and_then(usize::checked_next_power_of_two)
                .ok_or_else(|| invalid("Bluestein workspace overflow"))?
        };
        rows.checked_mul(work_length)
            .filter(|&size| size <= u32::MAX as usize)
            .ok_or_else(|| invalid("FFT workspace exceeds device indexing"))?;
        if work_length > u32::MAX as usize {
            return Err(invalid("FFT coefficient table exceeds device indexing"));
        }
        let mut tw_re = Vec::with_capacity((work_length / 2).max(1));
        let mut tw_im = Vec::with_capacity((work_length / 2).max(1));
        for k in 0..(work_length / 2).max(1) {
            let phase = -2.0 * std::f64::consts::PI * k as f64 / work_length as f64;
            let (im, re) = phase.sin_cos();
            tw_re.push(re);
            tw_im.push(im);
        }
        let mut cr = vec![1.0; length];
        let mut ci = vec![0.0; length];
        let mut br = vec![0.0; work_length];
        let mut bi = vec![0.0; work_length];
        if !length.is_power_of_two() {
            let sign = mode.sign() as f64;
            for k in 0..length {
                let square = (k as u64 * k as u64) % (2 * length as u64);
                let phase = sign * std::f64::consts::PI * square as f64 / length as f64;
                let (im, re) = phase.sin_cos();
                cr[k] = re;
                ci[k] = im;
                br[k] = re;
                bi[k] = -im;
                if k != 0 {
                    br[work_length - k] = re;
                    bi[work_length - k] = -im;
                }
            }
            coefficient_fft(&mut br, &mut bi);
        }
        let work = std::array::from_fn(|_| {
            empty_device_contiguous_dtype(
                like.client.clone(),
                like.device.clone(),
                [rows, work_length].into(),
                DType::F64,
            )
        });
        Ok(Self {
            shape: like.meta.shape().clone(),
            axis,
            length,
            work_length,
            rows,
            mode,
            work,
            twiddle: [table(like, tw_re), table(like, tw_im)],
            chirp: [table(like, cr), table(like, ci)],
            spectrum: [table(like, br), table(like, bi)],
        })
    }

    pub fn execute(
        &mut self,
        real: &RudaTensor<R>,
        imag: &RudaTensor<R>,
        normalize: bool,
    ) -> Result<(RudaTensor<R>, RudaTensor<R>), LaunchError> {
        self.run(real, Some(imag), false, self.length, normalize)
    }

    pub(crate) fn run(
        &mut self,
        real: &RudaTensor<R>,
        imag: Option<&RudaTensor<R>>,
        hermitian: bool,
        output_width: usize,
        normalize: bool,
    ) -> Result<(RudaTensor<R>, RudaTensor<R>), LaunchError> {
        for input in std::iter::once(real).chain(imag) {
            if input.dtype != DType::F64
                || input.qparams.is_some()
                || input.meta.shape() != &self.shape
                || !input.client.same_execution_queue(&self.work[0].client)
            {
                return Err(invalid(
                    "FFT input differs from the plan shape, dtype or execution queue",
                ));
            }
        }
        let mut shape = self.shape.clone();
        shape[self.axis] = output_width;
        let out_re = empty_device_contiguous_dtype(
            real.client.clone(),
            real.device.clone(),
            shape.clone(),
            DType::F64,
        );
        let out_im = empty_device_contiguous_dtype(
            real.client.clone(),
            real.device.clone(),
            shape,
            DType::F64,
        );
        if self.rows == 0 {
            return Ok((out_re, out_im));
        }
        let total = self.rows * self.work_length;
        let dim = RudaDim::new(real.client.properties(), total);
        prepare::launch::<R>(
            &real.client,
            calculate_ruda_count_elemwise(&real.client, total, dim),
            dim,
            real.clone().into_tensor_arg(),
            imag.unwrap_or(real).clone().into_tensor_arg(),
            self.chirp[0].clone().into_array_arg(),
            self.chirp[1].clone().into_array_arg(),
            self.work[0].clone().into_array_arg(),
            self.work[1].clone().into_array_arg(),
            self.length,
            self.work_length,
            self.axis,
            self.work_length.trailing_zeros(),
            imag.is_some(),
            hermitian,
            !self.length.is_power_of_two(),
        );
        let mut pair = self.radix(
            0,
            if self.length.is_power_of_two() {
                self.mode
            } else {
                FftMode::Forward
            },
        );
        if !self.length.is_power_of_two() {
            let other = 2 - pair;
            product_reverse::launch::<R>(
                &real.client,
                calculate_ruda_count_elemwise(&real.client, total, dim),
                dim,
                self.work[pair].clone().into_array_arg(),
                self.work[pair + 1].clone().into_array_arg(),
                self.spectrum[0].clone().into_array_arg(),
                self.spectrum[1].clone().into_array_arg(),
                self.work[other].clone().into_array_arg(),
                self.work[other + 1].clone().into_array_arg(),
                self.work_length,
                self.work_length.trailing_zeros(),
            );
            pair = self.radix(other, FftMode::Inverse);
        }
        let scale = (if self.length.is_power_of_two() {
            1.0
        } else {
            1.0 / self.work_length as f64
        }) * if normalize {
            1.0 / self.length as f64
        } else {
            1.0
        };
        let count = self.rows * output_width;
        finish::launch::<R>(
            &real.client,
            calculate_ruda_count_elemwise(&real.client, count, dim),
            dim,
            self.work[pair].clone().into_array_arg(),
            self.work[pair + 1].clone().into_array_arg(),
            self.chirp[0].clone().into_array_arg(),
            self.chirp[1].clone().into_array_arg(),
            out_re.clone().into_tensor_arg(),
            out_im.clone().into_tensor_arg(),
            self.work_length,
            self.axis,
            scale,
            !self.length.is_power_of_two(),
        );
        Ok((out_re, out_im))
    }

    fn radix(&self, mut pair: usize, mode: FftMode) -> usize {
        let count = self.rows * self.work_length / 2;
        let dim = RudaDim::new(self.work[0].client.properties(), count.max(1));
        let grid = calculate_ruda_count_elemwise(&self.work[0].client, count, dim);
        let mut half = 1;
        while half < self.work_length {
            let other = 2 - pair;
            butterfly::launch::<R>(
                &self.work[0].client,
                grid.clone(),
                dim,
                self.work[pair].clone().into_array_arg(),
                self.work[pair + 1].clone().into_array_arg(),
                self.twiddle[0].clone().into_array_arg(),
                self.twiddle[1].clone().into_array_arg(),
                self.work[other].clone().into_array_arg(),
                self.work[other + 1].clone().into_array_arg(),
                self.work_length,
                half,
                mode == FftMode::Inverse,
            );
            pair = other;
            half *= 2;
        }
        pair
    }

    pub fn retained_bytes(&self) -> usize {
        self.work.iter().chain(self.twiddle.iter()).chain(self.chirp.iter())
            .chain(self.spectrum.iter()).map(|t| t.meta.shape().iter().product::<usize>() * 8).sum()
    }
}

pub fn fft_f64<R: Runtime>(
    real: RudaTensor<R>,
    imag: RudaTensor<R>,
    axis: usize,
    n: Option<usize>,
    mode: FftMode,
    normalize: bool,
) -> Result<(RudaTensor<R>, RudaTensor<R>), LaunchError> {
    let length = n
        .or_else(|| real.meta.shape().get(axis).copied())
        .ok_or_else(|| invalid("FFT axis out of range"))?;
    DoubleFftPlan::new(&real, axis, length, mode)?.execute(&real, &imag, normalize)
}

pub fn rfft_f64<R: Runtime>(
    signal: RudaTensor<R>,
    axis: usize,
    n: Option<usize>,
) -> Result<(RudaTensor<R>, RudaTensor<R>), LaunchError> {
    let length = n
        .or_else(|| signal.meta.shape().get(axis).copied())
        .ok_or_else(|| invalid("FFT axis out of range"))?;
    DoubleFftPlan::new(&signal, axis, length, FftMode::Forward)?.run(
        &signal,
        None,
        false,
        length / 2 + 1,
        false,
    )
}

pub fn irfft_f64<R: Runtime>(
    real: RudaTensor<R>,
    imag: RudaTensor<R>,
    axis: usize,
    n: usize,
) -> Result<RudaTensor<R>, LaunchError> {
    Ok(DoubleFftPlan::new(&real, axis, n, FftMode::Inverse)?
        .run(&real, Some(&imag), true, n, true)?
        .0)
}

#[ruda]
fn reverse(mut value: usize, #[comptime] bits: u32) -> usize {
    let mut out = 0usize;
    for _ in 0..bits {
        out = (out << 1usize) | (value & 1usize);
        value >>= 1usize;
    }
    out
}

#[ruda(launch)]
fn prepare(
    real: &Tensor<f64>,
    imag: &Tensor<f64>,
    cr: &Array<f64>,
    ci: &Array<f64>,
    wr: &mut Array<f64>,
    wi: &mut Array<f64>,
    #[comptime] n: usize,
    #[comptime] m: usize,
    #[comptime] axis: usize,
    #[comptime] bits: u32,
    #[comptime] has_imag: bool,
    #[comptime] hermitian: bool,
    #[comptime] chirped: bool,
) {
    let pos = ABSOLUTE_POS;
    if pos >= wr.len() {
        terminate!();
    }
    let row = pos / m;
    let k = pos % m;
    let mut xr = 0.0f64;
    let mut xi = 0.0f64;
    let bin = if hermitian && k < n && k > n / 2 {
        n - k
    } else {
        k
    };
    if k < n && bin < real.shape(axis) {
        let mut rem = row;
        let mut ro = bin * real.stride(axis);
        let mut io = bin * imag.stride(axis);
        let mut i = real.rank();
        while i > 0 {
            i -= 1;
            if i != axis {
                let coordinate = rem % real.shape(i);
                rem /= real.shape(i);
                ro += coordinate * real.stride(i);
                io += coordinate * imag.stride(i);
            }
        }
        xr = real[ro];
        if has_imag && (!hermitian || (bin != 0 && !(n % 2 == 0 && bin == n / 2))) {
            xi = imag[io];
            if hermitian && k > n / 2 {
                xi = -xi;
            }
        }
        if chirped {
            let old = xr;
            xr = old * cr[k] - xi * ci[k];
            xi = old * ci[k] + xi * cr[k];
        }
    }
    let destination = row * m + reverse(k, bits);
    wr[destination] = xr;
    wi[destination] = xi;
}

#[ruda(launch)]
fn butterfly(
    ar: &Array<f64>,
    ai: &Array<f64>,
    tr: &Array<f64>,
    ti: &Array<f64>,
    br: &mut Array<f64>,
    bi: &mut Array<f64>,
    #[comptime] n: usize,
    #[comptime] half: usize,
    #[comptime] inverse: bool,
) {
    let pos = ABSOLUTE_POS;
    if pos >= ar.len() / 2 {
        terminate!();
    }
    let row = pos / (n / 2);
    let butterfly = pos % (n / 2);
    let k = butterfly % half;
    let a = row * n + (butterfly / half) * (2 * half) + k;
    let b = a + half;
    let twiddle = k * (n / (2 * half));
    let wr = tr[twiddle];
    let mut wi = ti[twiddle];
    if inverse {
        wi = -wi;
    }
    let xr = wr * ar[b] - wi * ai[b];
    let xi = wr * ai[b] + wi * ar[b];
    br[a] = ar[a] + xr;
    bi[a] = ai[a] + xi;
    br[b] = ar[a] - xr;
    bi[b] = ai[a] - xi;
}

#[ruda(launch)]
fn product_reverse(
    ar: &Array<f64>,
    ai: &Array<f64>,
    sr: &Array<f64>,
    si: &Array<f64>,
    br: &mut Array<f64>,
    bi: &mut Array<f64>,
    #[comptime] n: usize,
    #[comptime] bits: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos >= ar.len() {
        terminate!();
    }
    let k = pos % n;
    let destination = (pos / n) * n + reverse(k, bits);
    br[destination] = ar[pos] * sr[k] - ai[pos] * si[k];
    bi[destination] = ar[pos] * si[k] + ai[pos] * sr[k];
}

#[ruda(launch)]
fn finish(
    ar: &Array<f64>,
    ai: &Array<f64>,
    cr: &Array<f64>,
    ci: &Array<f64>,
    out_re: &mut Tensor<f64>,
    out_im: &mut Tensor<f64>,
    #[comptime] m: usize,
    #[comptime] axis: usize,
    scale: f64,
    #[comptime] chirped: bool,
) {
    let pos = ABSOLUTE_POS;
    if pos >= out_re.len() {
        terminate!();
    }
    let mut rem = pos;
    let mut row = 0usize;
    let mut row_stride = 1usize;
    let mut k = 0usize;
    let mut i = out_re.rank();
    while i > 0 {
        i -= 1;
        let coordinate = rem % out_re.shape(i);
        rem /= out_re.shape(i);
        if i == axis {
            k = coordinate;
        } else {
            row += coordinate * row_stride;
            row_stride *= out_re.shape(i);
        }
    }
    let xr = ar[row * m + k];
    let xi = ai[row * m + k];
    if chirped {
        out_re[pos] = (xr * cr[k] - xi * ci[k]) * scale;
        out_im[pos] = (xr * ci[k] + xi * cr[k]) * scale;
    } else {
        out_re[pos] = xr * scale;
        out_im[pos] = xi * scale;
    }
}

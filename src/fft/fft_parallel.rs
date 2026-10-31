//! Intra-ruda-parallel radix-2 FFT primitives used by the RFFT kernels.
//!
//! Each ruda processes one FFT window with `RUDA_DIM` units. Inputs and
//! outputs live in per-ruda `SharedMemory<F>`; callers load a bit-reversed
//! window before calling [`fft_butterfly_parallel`].

use ruda_kernel::dsl as kernel_dsl;
use core::f32::consts::PI;

use ruda_kernel::dsl::prelude::*;

use crate::fft::FftMode;

pub(crate) fn warp_stages<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let hardware = &client.properties().hardware;
    hardware.plane_size_min == 32
        && hardware.plane_size_max == 32
        && client.features().plane.contains(kernel_dsl::ir::features::Plane::Ops)
}

/// Reverse the lowest `log2_n` bits of `i`.
#[ruda]
pub(crate) fn bit_reverse(i: usize, #[comptime] log2_n: usize) -> usize {
    let mut j = 0usize;
    let mut x = i;
    #[unroll]
    for _ in 0..log2_n {
        j = (j << 1usize) | (x & 1usize);
        x >>= 1usize;
    }
    j
}

/// Parallel radix-2 butterfly stages on an already-bit-reversed window.
#[ruda]
pub(crate) fn fft_butterfly_parallel<F: Float>(
    shared_re: &mut SharedMemory<F>,
    shared_im: &mut SharedMemory<F>,
    #[comptime] n_fft: usize,
    #[comptime] log2_n: usize,
    #[comptime] threads_per_ruda: usize,
    #[comptime] fft_mode: FftMode,
    #[comptime] warp_stages: bool,
) {
    let num_butterflies = comptime![n_fft / 2];
    let sign = F::new(fft_mode.sign());
    let two_pi = F::new(2.0 * PI);

    let mut s = 0usize;
    if comptime!(warp_stages && n_fft >= 32 && threads_per_ruda >= 32
        && threads_per_ruda % 32 == 0 && n_fft % threads_per_ruda == 0) {
        let lane = UNIT_POS as usize % 32;
        let mut position = UNIT_POS as usize;
        while position < n_fft {
            let mut re = shared_re[position];
            let mut im = shared_im[position];
            #[unroll]
            for stage in 0usize..5 {
                let half_m = comptime!(1usize << stage);
                let m = comptime!(half_m * 2);
                let j = lane % half_m;
                let low = (lane & half_m) == 0;
                let partner_re = plane_shuffle_xor(re, half_m as u32);
                let partner_im = plane_shuffle_xor(im, half_m as u32);
                let ar = select(low, re, partner_re);
                let ai = select(low, im, partner_im);
                let br = select(low, partner_re, re);
                let bi = select(low, partner_im, im);
                let theta: F = sign * two_pi * F::cast_from(j) / F::cast_from(m);
                let w_re = theta.cos();
                let w_im = theta.sin();
                let tr = w_re * br - w_im * bi;
                let ti = w_re * bi + w_im * br;
                re = select(low, ar + tr, ar - tr);
                im = select(low, ai + ti, ai - ti);
            }
            shared_re[position] = re;
            shared_im[position] = im;
            position += threads_per_ruda;
        }
        sync_ruda();
        s = 5usize;
    }
    while s < log2_n {
        let half_m = 1usize << s;
        let m = half_m << 1usize;

        let mut b = UNIT_POS as usize;
        while b < num_butterflies {
            let group = b / half_m;
            let j = b - group * half_m;
            let i0 = group * m + j;
            let i1 = i0 + half_m;

            let theta: F = sign * two_pi * F::cast_from(j) / F::cast_from(m);
            let w_re = theta.cos();
            let w_im = theta.sin();

            let ar = shared_re[i0];
            let ai = shared_im[i0];
            let br = shared_re[i1];
            let bi = shared_im[i1];

            let tr = w_re * br - w_im * bi;
            let ti = w_re * bi + w_im * br;

            shared_re[i0] = ar + tr;
            shared_im[i0] = ai + ti;
            shared_re[i1] = ar - tr;
            shared_im[i1] = ai - ti;

            b += threads_per_ruda;
        }
        sync_ruda();
        s += 1usize;
    }
}

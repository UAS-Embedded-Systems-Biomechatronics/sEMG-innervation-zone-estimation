/* 
* Copyright 2023 Malte Mechtenberg
* 
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

//! # Hermite Hermite-Rodriguez
//! A Collection fo functions to calculate the wavelt tracking for EMG data.
//! Implementing the hermite rotriguez function and a helper function to 
//! calculate the convolution.
//!
//!
use std::f64::consts::PI;
use factorial::Factorial;

pub fn hermite(t : f64, n : usize) -> f64 {
    if n == 0{
        return 1_f64;
    } else if n == 1 {
        return 2_f64 * t;
    } else {
        return 2_f64 * t * hermite(t, n - 1 ) 
            - 2_f64 * ((n - 1) as f64) * hermite(t, n-2);
    }
}

pub fn omega(t : f64, lam : f64, n : usize) ->  f64
{
    /*
    Hermite-Rodriguez Series Expansion Loredana 1994
        url: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=335863
    */
    let a : f64 = 2.0_f64.powi(n as i32) * n.factorial() as f64;
    let b : f64 = 1.0 / ((PI).sqrt() * lam); // negative lam here is the probelm

    return (1.0 / a.sqrt()) 
        * hermite(t / lam, n) 
        * b
        * (- t.powi(2) / (lam.powi(2)) ).exp()
}

pub fn correlate_with_wavlet(
    w : Box<dyn Fn(f64) -> f64>,
    a : f64,
    tau: f64,
    time_disc : &[f64],
    data_disc : &[f64]) -> Vec<f64> {
    let a = a.sqrt();
    data_disc.iter().zip(time_disc).map(|(d, t)|  {
            return *d * w( (*t - tau)/a ) / a
        }).collect()
}

/// Convolute the discrete vector `data_disc` with a wavelet function `w`.
///
/// # Arguments
///
/// * `w` a closure or function that takes one argument `fn(t : f64)`. The Argument `t`
///       is the time
/// * `a` is a scaling factor for the wavelet in the time domain and for the amplitude
///          `w( (t - tau)/a ) / sqrt(a)`
/// * `tau` defines the amount with wich the wavelet is shifted in time.
/// * `time_disc` is the discrete time vector corresponding to `data_disc`
/// * `data_disc` is the vector of data which is convolted with the wavelet `w`
pub fn convolute_with_wavelet(
    w : Box<dyn Fn(f64) -> f64>,
    a : f64,
    tau: f64,
    time_disc : &[f64],
    data_disc : &[f64]) -> f64 
{
    let a = a.sqrt();
    data_disc.iter().zip(time_disc).map(|(d, t)|  {
            return *d * w( (*t - tau)/a ) / a
        }).sum()
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::fs;
    use float_cmp::{approx_eq};

    #[test]
    fn basic_covolution_test() {
        let time : Vec<f64> = (0..1000).map(|x| x as f64 * 0.001).collect();
        let data : Vec<f64> = time.iter().map(|t| omega(*t, 0.001, 2)).collect();

        let res = convolute_with_wavelet(Box::new(|_| 1.0), 1_f64, 0.5, &time, &data);

        assert_eq!(res, data.iter().sum())
    }

    #[derive(serde_derive::Deserialize)]
    struct TestData {
        second_order : OmegaTestParmas,
        corellate_with_wavelet : CorellateWithWavelet,
    }

    #[derive(serde_derive::Deserialize)]
    struct CorellateWithWavelet {
        time   : Vec<f64>,
        input  : Vec<f64>,
        output : Vec<f64>,
        lambda : f64,
        n      : usize,
        a      : f64,
    }

    #[derive(serde_derive::Deserialize)]
    struct OmegaTestParmas {
        time_s : Vec<f64>,
        data : Vec<f64>,
        lambda : f64,
        n      : usize 
    }

    
    fn load_test_data() -> TestData {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("test_data");
        p.push("hermite_rodriguez.toml");

        let test_config = fs::read_to_string(p).unwrap();
        let data : TestData = toml::from_str(&test_config).unwrap();

        return data;
    }

    #[test]
    fn corellate() {
        let test_data = load_test_data();
        let test_data = test_data.corellate_with_wavelet;

        let w = Box::new(move |t| omega(t as f64, test_data.lambda, test_data.n));

        let conv_res :Vec<f64> = test_data.time
            .iter()
            .map(
                | tau | convolute_with_wavelet(
                    w.clone(),
                    test_data.a,
                    *tau,
                    &test_data.time,
                    &test_data.input)
            ).collect();

        let mae = conv_res
            .iter()
            .zip(test_data.output.clone())
            .map(|(c,o)| (c - o).abs())
            .sum::<f64>()  /  conv_res.len() as f64;

        let mut c_max = 0.0;
        for c in conv_res {
            if c > c_max {
                c_max = c;
            }

        }

        let mut c_max_out = 0.0;
        for c in test_data.output {
            if c > c_max_out {
                c_max_out = c;
            }
        }

        assert_eq!(c_max, c_max_out);

        println!("c_max {c_max}; c_max_out {c_max_out}");
        println!("mae {mae}");
        assert!(approx_eq!(f64, mae, 0.0))
    }

    #[test]
    fn compare_with_data() {
        let test_data = load_test_data();


        let w_res : Vec<f64> = test_data.second_order.time_s
            .iter()
            .map(|t| omega(*t, test_data.second_order.lambda, test_data.second_order.n))
            .collect();

        let error = w_res
            .iter()
            .zip(test_data.second_order.data)
            .map(|(&r, p)| (r-p).abs())
            .sum::<f64>() / w_res.len() as f64;

        assert!(float_cmp::approx_eq!(f64,error, 0.0_f64));
    }
}

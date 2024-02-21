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

use std::f64::consts::PI;


pub enum WindowKind {
    Symetric,
    Periodic
}

/// TODO actually use the hann window
fn discrete_hann (n : usize, N : usize)  -> f64 {
    assert!(n<N);

    // Wikipedia implementation of the hann window that
    // is scaled to amplitute 1
    //let a = (PI * (n as f64) ) / (N as f64);
    //a.sin().powi(2)
    //
    // The following is the scipy.signal.hann implementation
    let a = (2_f64 *PI*n as f64) / ((N - 1) as f64);
    0.5_f64 - 0.5_f64 * a.cos()
}

pub fn discrete_hann_window (N :usize, kind : WindowKind) -> Vec<f64> {
    let n_passdown;
    let mut w : Vec<f64> = Vec::new();
    w.reserve(N);

    match kind {
        WindowKind::Symetric => n_passdown = N,
        WindowKind::Periodic  => n_passdown = N+1
    }

    for idx in 0..N {
        w.push(discrete_hann(idx, n_passdown));
    }

    return w;
}


#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;
    use std::path::PathBuf;
    use std::fs;

    #[derive(serde_derive::Deserialize)]
    struct TestData {
        hann_window_periodic :  HannWindow,
        hann_window_symetric :  HannWindow,
    }

    #[derive(serde_derive::Deserialize)]
    struct HannWindow {
        disc_window : Vec<f64>,
        n           : usize
    }

    fn load_test_data() -> TestData {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("test_data");
        p.push("window_functions.toml");

        let test_config = fs::read_to_string(p).unwrap();
        let data : TestData = toml::from_str(&test_config).unwrap();

        return data;
    }

    fn do_window_test(test_window : HannWindow, w: &Vec<f64>, plot_name : &'static str) {

        assert_eq!(w.len(), test_window.disc_window.len());

        let mae : f64 = w.iter()
            .zip(test_window.disc_window.clone())
            .map(|(w, w_test)| (w-w_test).abs())
            .sum::<f64>()  / w.len() as f64;

        use plotters::prelude::*;
        {
            let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            p.push("test_plots");
            p.push(plot_name);
            let p = p;
            let figure = plotters_svg::SVGBackend::new(&p, (1024, 720)).into_drawing_area();

            let x_lim  = (
                 0.0,
                test_window.disc_window.len() as f64 - 1.0);
            let y_lim  : (f64, f64) = (
                *test_window.disc_window.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).unwrap()
                , *test_window.disc_window.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap());

            figure.fill(&WHITE).unwrap();
            let mut ctx = ChartBuilder::on(&figure)
                .set_label_area_size(LabelAreaPosition::Left, 40)
                .set_label_area_size(LabelAreaPosition::Bottom, 40)
                .build_cartesian_2d(x_lim.0..x_lim.1, y_lim.0..y_lim.1)
                .unwrap();

            ctx.configure_mesh().draw().unwrap();
            ctx.draw_series(
                LineSeries::new(
                    test_window.disc_window.clone()
                        .iter()
                        .enumerate()
                        .map(|(t,y)| (t as f64, *y))
                    , &BLACK
                )
            ).unwrap();
            ctx.draw_series(
                LineSeries::new(
                    w.iter().enumerate()
                    .map(|(t, y)| (t as f64, *y as f64))
                    , &BLUE
                )
            ).unwrap();
        }
        //println!("{:#?}", w);
        let fract_of_eps : usize = ((mae - 0_f64) / f64::EPSILON).ceil() as usize;

        assert!(approx_eq!(f64, mae, 0.0), 
            "error is {mae} that is fraction of eps {fract_of_eps} eps is {}", f64::EPSILON);
    }

    #[test]
    fn comparison_with_scipy() {
        let test_data = load_test_data();

        let w = discrete_hann_window(test_data.hann_window_periodic.n, WindowKind::Periodic);
        do_window_test( test_data.hann_window_periodic, &w, "cmp_window_scipy_periodic.svg");

        let w = discrete_hann_window(test_data.hann_window_symetric.n, WindowKind::Symetric);
        do_window_test( test_data.hann_window_symetric, &w, "cmp_window_scipy_symetric.svg");
    }
}

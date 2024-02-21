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

use itertools::Itertools;
use num_traits::{FromPrimitive, ToPrimitive};
use threadpool::ThreadPool;
use std::sync::{
    Arc,
    Mutex
};

use crate::{
    util::{LineIntersection, LineX},
    data::{EmgArray, SliceEmgArray},
    window_functions::WindowKind,
};

pub mod util;
pub mod hermite_rodriguez;
pub mod data;
pub mod window_functions;

#[derive(Debug, Clone, Copy)]
pub struct IzEstimate {
    pub time :  f64,
    pub pos_in_array : f64
}

impl IzEstimate {
    fn from_line_intersection(li : LineIntersection<f64> ) -> IzEstimate {
        IzEstimate {
            time : li.x,
            pos_in_array : li.y
        }
    }
}


struct WindowParameters {
    window_width : usize,
    window_step  : usize
}

impl WindowParameters {
    fn from_times(
        t0 : f64, t1: f64,
        window_width_in_time : f64,
        window_step_in_time : f64) -> Result<WindowParameters, &'static str> {

        let sampling_period    = t1 - t0;

        let window_width : Option<usize> = (window_width_in_time / sampling_period)
            .to_usize();
        let window_width = match window_width {
            Some(s) => s,
            None    => return Err("Could not claculate the window size.")
        };

        let window_step : Option<usize> = (window_step_in_time / sampling_period)
            .to_usize();
        let window_step = match window_step {
            Some(s) => s,
            None    => return Err("Could not calculate the step width of the sliding window"),
        };

        let window_parameters = WindowParameters {
            window_width,
            window_step
        };

        Ok(window_parameters)
    }

    ///
    ///# Window layout:
    /// `          0                        (n_data - 1)`
    /// `data:     |..............................|`
    ///
    /// `.         |<.....w.....>|                 `
    /// `window 0: |<.wst.>x.....|                 `
    /// `window 1:         |<.wst.>x.....|         `
    /// `widnow n:                  |<.wst.>x.....|`
    ///
    /// where `w` is the `window_width and  `wst` is the `window_step`
    pub fn get_n_windows(&self, n_data : usize) -> usize {
        assert!(self.window_width <= n_data);

        let d         = n_data - self.window_width;
        let n_windows = d / self.window_step + 1; // sollte passen wegen der interger devision

        return n_windows
    }
}

#[derive(Clone, Copy)]
pub struct EstimateForLinearArray {
    lambda: f64,
    epsilon: f64,
    electrode_distance: f64,
    expected_v_conduction: f64
}

impl EstimateForLinearArray {
    pub fn new( 
        lambda: f64,
        epsilon: f64,
        electrode_distance: f64,
        expected_v_conduction: f64) -> EstimateForLinearArray {
        EstimateForLinearArray {
            lambda,
            epsilon,
            electrode_distance,
            expected_v_conduction 
        }
    }

    pub fn p_with_sliding_window(
        &self,
        emg_array : & EmgArray,
        window_width_in_time : f64,
        window_step_in_time : f64,
        n_worker : usize,
    ) -> Result<Vec<Option<IzEstimate>>, &'static str> {
        if window_step_in_time <= 0.0 {
            return Err(
                "parameter window_step_in_time has to be a positive \
                 float64: it was {window_step_in_time}"
            )
        }

        let window_parameters = Arc::new(WindowParameters::from_times(
            emg_array.time[0],
            emg_array.time[1],
            window_width_in_time,
            window_step_in_time)?);
        let n_windows = window_parameters.get_n_windows( emg_array.time.len());


        let mut iz_estimates : Vec<Option<IzEstimate>> = Vec::new();
        iz_estimates.reserve(n_windows);
        for _ in 0..n_windows {
            iz_estimates.push(None);
        }
        let iz_estimates = Arc::new(Mutex::new(iz_estimates));

        let d_hann_window = Arc::new(window_functions::discrete_hann_window(
            window_parameters.window_width,
            WindowKind::Periodic
        ));

        let worker_pool = ThreadPool::new(n_worker);
        let emg_array = Arc::new(emg_array.clone());

        for idx_window in 0..n_windows {
            let d_hann_window = d_hann_window.clone();
            let window_parameters = window_parameters.clone();
            let efla = self.clone();
            let emg_array = emg_array.clone();

            let iz_estimates = iz_estimates.clone();

            worker_pool.execute( move || {
                let li : usize = idx_window * window_parameters.window_step;
                let lu = li + window_parameters.window_width;

                let emg_array_slice = emg_array.get_slice_of_time_in_column(li, lu).unwrap();

                let emg_array_local = emg_array_slice.apply_window(d_hann_window.as_ref());

                let estimate : Option<IzEstimate> = efla.single_array(
                     &emg_array_local,
                 );

                let mut iz_estimates = iz_estimates.lock().unwrap();
                iz_estimates[idx_window] = estimate;
            })
        }

        worker_pool.join();

        let iz_estimates = Arc::try_unwrap(iz_estimates).unwrap()
                                .into_inner().unwrap();

        return Ok(iz_estimates);
    }

    pub fn with_sliding_window(
        &self,
        emg_array : & EmgArray,
        window_width_in_time : f64,
        window_step_in_time : f64,
    ) -> Result<Vec<Option<IzEstimate>>, &'static str> {
        if window_step_in_time <= 0.0 {
            return Err(
                "parameter window_step_in_time has to be a positive \
                 float64: it was {window_step_in_time}"
            )
        }

        let window_parameters = WindowParameters::from_times(
            emg_array.time[0],
            emg_array.time[1],
            window_width_in_time,
            window_step_in_time)?;
        let n_windows = window_parameters.get_n_windows( emg_array.time.len());


        let mut iz_estimates : Vec<Option<IzEstimate>> = Vec::new();
        iz_estimates.reserve(n_windows);

        let d_hann_window = window_functions::discrete_hann_window(
            window_parameters.window_width,
            WindowKind::Periodic
        );

        for idx_window in 0..n_windows {
            let li : usize = idx_window * window_parameters.window_step;
            let lu = li + window_parameters.window_width;

            let emg_array_slice = emg_array.get_slice_of_time_in_column(li, lu)?;
            let emg_array_local = emg_array_slice.apply_window(&d_hann_window);

            let estimate : Option<IzEstimate> = self.single_array(
                 &emg_array_local,
             );

            #[cfg(debug_assertions)]
            if estimate.is_none() {
                println!(" li : {li}, lu : {lu}, \t is None");
            } else {
                println!(" li : {li}, lu : {lu}, \t is {:?}", estimate.as_ref().unwrap());
            }

            iz_estimates.push(estimate);
        }

        return Ok(iz_estimates);
    }

    pub fn single_array(&self, emg_array : &EmgArray) -> Option<IzEstimate> {
        let emg_array = emg_array
            .get_slice_of_time_in_column(0, emg_array.time.len())
            .unwrap();

        return self.single_slice(&emg_array)
    }

    pub fn single_slice (
            &self,
            emg_array: &SliceEmgArray,
        )-> Option<IzEstimate> {
        assert_eq!(emg_array.time.len(), emg_array.data[0].len());

        let cluster_param = DbscanParameters {
            eps : self.epsilon,
            min_points : 3,
        };

        let mut muap_positions : Vec<f64> = Vec::new();
        muap_positions.reserve(emg_array.data.len());

        for electrode_potential in &emg_array.data {
            let mp = muap_position(
                emg_array.time,
                electrode_potential,
                self.lambda);
            muap_positions.push(mp);
        }


        let mut blc = ByLineCluster::new(
            muap_positions,
            emg_array.electrode_x.clone(),
            cluster_param,
            self.electrode_distance,
            self.expected_v_conduction);
        
        match blc.predict_iz() {
            Some(s) => return Some(IzEstimate::from_line_intersection(s)),
            None => return None
        }
    }
}

fn muap_position(time: &[f64], data: &[f64], lambda: f64) -> f64 {
    let w = Box::new(move |t| hermite_rodriguez::omega(t as f64, lambda, 2) );
    let mut c_t_max = 0_f64;
    let mut t_max   = 0_f64;

    for &t in time {
        let c_t = hermite_rodriguez::convolute_with_wavelet(
            w.clone(), 1.0, t, time, data
        );
        if c_t > c_t_max {
            c_t_max = c_t;
            t_max   = t;
        }
    }

    return t_max;
}

#[derive(Debug)]
pub struct DbscanParameters {
    eps: f64,
    min_points: usize,
}

#[derive(Debug)]
pub struct ScaleIntersection {
    x: f64,
    y: f64,
}

struct ByLineCluster<T> {
    time: Vec<T>,
    electrode_x: Vec<T>,

    lines: Vec<LineX<T>>,

    intersections: Vec<LineIntersection<T>>,

    norm: ScaleIntersection,

    cluster_param: DbscanParameters,
    cluster_classes: Option<Vec<dbscan::Classification>>,
    cluster_max_label: Option<i64>,
}

impl<T> ByLineCluster<T>
where
    T: std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::DivAssign
        + std::ops::Add<Output = T>
        + std::ops::AddAssign
        + std::ops::Sub<Output = T>
        + std::cmp::PartialOrd<T>
        + std::fmt::Debug
        + num_traits::Float
        //+ std::cmp::PartialEq
        + num_traits::cast::FromPrimitive
        + Copy,
    f64: From<T>,
{
    // TODO fix this one
    fn default(
        time: Vec<T>,
        electrode_x: Vec<T>,
        cluster_param: DbscanParameters,
    ) -> ByLineCluster<T> {
        return Self::new(time, electrode_x, cluster_param, 1.0, 4.0);
    }

    ///
    ///  TODO pass references to `time` and `electrode_x`
    ///
    ///  `expected_v_conduction` 4 m/s is a good value the value
    ///  is always supplied with the unit m/s
    fn new(
        time: Vec<T>,
        electrode_x: Vec<T>,
        cluster_param: DbscanParameters,
        electrode_distance: f64,
        expected_v_conduction: f64,
    ) -> ByLineCluster<T> {
        let lines = LineX::from_points(&time, &electrode_x);

        let intersections: Vec<LineIntersection<T>> = Vec::new();
        let cluster_classes = None;
        let cluster_max_label = None;

        let _m = 1.0;
        let _s = 1.0;
        let _mps = _m / _s;

        let norm = ScaleIntersection {
            x: electrode_distance / (expected_v_conduction * _mps),
            y: 1.0, //y: f64::from_usize(electrode_x.len()).unwrap(),
        };

        ByLineCluster {
            time,
            electrode_x,
            lines,
            intersections,
            cluster_param,
            cluster_classes,
            cluster_max_label,
            norm,
        }
    }

    fn predict_iz(&mut self) -> Option<LineIntersection<T>> {
        self.calc_line_intersections();
        self.cluster_intersections();
        return self.get_mean_of_biggest_cluster();
    }

    fn reset(&mut self) {
        self.intersections.clear();
        self.cluster_classes = None;
        self.cluster_max_label = None;
    }

    fn calc_line_intersections(&mut self) {
        self.reset();

        let mut idx_pos = Vec::new();
        let mut idx_neg = Vec::new();

        for idx in 0..self.lines.len() {
            if self.lines[idx].is_positive {
                idx_pos.push(idx);
            } else {
                idx_neg.push(idx);
            }
        }

        for n in &idx_neg {
            for p in &idx_pos {
                match self.lines[*p].get_intersection(&self.lines[*n]) {
                    Some(l) => self.intersections.push(l),
                    None => continue,
                };
            }
        }
    }

    fn normalize_intersections(&self) -> Vec<Vec<T>> {
        let mut normailized_intersections: Vec<Vec<T>> = Vec::new();
        let scaling = [1.0 / self.norm.x, 1.0 / self.norm.y];

        for intersect in self.intersections.as_slice() {
            normailized_intersections.push(
                (intersect
                    .as_vec()
                    .iter()
                    .zip(&scaling)
                    .map(|(&i1, &i2)| i1 * FromPrimitive::from_f64(i2).unwrap())
                    .collect::<Vec<T>>())
                .to_vec(),
            );
        }

        return normailized_intersections;
    }

    fn cluster_intersections(&mut self) {
        let input = self.normalize_intersections();

        let dbscan = dbscan::Model::new(self.cluster_param.eps, self.cluster_param.min_points);
        self.cluster_classes = Some(dbscan.run(&input));
    }

    fn search_for_biggest_cluster(&mut self) -> Option<i64> {
        let u_label_id_s: Vec<(usize, i64)> = self
            .cluster_classes
            .as_ref()
            .unwrap()
            .iter()
            .map(|&x| dbscan_class_get_value(x).unwrap_or_else(|| -1))
            .sorted()
            .dedup_with_count()
            .collect();

        let max_label_non_noise: Vec<(usize, i64)> = u_label_id_s
            .iter()
            .filter(|&x| x.1 > -1)
            .map(|x| *x)
            .collect();
        //.max_by(|&l, &r| return l.0.cmp(&r.0))
        //
        //
        //what should happen when there are two classes with the same
        //amount of samples? maybe the cluster with the smallest dtabw?

        let mut max_l: Option<i64> = None;
        let mut n_max_l = 0 as usize;
        for (n, l) in max_label_non_noise {
            if n > n_max_l {
                n_max_l = n;
                max_l = Some(l);
            }
        }
        self.cluster_max_label = max_l;
        max_l
    }

    fn get_mean_of_biggest_cluster(&mut self) -> Option<LineIntersection<T>> {
        let max_label = self.search_for_biggest_cluster();
        if None == max_label {
            return None
        }
        let max_label = max_label.unwrap();
        
        let cluster_classes = self.cluster_classes.as_ref().unwrap();

        let mut mean_intersection: Option<Box<LineIntersection<T>>> = Option::None;
        let mut n = 0 as usize;

        for idx in 0..self.intersections.len() {
            if cluster_classes[idx] == dbscan::Classification::Noise {
                continue;
            }
            if dbscan_class_get_value(cluster_classes[idx]).unwrap() == max_label {
                let li = mean_intersection.get_or_insert(Box::new(LineIntersection::<T> {
                    x: FromPrimitive::from_usize(0).unwrap(),
                    y: FromPrimitive::from_usize(0).unwrap(),
                }));

                **li += self.intersections[idx];
                n += 1;
            }
        }

        let n: T = FromPrimitive::from_usize(n).unwrap();

        let mean_intersection = *mean_intersection.unwrap() / n;

        return Some(mean_intersection);
    }
}

//TODO extend dbscan crate
//impl dbscan::Classification {
fn dbscan_class_get_value(c: dbscan::Classification) -> Option<i64> {
    match c {
        dbscan::Classification::Noise => return Option::None,
        dbscan::Classification::Core(v) | dbscan::Classification::Edge(v) => {
            return Option::Some(v.try_into().unwrap())
        }
    }
}
//}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::path::PathBuf;
    use super::*;
    use float_cmp::{approx_eq, Ulps};
    use std::fs;

    #[test]
    fn get_class_value() {
        let a = dbscan::Classification::Core(1);
        let a = dbscan_class_get_value(a).unwrap();
        assert_eq!(a, 1);

        let b = dbscan::Classification::Edge(1);
        let b = dbscan_class_get_value(b).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn get_biggest_cluster() {
        let time: Vec<f64> = vec![0.0, 1.0];
        let electrode_x: Vec<f64> = vec![-1.0, 1.0];
        let cluster_param = DbscanParameters {
            eps: 1.0,
            min_points: 1,
        };

        let mut ip_by_line = ByLineCluster::default(time, electrode_x, cluster_param);

        let test_case = vec![
            dbscan::Classification::Core(1),
            dbscan::Classification::Edge(1),
            dbscan::Classification::Core(2),
            dbscan::Classification::Noise,
            dbscan::Classification::Noise,
            dbscan::Classification::Noise,
        ];
        ip_by_line.cluster_classes = Some(test_case.clone());

        assert_eq!(ip_by_line.search_for_biggest_cluster().unwrap(), 1);
        assert_ne!(ip_by_line.search_for_biggest_cluster().unwrap(), 2);
    }

    #[test]
    fn normalize_intersections() {
        let time_vec: Vec<f64> = vec![0.0, 1.0];
        let elec_vec: Vec<f64> = vec![-1.0, 1.0];
        let cluster_param = DbscanParameters {
            eps: 1.0,
            min_points: 1,
        };

        let mut ip_lc = ByLineCluster::new(
            time_vec,
            elec_vec,
            DbscanParameters {
                eps: 1.10083333,
                min_points: 3,
            },
            0.005, // 5 mm
            4.0,   // m/s
        );

        assert_eq!(ip_lc.norm.x, 0.00125);
        assert_eq!(ip_lc.norm.y, 1.0);

        ip_lc.intersections = vec![LineIntersection { x: 1.0, y: 1.0 }];

        let nom_inters = ip_lc.normalize_intersections();

        // array([[800.,   1.]])
        assert_eq!(nom_inters[0][0], 800.0);
        assert_eq!(nom_inters[0][1], 1.0);
    }

    #[test]
    fn get_mean_biggest_cluster() {
        let time: Vec<f64> = vec![0.0, 1.0];
        let electrode_x: Vec<f64> = vec![-1.0, 1.0];
        let cluster_param = DbscanParameters {
            eps: 1.0,
            min_points: 1,
        };

        let mut ip_by_line = ByLineCluster::default(time, electrode_x, cluster_param);

        let x_vec_c0_p1 = vec![0.2, 1.0, 0.5556];
        let y_vec_c0_p1 = vec![0.3, 5.0, 0.2];

        let x_vec_c1_p1 = vec![0.9, 5.0, 0.1];
        let y_vec_c1_p1 = vec![0.9, 1.0, 0.2];

        let x_vec_c0_p2 = vec![0.9, 4.0, 0.1];
        let y_vec_c0_p2 = vec![1.9, 1.0, 0.1];

        let mut x_vec_c0: Vec<f64> = Vec::new();
        x_vec_c0.append(&mut x_vec_c0_p1.clone());
        x_vec_c0.append(&mut x_vec_c0_p2.clone());

        let mut y_vec_c0: Vec<f64> = Vec::new();
        y_vec_c0.append(&mut y_vec_c0_p1.clone());
        y_vec_c0.append(&mut y_vec_c0_p2.clone());

        let mut x_vec: Vec<f64> = Vec::new();
        x_vec.append(&mut x_vec_c0_p1.clone());
        x_vec.push(1.0);
        x_vec.append(&mut x_vec_c1_p1.clone());
        x_vec.push(2.0);
        x_vec.append(&mut x_vec_c0_p2.clone());
        x_vec.push(3.0);

        let mut y_vec: Vec<f64> = Vec::new();
        y_vec.append(&mut y_vec_c0_p1.clone());
        y_vec.push(1.0);
        y_vec.append(&mut y_vec_c1_p1.clone());
        y_vec.push(1.0);
        y_vec.append(&mut y_vec_c0_p2.clone());
        y_vec.push(1.0);

        for idx in 0..x_vec.len() {
            ip_by_line.intersections.push(LineIntersection {
                x: x_vec[idx],
                y: y_vec[idx],
            })
        }

        assert_eq!(ip_by_line.intersections.len(), 12);

        let test_case = vec![
            dbscan::Classification::Core(0),
            dbscan::Classification::Edge(0),
            dbscan::Classification::Edge(0),
            dbscan::Classification::Noise,
            dbscan::Classification::Core(1),
            dbscan::Classification::Edge(1),
            dbscan::Classification::Edge(1),
            dbscan::Classification::Noise,
            dbscan::Classification::Edge(0),
            dbscan::Classification::Edge(0),
            dbscan::Classification::Edge(0),
            dbscan::Classification::Noise,
        ];
        ip_by_line.cluster_classes = Some(test_case.clone());

        assert_eq!(ip_by_line.search_for_biggest_cluster().unwrap(), 0);

        let mean_intersection = ip_by_line.get_mean_of_biggest_cluster().unwrap();

        assert_eq!(x_vec_c0.len(), 6);
        assert_eq!(y_vec_c0.len(), 6);
        let x_c0_mean: f64 = x_vec_c0.iter().sum::<f64>() as f64 / x_vec_c0.len() as f64;
        let y_c0_mean: f64 = y_vec_c0.iter().sum::<f64>() as f64 / y_vec_c0.len() as f64;

        assert_eq!(mean_intersection.x, x_c0_mean);
        assert_eq!(mean_intersection.y, y_c0_mean);
    }

    #[derive(serde_derive::Deserialize)]
    struct TestData {
        EstimateByLineIntersectionCluster :  td_eblic,
        wavelet_pos : WavletPos ,
        iz_pos : IzPos,
        iz_pos_sliding_window : IzPosSlidingWindow,
        iz_pos_sliding_window_zero : IzPosSlidingWindow,
        find_muap_data_already_windowd : FindMuapDataAlreadyWindowd,
    }

    #[derive(serde_derive::Deserialize)]
    struct IzPos {
        emg_array : Vec<Vec<f64>>,
        time_s    : Vec<f64>,
        estimated_pos : [f64;2],
        lambda : f64,
        epsilon : f64,
        electrode_distance : f64,
        expected_v_conduction : f64,
    }

    #[derive(serde_derive::Deserialize)]
    struct IzPosSlidingWindow {
        emg_array : Vec<Vec<f64>>,
        time_s    : Vec<f64>,
        t_ipzs : Vec<f64>,
        p_ipzs : Vec<f64>,
        window_width_s : f64,
        window_step_s : f64,
        lambda : f64,
        epsilon : f64,
        electrode_distance : f64,
        expected_v_conduction : f64,
    }

    #[derive(serde_derive::Deserialize)]
    struct td_eblic {
        MUAP_instances : Vec<f64>,
        electrode_positions : Vec<f64>,
    }

    #[derive(serde_derive::Deserialize)]
    struct WavletPos {
        emg : Vec<f64>,
        time_s : Vec<f64>,
        lambda : f64,
        estimated_pos : f64,
    }

    #[derive(serde_derive::Deserialize)]
    struct FindMuapDataAlreadyWindowd {
        lambda : f64,
        time   : Vec<f64>,
        input  : Vec<f64>,
    }

    fn load_test_data(datastr : &'static str) -> TestData {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("test_data");
        p.push(datastr);

        let test_config = fs::read_to_string(p).unwrap();
        let data : TestData = toml::from_str(&test_config).unwrap();

        return data;
    }

    #[test]
    fn ip_by_line_clusters_test() {
        let test_data = load_test_data("iz_estimation.toml");
        let time_vec  = test_data
            .EstimateByLineIntersectionCluster
            .MUAP_instances;

        //let elec_vec: Vec<f64> = (0..time_vec.len()).map(|x| x.to_f64().unwrap()).collect();
        let elec_vec = test_data
            .EstimateByLineIntersectionCluster
            .electrode_positions;

        let expected_iz_result = LineIntersection {
            x: 0.6262878205128205,
            y: 4.9404151404151335,
        };

        let mut ip_lc = ByLineCluster::new(
            time_vec,
            elec_vec,
            DbscanParameters {
                eps: 1.10083333,
                min_points: 3,
            },
            0.005, // 5 mm
            4.0,   // m/s
        );

        //let iz = ip_lc.predict_iz();

        ip_lc.calc_line_intersections();
        assert_eq!(ip_lc.intersections.len(), 42);

        let _i_n = ip_lc.normalize_intersections();

        ip_lc.cluster_intersections();

        println!("biggest_cluster: {:?}", ip_lc.search_for_biggest_cluster().unwrap());

        let _cluster_classes: Vec<i64> = ip_lc
            .cluster_classes
            .clone()
            .unwrap()
            .iter()
            .map(|x| dbscan_class_get_value(*x).unwrap_or(-1))
            .collect();

        let bc_a = ip_lc.search_for_biggest_cluster().unwrap();
        let bc_b = ip_lc.search_for_biggest_cluster().unwrap();

        assert_eq!(bc_a, bc_b);

        let mbc_a = ip_lc.get_mean_of_biggest_cluster().unwrap();
        let mbc_b = ip_lc.get_mean_of_biggest_cluster().unwrap();

        assert_eq!(mbc_a.x, mbc_b.x);
        assert_eq!(mbc_a.y, mbc_b.y);

        let iz = mbc_b;
        let error_iz = expected_iz_result - iz;

        assert_eq!(error_iz.x, 0.0);
    }

    #[test]
    fn test_muap_position () {
        let test_data = load_test_data("iz_estimation.toml");

        let test_estimate = muap_position(
            &test_data.wavelet_pos.time_s,
            &test_data.wavelet_pos.emg,
            test_data.wavelet_pos.lambda
        );

        assert_eq!(test_estimate, test_data.wavelet_pos.estimated_pos);



        let electrode_pos : Vec<f64> = (
            0..test_data.iz_pos_sliding_window.emg_array[0].len())
            .map(|x| x as f64)
            .collect();

        let mut emg_array = EmgArray::from_data_time_in_column(
            test_data.iz_pos_sliding_window.time_s.clone(),
            test_data.iz_pos_sliding_window.emg_array.clone(),
            electrode_pos
        );
        emg_array.transpose_data();


        let window_parameters :WindowParameters = WindowParameters::from_times(
            emg_array.time[0],
            emg_array.time[1],
            40e-3,
            20e-3).unwrap();

        let li : usize = 400;
        let lu : usize = 560;

        let d_hann_window = super::window_functions::discrete_hann_window(
            window_parameters.window_width,
            WindowKind::Periodic
        );
        let emg_array_slice = emg_array.get_slice_of_time_in_column(li, lu).unwrap();
        let emg_array_local = emg_array_slice.apply_window(&(d_hann_window.clone()));

        let mp = muap_position(
            &emg_array_local.time
            , &emg_array_local.data[6]
            , test_data.iz_pos_sliding_window.lambda);

        let mp_external = muap_position(
            &test_data.find_muap_data_already_windowd.time
            , &test_data.find_muap_data_already_windowd.input
            , test_data.find_muap_data_already_windowd.lambda);

        use plotters::prelude::*;

        {
            let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            p.push("test_plots");
            p.push("cmp_widowd_data.svg");
            let p = p;
            let figure = plotters_svg::SVGBackend::new(&p, (1024, 720)).into_drawing_area();

            let x_lim  = (
                 test_data.find_muap_data_already_windowd.time[0],
                *test_data.find_muap_data_already_windowd.time.last().unwrap());
            let y_lim  : (f64, f64) = (*test_data.find_muap_data_already_windowd.input.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).unwrap()
                , *test_data.find_muap_data_already_windowd.input.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap());

            figure.fill(&WHITE).unwrap();
            let mut ctx = ChartBuilder::on(&figure)
                .set_label_area_size(LabelAreaPosition::Left, 40)
                .set_label_area_size(LabelAreaPosition::Bottom, 40)
                .build_cartesian_2d(x_lim.0..x_lim.1, -10.0..5.0)
                .unwrap();

            ctx.configure_mesh().draw().unwrap();
            ctx.draw_series(
                LineSeries::new(
                    test_data.find_muap_data_already_windowd.time
                        .into_iter()
                        .zip(test_data.find_muap_data_already_windowd.input.clone())
                    , &BLACK
                )
            ).unwrap();
            ctx.draw_series(
                LineSeries::new(
                    emg_array_local.time.clone().into_iter().zip(emg_array_local.data[6].clone())
                    , &BLUE
                )
            ).unwrap();
            ctx.draw_series(
                LineSeries::new(
                    emg_array_local.time.clone().into_iter().zip(emg_array_slice.data[6].clone()).map(|(t,x)| (t,*x))
                    , &BLUE
                )
            ).unwrap();
        }

        let mae_window_cmp = test_data.find_muap_data_already_windowd.input
            .iter()
            .zip(&emg_array_local.data[6])
            .map(|(t,e)| (t-e).abs())
            .sum::<f64>() / emg_array_local.time.len() as f64;
        //assert!(float_cmp::approx_eq!(f64, mae_window_cmp, 0.0), "map : {mae_window_cmp}");

        assert_eq!(mp_external, 0.123);

        //assert_eq!(mp, 0.1085);
        assert_eq!(mp, 0.123);
        assert_eq!(mp_external, mp);

        // TODO this originates in differences in coralete with hermite_rodriguez wavelet_pos 
        // between python and rust implementation
        // this also seems to be the root of the problem in the comparison of 
        // the complete ip estimation for the sliding window
        //
        // Two ideas to investigate:
        //  PRIORITY
        //      - ich konnte das problem iengrenzen  auf die fensterung da gibt es ungereimtheiten
        //          obwohl die fnesterfunktion exact der in scipy entspricht!!
        //          betrifft vielleicht die SliceEmgArray::apply_window()
        //          oder irgentwas ist in der python variante komisch
        //
        //
        //      - the hermite_rodriguez wavlet is different -> ausgeschlossen
        //      - difference in the time shift of the wavelet -> vielleciht aber unwarscheinliche
        //
        //  Minor
        //      - a difference in corellation calclutaitin (alrady check that the 
        //        correlation ruslt seems to be different maybe plot the rust 
        //        version to confirm)
    }

    #[test]
    fn test_iz_estimation_on_single_window() {
        let test_data = load_test_data("iz_estimation.toml");
        assert_eq!(test_data.iz_pos.time_s.len(), test_data.iz_pos.emg_array.len());
        
        let iz_pos    = &test_data.iz_pos;

        let electrode_pos : Vec<f64> = (0..iz_pos.emg_array[0].len())
            .map(|x| x as f64)
            .collect();


        let mut emg_array = EmgArray::from_data_time_in_column(
            iz_pos.time_s.clone(),
            iz_pos.emg_array.clone(),
            electrode_pos,
        );
        emg_array.transpose_data();


        let estimate_for_linear_array = EstimateForLinearArray::new(
            iz_pos.lambda,
            iz_pos.epsilon,
            iz_pos.electrode_distance,
            iz_pos.expected_v_conduction);

        let res : IzEstimate = estimate_for_linear_array
            .single_array( &emg_array).unwrap();

        assert!(
            approx_eq!(f64, res.time, iz_pos.estimated_pos[0], ulps = 1),
            "time -> left: {t}, right: {x}", t = res.time, x = iz_pos.estimated_pos[0]
        );

        // TODO sanity check on high ulps
        //      may be useful to check the internal state i.e. clusters used
        //      for mean calculation
        //      Hoever the difference is whay below the expected 
        //      spacial resolution
        assert!(
            approx_eq!(f64, res.pos_in_array, iz_pos.estimated_pos[1], ulps = 35),
            "pos_in_array -> left: {t}, right: {x}", t = res.pos_in_array, x = iz_pos.estimated_pos[1]
        );
    }

    #[test]
    fn test_get_n_windows() {
        let wp = WindowParameters {
            window_width : 10,
            window_step  : 5,
        };
        assert_eq!( wp.get_n_windows(20), 3);

        let wp = WindowParameters {
            window_width : 10,
            window_step  : 2,
        };
        assert_eq!( wp.get_n_windows(20), 6);

        let wp = WindowParameters {
            window_width : 10,
            window_step  : 10,
        };
        assert_eq!( wp.get_n_windows(20), 2);
    }

    #[test]
    fn test_window_parameters() {
        let test_data = load_test_data("iz_estimation.toml");
        let test_data = test_data.iz_pos_sliding_window;

        let time = &test_data.time_s;

        let wp = WindowParameters::from_times(
            time[0], time[1],
            40e-3,
            20e-3).unwrap();

        assert_eq!(wp.window_width , 160);
        assert_eq!(wp.window_step,    80);
    }

    #[test]
    fn test_sliding_windows_iz_estimation_one_window(){
        let test_data = load_test_data("iz_estimation.toml");
        let test_data = test_data.iz_pos_sliding_window_zero;

        let electrode_pos : Vec<f64> = (0..test_data.emg_array[0].len())
            .map(|x| x as f64)
            .collect();

        assert_eq!(electrode_pos.len(), 15);


        let mut emg_array = EmgArray::from_data_time_in_column(
            test_data.time_s.clone(),
            test_data.emg_array.clone(),
            electrode_pos,
        );
        emg_array.transpose_data();

        assert_eq!(emg_array.data[0].len(), 160);

        let estimate_for_linear_array = EstimateForLinearArray::new(
            test_data.lambda,
            test_data.epsilon,
            test_data.electrode_distance,
            test_data.expected_v_conduction);

        let result = estimate_for_linear_array.with_sliding_window(
            &emg_array,
            test_data.window_width_s,
            test_data.window_step_s
        );

        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(result.len(), 1);
        let result = result[0].as_ref().unwrap();

        assert_eq!(test_data.t_ipzs[0], result.time);
        assert_eq!(test_data.p_ipzs[0], result.pos_in_array);
    }


    #[test]
    fn test_parralel_sliding_windows_iz_estimation() {
        let test_data = load_test_data("iz_estimation.toml");
        let test_data = test_data.iz_pos_sliding_window;

        let electrode_pos : Vec<f64> = (0..test_data.emg_array[0].len())
            .map(|x| x as f64)
            .collect();


        let mut emg_array = EmgArray::from_data_time_in_column(
            test_data.time_s.clone(),
            test_data.emg_array.clone(),
            electrode_pos,
        );
        emg_array.transpose_data();

        let estimate_for_linear_array = EstimateForLinearArray::new(
            test_data.lambda,
            test_data.epsilon,
            test_data.electrode_distance,
            test_data.expected_v_conduction);

        let result = estimate_for_linear_array.p_with_sliding_window(
            &emg_array,
            test_data.window_width_s,
            test_data.window_step_s,
            1
        );

        // TODO check difference in amount of slices compared to the
        // python version i suppose it is due to a partial slice at the end in the python version
        // or due to a missing slice in the rust version
        //
        // I bet it is the first option
        assert!(result.is_ok());
        let result = result.unwrap();
        println!("{:#?}", result);

        assert_eq!(result.len(), 7); // Last data is not a full window so its rejecte
                                     // this is intendet behavior but not the 
                                     // same as in the original impelemtation

        let target : Vec<(f64, f64)> = test_data.t_ipzs.into_iter().zip(test_data.p_ipzs).collect();

        for (num , (r, (t_target, p_target))) in result.iter().zip(target).enumerate() {
            if let Some(r) = r {
                let t_error = r.time - t_target;
                assert!(approx_eq!(f64, r.time, t_target),
                    "num : {num} with error in time {t_error}");
                let p_error = r.pos_in_array - p_target;
                assert!(approx_eq!(f64, r.pos_in_array, p_target, epsilon = 0.0000000000001),
                    "num: {num} with error in pos {p_error}\n {} {p_target}", r.pos_in_array);
            } else {
                assert_eq!(num, 4)
            }
        }
    }


    #[test]
    fn test_sliding_windows_iz_estimation() {
        let test_data = load_test_data("iz_estimation.toml");
        let test_data = test_data.iz_pos_sliding_window;

        let electrode_pos : Vec<f64> = (0..test_data.emg_array[0].len())
            .map(|x| x as f64)
            .collect();


        let mut emg_array = EmgArray::from_data_time_in_column(
            test_data.time_s.clone(),
            test_data.emg_array.clone(),
            electrode_pos,
        );
        emg_array.transpose_data();

        let estimate_for_linear_array = EstimateForLinearArray::new(
            test_data.lambda,
            test_data.epsilon,
            test_data.electrode_distance,
            test_data.expected_v_conduction);

        let result = estimate_for_linear_array.with_sliding_window(
            &emg_array,
            test_data.window_width_s,
            test_data.window_step_s,
        );

        // TODO check difference in amount of slices compared to the
        // python version i suppose it is due to a partial slice at the end in the python version
        // or due to a missing slice in the rust version
        //
        // I bet it is the first option
        assert!(result.is_ok());
        let result = result.unwrap();
        println!("{:#?}", result);

        assert_eq!(result.len(), 7); // Last data is not a full window so its rejecte
                                     // this is intendet behavior but not the 
                                     // same as in the original impelemtation

        let target : Vec<(f64, f64)> = test_data.t_ipzs.into_iter().zip(test_data.p_ipzs).collect();

        for (num , (r, (t_target, p_target))) in result.iter().zip(target).enumerate() {
            if let Some(r) = r {
                let t_error = r.time - t_target;
                assert!(approx_eq!(f64, r.time, t_target),
                    "num : {num} with error in time {t_error}");
                let p_error = r.pos_in_array - p_target;
                assert!(approx_eq!(f64, r.pos_in_array, p_target, epsilon = 0.0000000000001),
                    "num: {num} with error in pos {p_error}\n {} {p_target}", r.pos_in_array);
            } else {
                assert_eq!(num, 4)
            }
        }
    }
}

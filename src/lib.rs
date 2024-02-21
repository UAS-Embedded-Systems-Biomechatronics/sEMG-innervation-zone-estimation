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

use pyo3::prelude::*;
use sEMG_innervation_zone_estimation::EstimateForLinearArray;
use sEMG_innervation_zone_estimation::data::EmgArray;


#[pyclass]
pub struct IzEstimation {
    #[pyo3(get,set)] pub window_width_s : f64,
    #[pyo3(get,set)] pub window_step_s : f64,
    #[pyo3(get,set)] pub lam : f64,
    #[pyo3(get,set)] pub epsilon : f64,
    #[pyo3(get,set)] pub electrode_distance : f64,
    #[pyo3(get,set)] pub expected_v_conduction : f64,
}

#[pymethods]
impl IzEstimation {
    #[new]
    fn new(
        window_width_s : f64,
        window_step_s : f64,
        lam : f64,
        epsilon : f64,
        electrode_distance : f64,
        expected_v_conduction : f64) -> Self {
        IzEstimation { 
            window_width_s,
            window_step_s,
            lam,
            epsilon,
            electrode_distance,
            expected_v_conduction}
    }

    fn find_IPs_parallel(&self,
        time_s    : Vec<f64>,
        emg_array : Vec<Vec<f64>>,
        electrode_pos : Vec<f64>,
        n_worker : usize
    ) -> [Vec<f64>; 2]{

        let mut emg_array = EmgArray::from_data_time_in_column(
            time_s.clone(),
            emg_array.clone(),
            electrode_pos,
        );
        emg_array.transpose_data();

        let estimate_for_linear_array = EstimateForLinearArray::new(
            self.lam,
            self.epsilon,
            self.electrode_distance,
            self.expected_v_conduction);

        let r = estimate_for_linear_array.p_with_sliding_window(
            &emg_array,
            self.window_width_s,
            self.window_step_s,
            n_worker
        );

        let mut result = [Vec::<f64>::new(), Vec::<f64>::new()];
        if let Ok(r) = r {
            for item in r {
                if let Some(item) = item {
                    result[0].push(item.time);
                    result[1].push(item.pos_in_array);
                }
            }
        }

        result
    }

    fn find_IPs(&self,
        time_s    : Vec<f64>,
        emg_array : Vec<Vec<f64>>,
        electrode_pos : Vec<f64>
    ) -> [Vec<f64>; 2]{

        let mut emg_array = EmgArray::from_data_time_in_column(
            time_s.clone(),
            emg_array.clone(),
            electrode_pos,
        );
        emg_array.transpose_data();

        let estimate_for_linear_array = EstimateForLinearArray::new(
            self.lam,
            self.epsilon,
            self.electrode_distance,
            self.expected_v_conduction);

        let r = estimate_for_linear_array.with_sliding_window(
            &emg_array,
            self.window_width_s,
            self.window_step_s,
        );

        let mut result = [Vec::<f64>::new(), Vec::<f64>::new()];
        if let Ok(r) = r {
            for item in r {
                if let Some(item) = item {
                    result[0].push(item.time);
                    result[1].push(item.pos_in_array);
                }
            }
        }

        result
    }
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name= "_lib")]
fn iz_tracking(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<IzEstimation>()?;
    Ok(())
}





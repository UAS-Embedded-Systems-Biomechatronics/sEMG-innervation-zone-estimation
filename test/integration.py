# Copyright 2023 Malte Mechtenberg
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import unittest
import toml
import semg_iz_estimation

import numpy as np

N_PROC = 4
TEST_CASE_FILES = {
    'IZ_ESTIMATION' : "../sEMG_innervation_zone_estimation/test_data/iz_estimation.toml"
}

class test_BaseConfig(unittest.TestCase):

    def test_iz_estimation(self):
        test_data = toml.load(TEST_CASE_FILES["IZ_ESTIMATION"])['iz_pos_sliding_window']
        #['electrode_distance', 'expected_v_conduction', 'lambda', 'epsilon',
        #'window_width_s', 'window_step_s', 't_ipzs', 'p_ipzs', 'time_s', 'emg_array']

        ize = semg_iz_estimation.IzEstimation(
            window_width_s = test_data['window_width_s'],
            window_step_s  = test_data['window_step_s'],
            lam = test_data['lambda'],
            epsilon = test_data['epsilon'],
            electrode_distance = test_data['electrode_distance'],
            expected_v_conduction = test_data['expected_v_conduction']
        ) 

        ep = list(range(15))

        r_single_trhead = ize.find_IPs(
            time_s = test_data['time_s'],
            emg_array = test_data['emg_array'],
            electrode_pos = ep)
        r_single_trhead = np.array(r_single_trhead)

        r_n_threads = ize.find_IPs_parallel(
            time_s = test_data['time_s'],
            emg_array = test_data['emg_array'],
            electrode_pos = ep,
            n_worker = N_PROC
        )
        r_n_threads = np.array(r_n_threads)

        self.assertTrue((r_single_trhead == r_n_threads).all())

        t_ipzs = np.delete(np.array(test_data['t_ipzs']), 4)[:-1]
        p_ipzs = np.delete(np.array(test_data['p_ipzs']), 4)[:-1]

        for (id, (target_t, target_p)) in enumerate(zip(t_ipzs, p_ipzs)):
            print("target : [{}, {}]".format(target_t, target_p))
            print("acutal :   {}".format(r_n_threads[:, id]))

            self.assertAlmostEqual(r_n_threads[0, id], target_t)
            self.assertAlmostEqual(r_n_threads[1, id], target_p)


if __name__ == '__main__':
    unittest.main()

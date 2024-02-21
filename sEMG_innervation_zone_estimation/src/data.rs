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

#[derive(Clone,Copy,Debug,PartialEq)]
pub enum Orientation {
    TimeInRow,
    TimeInColumn,
}

impl Orientation {
    fn invert(&self) -> Orientation {
        match *self {
            Self::TimeInRow => Self::TimeInColumn,
            Self::TimeInColumn => Self::TimeInRow,
        }
    }
}

#[derive(Clone)]
pub struct EmgArray {
    pub time : Vec<f64>,
    pub data : Vec<Vec<f64>>,
    pub electrode_x : Vec<f64>,
    pub orientation : Orientation,
}

impl EmgArray {
    pub fn from_data_time_in_row(
        time : Vec<f64>,
        data : Vec<Vec<f64>>,
        electrode_x : Vec<f64>) -> EmgArray {

        let orientation = Orientation::TimeInRow;

        EmgArray {
            time,
            data,
            electrode_x,
            orientation
        }
    }

    pub fn from_data_time_in_column(
        time : Vec<f64>,
        data : Vec<Vec<f64>>,
        electrode_x : Vec<f64>) -> EmgArray {

        let orientation = Orientation::TimeInColumn;

        EmgArray {
            time,
            data,
            electrode_x,
            orientation
        }
    }

    pub fn transpose_data(&mut self) {
        self.data  = (0..self.data[0].len()).map(|col| {
            (0..self.data.len())
                .map(|row| self.data[row][col])
                .collect()
        }).collect();

        self.orientation = self.orientation.invert();
    }

    pub fn get_slice_of_time_in_column<'a>(
        &'a self,
        start:usize,
        end:usize
    ) -> Result<SliceEmgArray<'a>, &'static str> {
        if Orientation::TimeInColumn == self.orientation {
            return Err("Data must have the property  Orientation::TimeInColumn");
        }

        let time = &self.time[start..end];
        let data : Vec<&'a [f64]> = self.data
            .iter()
            .map( |x| & x[start..end] )
            .collect();

        let electrode_x = &self.electrode_x;
        let orientation = &self.orientation;

        Result::Ok(SliceEmgArray { time, data, electrode_x, orientation })
    }
}

pub struct SliceEmgArray<'a> {
    pub time : &'a[f64],
    pub data : Vec<&'a [f64]>,
    pub electrode_x : &'a Vec<f64>,
    pub orientation : &'a Orientation,
}

impl SliceEmgArray<'_> {
    ///Assumes the date in `Orientation::TimeInRow`
    ///TODO test with hann window on simple example
    pub fn apply_window(& self, w : &Vec<f64> ) -> EmgArray {
        assert_eq!(*self.orientation, Orientation::TimeInRow);
        assert_eq!(self.time.len(), w.len());

        let data : Vec<Vec<f64>> = self.data
            .iter().map(|&d| d.iter().zip(w).map(|(d,w)| d*w).collect())
            .collect();

        EmgArray { 
            time: self.time.to_vec(),
            data,
            electrode_x: self.electrode_x.to_vec(),
            orientation: *self.orientation 
        }
    }

    pub fn clone(& self) -> EmgArray {
        let data : Vec<Vec<f64>> = self.data
            .iter().map(|&d| d.to_vec())
            .collect();

        EmgArray { 
            time: self.time.to_vec(),
            data,
            electrode_x: self.electrode_x.to_vec(),
            orientation: *self.orientation 
        }
    }
    
}


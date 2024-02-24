extern crate csv;
use candle_core::{Device, Tensor};
use core::panic;
use linear_regression::{r2_score, LinearRegression};
use std::fs::File;

const MALE: f32 = 0.5;
const FEMALE: f32 = -0.5;

const YES: f32 = 0.5;
const NO: f32 = -0.5;

const NORTHWEST: f32 = 0.25;
const NORTHEAST: f32 = -0.25;
const SOUTHWEST: f32 = 0.5;
const SOUTHEAST: f32 = -0.5;

const LEARNING_RATE: f32 = 0.01;
const ITERATIONS: i32 = 100000;

fn data_labels() -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    // https://www.kaggle.com/mirichoi0218/insurance

    let file = File::open("insurance.csv")?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut features: Vec<Vec<f32>> = vec![];
    let mut labels: Vec<f32> = vec![];

    for result in rdr.records() {
        let record = result?;
        let age: f32 = (record[0].parse::<u32>()? as f32) / 100.0;
        let gender = match record[1].parse::<String>()?.as_str() {
            "male" => MALE,
            "female" => FEMALE,
            _ => panic!("Invalid Gender"),
        };
        let bmi: f32 = record[2].parse::<f32>()? / 100.0;
        let children: f32 = record[3].parse()?;
        let smoker = match record[4].parse::<String>()?.as_str() {
            "yes" => YES,
            "no" => NO,
            _ => panic!("Invalid Smoker"),
        };
        let region = match record[5].parse::<String>()?.as_str() {
            "northwest" => NORTHWEST,
            "northeast" => NORTHEAST,
            "southwest" => SOUTHWEST,
            "southeast" => SOUTHEAST,
            _ => panic!("Invalid Region"),
        };
        let charges: f32 = record[6].parse()?;

        let row = vec![1.0, age, gender, bmi, children, smoker, region];
        features.push(row);

        let label = charges;
        labels.push(label);
    }
    let features = features.iter().flatten().copied().collect::<Vec<f32>>();
    Ok((features, labels))
}

// https://www.youtube.com/watch?v=UVCFaaEBnTE

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available(0)?;

    let (data, labels) = data_labels()?;
    let training_size = (labels.len() as f32 * 0.8) as usize;
    let feature_cnt = data.len() / labels.len();

    let train_data = &data[0..training_size * feature_cnt];
    let train_data = Tensor::from_slice(train_data, (training_size, feature_cnt), &device)?;

    let train_labels = &labels[0..training_size];
    let train_labels = Tensor::from_slice(train_labels, (training_size,), &device)?;

    let mut model = LinearRegression::new(feature_cnt, true)?;

    for _ in 0..ITERATIONS {
        model.train(&train_data, &train_labels, LEARNING_RATE)?;
    }

    let test_data = &data[training_size * feature_cnt..];
    let len = test_data.len();
    let test_labels = Tensor::from_slice(&labels[training_size..], (len / feature_cnt,), &device)?;
    let test_data = Tensor::from_slice(test_data, (len / feature_cnt, feature_cnt), &device)?;

    let predictions = model.predict(&test_data)?;
    let r2 = r2_score(&predictions, &test_labels)?;
    println!("r2: {r2}");

    Ok(())
}

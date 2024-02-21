extern crate csv;
use candle_core::{backprop::GradStore, cuda_backend::cudarc::driver::result::device, Device, Tensor, D};
use core::panic;
use std::fs::File;

const MALE: f32 = 0.0;
const FEMALE: f32 = 1.0;

const YES: f32 = 1.0;
const NO: f32 = 0.0;

const NORTHWEST: f32 = 0.1;
const NORTHEAST: f32 = 0.2;
const SOUTHWEST: f32 = 0.3;
const SOUTHEAST: f32 = 0.4;

const LEARNING_RATE: f32 = 0.1;
const ITERATIONS: i32 = 100000;

fn data_labels() -> Result<(Vec<Vec<f32>>, Vec<f32>), Box<dyn std::error::Error>> {
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
    Ok((features, labels))
}

fn cost_function(
    data: &Tensor,
    labels: &Tensor,
    cofficients: &Tensor,
    device: &Device,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let (m, _) = data.shape().dims2()?;
    let predictions = data.matmul(&cofficients.unsqueeze(1)?)?;
    let errors = predictions.squeeze(1)?.sub(labels)?;
    let cost = errors
        .mul(&errors)?
        .mean(D::Minus1)?
        .div(&Tensor::new(2.0 * m as f32, &device)?)?;
    Ok(cost)
}

fn next_cofficients(
    data: &Tensor,
    labels: &Tensor,
    cofficients: &Tensor,
    device: &Device,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let (m, _) = data.shape().dims2()?;
    let predictions = data.matmul(&cofficients.unsqueeze(1)?)?;
    let errors = predictions.squeeze(1)?.sub(labels)?;
    let gradient = data
        .t()?
        .matmul(&errors.unsqueeze(D::Minus1)?)?.unsqueeze(D::Minus1)?
        .broadcast_div(&Tensor::new(m as f32, &device)?)?;
    let gradient = gradient.squeeze(D::Minus1)?.squeeze(D::Minus1)?;
    let cofficients =
        cofficients.sub(&gradient.broadcast_mul(&Tensor::new(LEARNING_RATE as f32, &device)?)?)?;
    Ok(cofficients)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available(0)?;

    let (data, labels) = data_labels()?;
    let training_size = 100; //(data.len() as f32 * 0.8) as usize;
    let feature_cnt = data[0].len();

    let cofficients: Vec<f32> = vec![0.0; feature_cnt];
    let mut cofficients = Tensor::from_vec(cofficients, (feature_cnt,), &device)?;

    let train_data = &data[0..training_size];
    let train_data = train_data.iter().flatten().copied().collect::<Vec<f32>>();
    let train_data = Tensor::from_vec(train_data, (training_size, feature_cnt), &device)?;

    let train_labels = &labels[0..training_size];
    let train_labels = Tensor::from_slice(train_labels, (training_size,), &device)?;

    let mut cost: Tensor = Tensor::from_slice(&[0.0],(1,), &device)?;
    for i in 0..ITERATIONS {
        cost = cost_function(&train_data, &train_labels, &cofficients, &device)?;
        cofficients = next_cofficients(&train_data, &train_labels, &cofficients, &device)?;
    }

    println!("iterations: {ITERATIONS} cost: {cost}" );

    let test_data = &data[training_size..];
    let test_labels = &labels[training_size..];

    Ok(())
}

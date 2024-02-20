extern crate csv;
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

const LEARNING_RATE: f32 = 0.01;

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

        println!(
            "{age} {:?} {bmi} {children} {:?} {:?} {charges}",
            gender, smoker, region
        );
    }
    Ok((features, labels))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (data, labels) = data_labels()?;
    let training_size = (data.len() as f32 * 0.8) as usize;
    let feature_cnt = data[0].len();

    let cofficients = vec![0.0; feature_cnt];

    let train_data = &data[0..training_size];
    let train_labels = &labels[0..training_size];

    let test_data = &data[training_size..];
    let test_labels = &labels[training_size..];

    Ok(())
}

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

fn features_labels() -> Result<(Vec<Vec<f32>>, Vec<f32>), Box<dyn std::error::Error>> {
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

        let row = vec![age, gender, bmi, children, smoker, region];
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
    let (features, labels) = features_labels()?;
    let training_size = (features.len() as f32 * 0.8) as usize;

    let train_features = &features[0..training_size];
    let train_labels = &labels[0..training_size];

    let test_features = &features[training_size..];
    let test_labels = &labels[training_size..];

    Ok(())
}

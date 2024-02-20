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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("insurance.csv")?;
    let mut rdr = csv::Reader::from_reader(file);
    for result in rdr.records() {
        let record = result?;
        let age: f32 = record[0].parse()?;
        let gender = match record[1].parse::<String>()?.as_str() {
            "male" => MALE,
            "female" => FEMALE,
            _ => panic!("Invalid Gender"),
        };
        let bmi: f32 = record[2].parse()?;
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

        let features = vec![age, gender, bmi, children, smoker, region];
        let y = charges;

        println!(
            "{age} {:?} {bmi} {children} {:?} {:?} {charges}",
            gender, smoker, region
        );
    }

    Ok(())
}

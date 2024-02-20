extern crate csv;
use core::panic;
use std::fs::File;

#[derive(Debug)]
enum Gender {
    Male = 0,
    Female = 1,
}

#[derive(Debug)]
enum Region {
    NorthWest = 0,
    NorthEast = 1,
    SouthWest = 2,
    SouthEast = 3,
}

#[derive(Debug)]
enum Smoker {
    Yes = 0,
    No = 1,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("insurance.csv")?;
    let mut rdr = csv::Reader::from_reader(file);
    for result in rdr.records() {
        let record = result?;
        let age: f32 = record[0].parse()?;
        let gender = match record[1].parse::<String>()?.as_str() {
            "male" => Gender::Male,
            "female" => Gender::Female,
            _ => panic!("Invalid Gender"),
        };
        let bmi: f32 = record[2].parse()?;
        let children: f32 = record[3].parse()?;
        let smoker = match record[4].parse::<String>()?.as_str() {
            "yes" => Smoker::Yes,
            "no" => Smoker::No,
            _ => panic!("Invalid Smoker"),
        };
        let region = match record[5].parse::<String>()?.as_str() {
            "northwest" => Region::NorthWest,
            "northeast" => Region::NorthEast,
            "southwest" => Region::SouthWest,
            "southeast" => Region::SouthEast,
            _ => panic!("Invalid Region"),
        };
        let charges: f32 = record[6].parse()?;
        println!(
            "{age} {:?} {bmi} {children} {:?} {:?} {charges}",
            gender, smoker, region
        );
    }

    Ok(())
}

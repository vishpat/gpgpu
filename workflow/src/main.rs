use candle_datasets::vision::mnist;
use candle_core::{Tensor, DType, Device};

fn log_softmax(x: &candle_core::Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(x.sub(&x.exp()?.sum(1)?.unsqueeze(1)?)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>>{
    let dataset = mnist::load()?;
    println!("Train images: {:?}", dataset.train_images);
    println!("Train labels: {:?}", dataset.train_labels);
    println!("Test images: {:?}", dataset.test_images);
    println!("Test labels: {:?}", dataset.test_labels);
    Ok(())
}

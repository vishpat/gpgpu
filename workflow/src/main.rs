use candle_core::{Device, Tensor, D};
use candle_datasets::vision::mnist;
use candle_nn::ops::log_softmax;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = mnist::load()?;
    println!("Train images: {:?}", dataset.train_images);
    println!("Train labels: {:?}", dataset.train_labels);
    println!("Test images: {:?}", dataset.test_images);
    println!("Test labels: {:?}", dataset.test_labels);
    let device = Device::cuda_if_available(0)?;
    let tensor = Tensor::arange(0f32, 3f32, &device)?;
    println!("Tensor {tensor}");
    let tensor = log_softmax(&tensor, D::Minus1)?;
    println!("Log softmax Tensor {tensor}");
    Ok(())
}

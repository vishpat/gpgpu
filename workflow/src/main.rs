use candle_datasets::vision::mnist;
use candle_core::{Tensor, DType, Device};
use candle_nn::ops::log_softmax;

fn log_softmax2(x: &candle_core::Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    let max_x = x.max(0)?.unsqueeze(0)?;
    let sub_x = x.broadcast_sub(&max_x)?;
    let exp_x = sub_x.exp()?;
    let sum_exp_x = exp_x.sum(0)?;
    let exp_x = exp_x.broadcast_div(&sum_exp_x)?;
    Ok(exp_x.log()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>>{
//    let dataset = mnist::load()?;
//    println!("Train images: {:?}", dataset.train_images);
//    println!("Train labels: {:?}", dataset.train_labels);
//    println!("Test images: {:?}", dataset.test_images);
//    println!("Test labels: {:?}", dataset.test_labels);
    let device = Device::cuda_if_available(0)?;
    let tensor = Tensor::arange(0f32, 3f32, &device)?;
    println!("Tensor {tensor}");
    let tensor = log_softmax2(&tensor)?;
    println!("Log softmax Tensor {tensor}");
    Ok(())
}

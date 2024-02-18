use candle_core::{Device, Tensor, DType};
use candle_datasets::vision::mnist;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = mnist::load()?;
    let (img_cnt, img_size) = dataset.train_images.shape().dims2()?;
    let device = Device::cuda_if_available(0)?;

    // Normalize the image data
    let images = dataset
        .train_images
        .to_device(&device)?
        .broadcast_div(&Tensor::new(255f32, &device)?)?;
    println!("{images}");

    // All none-zero labels are converted to 1
    let labels = dataset
        .train_labels
        .to_device(&device)?
        .to_dtype(DType::F32)?
        .broadcast_minimum(&Tensor::new(1f32, &device)?)?;
    println!("{labels}");

    Ok(())
}

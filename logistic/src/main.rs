use candle_core::{DType, Device, Tensor};
use candle_datasets::vision::mnist;

//
// Implementation of the logistic regression model as shown at
// https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/
//
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = mnist::load()?;
    let (img_cnt, img_size) = dataset.train_images.shape().dims2()?;
    let device = Device::cuda_if_available(0)?;

    // Normalize the image data
    let images = dataset
        .train_images
        .to_device(&device)?
        .broadcast_div(&Tensor::new(255f32, &device)?)?;

    // All zero labels are converted to 1
    let labels = dataset
        .train_labels
        .to_device(&device)?
        .to_dtype(DType::F32)?
        .broadcast_eq(&Tensor::new(0f32, &device)?)?;

    Ok(())
}

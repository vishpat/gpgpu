use candle_core::{Device, Tensor, D};
use candle_datasets::vision::mnist;
use candle_nn::ops::log_softmax;

const BATCH_SIZE: usize = 100;

fn model(batch: &Tensor, weights: &Tensor, bias: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    let output = batch.matmul(weights)?.add(bias)?;
    let output = log_softmax(&output, D::Minus1)?;
    Ok(output)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = mnist::load()?;
    println!("Train images: {:?}", dataset.train_images);
    println!("Train labels: {:?}", dataset.train_labels);
    
    let (img_cnt, img_size) = dataset.train_images.shape().dims2()?;
    println!("Image count: {:?}, Image size: {:?}", img_cnt, img_size);

    println!("Test images: {:?}", dataset.test_images);
    println!("Test labels: {:?}", dataset.test_labels);
    
    let device = Device::cuda_if_available(0)?;
    let weights = Tensor::rand(0f32, 1f32, (784, 10), &device)?; 
    let bias = Tensor::zeros((10,), candle_core::DType::F32, &device)?;

    let batch_cnt = (img_cnt / BATCH_SIZE) as usize;
    println!("Batch count: {:?}", batch_cnt);

    let img_batches = dataset.train_images.chunk(batch_cnt, D::Minus2)?;
    println!("Image batch count: {:?}", img_batches.len());

    let label_batches = dataset.train_labels.chunk(batch_cnt, D::Minus1)?;
    println!("Label batch count: {:?}", label_batches.len());

    for i in 0..1 {
        let batch = &img_batches[i];
        println!("Image Batch Shape: {:?}", batch.shape());

        let labels = &label_batches[i];
        println!("Label Batch Shape: {:?}", labels.shape());
    }

    Ok(())
}

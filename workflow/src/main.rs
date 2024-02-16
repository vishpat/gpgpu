use candle_core::{Device, Tensor, D};
use candle_datasets::vision::mnist;
use candle_nn::ops::log_softmax;

const BATCH_SIZE: usize = 100;

fn model(
    batch: &Tensor,
    weights: &Tensor,
    bias: &Tensor,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(batch.matmul(weights)?.broadcast_add(&bias)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = mnist::load()?;
    let (img_cnt, img_size) = dataset.train_images.shape().dims2()?;
    let device = Device::cuda_if_available(0)?;
    let weights = Tensor::rand(0f32, 1f32, (img_size, 10), &device)?;
    let bias = Tensor::zeros((10,), candle_core::DType::F32, &device)?;
    
    let batch_cnt = (img_cnt / BATCH_SIZE) as usize;
    let img_batches = dataset.train_images.chunk(batch_cnt, D::Minus2)?;
    let label_batches = dataset.train_labels.chunk(batch_cnt, D::Minus1)?;
    println!("Label batch count: {:?}", label_batches.len());

    for i in 0..1 {
        let batch = &img_batches[i];
        let labels = &label_batches[i];

        println!(
            "Batch Shape: {:?}, Weight Shape {:?}, Bias Shape {:?}",
            batch.shape(),
            weights.shape(),
            bias.shape()
        );

        let output = model(batch, &weights, &bias)?;
        println!("Output: {:?}", output);
    }

    Ok(())
}

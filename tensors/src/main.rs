use candle_core::{Device, Tensor};
use ndarray::array;

#[allow(dead_code)]
fn example1() -> Result<(), Box<dyn std::error::Error>> {
    let arr = array![[1., 2.], [4., 5.]];
    println!("ndarray: {:?}", arr);

    let arr_slice = arr.into_shape((2, 2)).to_owned()?.into_raw_vec();
    println!("Slice: {:?}", arr_slice);

    let device = Device::new_cuda(0)?;
    let tensor = Tensor::from_slice(&arr_slice, (2, 2), &device);
    println!("Candle {:?}", tensor);
    Ok(())
}

fn example2() -> Result<(), Box<dyn std::error::Error>> {
    let arr = array![[1., 2.], [4., 5.]];
    let arr_slice = arr.into_shape((2, 2)).to_owned()?.into_raw_vec();
    let device = Device::new_cuda(0)?;
    let tensor = Tensor::from_slice(&arr_slice, (2, 2), &device)?;
    let tensor2 = Tensor::rand(0f32, 1., tensor.shape(), &device)?;
    let slice = tensor2.to_vec2::<f32>()?;
    println!("Candle {:?}", slice);
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    example2()?;
    Ok(())
}

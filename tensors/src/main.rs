use candle_core::{DType, Device, Tensor};
use ndarray::array;

#[allow(dead_code)]
fn example0() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let scalar_tensor = Tensor::new(std::f32::consts::PI, &device)?;
    let scalar = scalar_tensor.to_scalar::<f32>()?;
    println!("{scalar}");
    Ok(())
}

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

#[allow(dead_code)]
fn example2() -> Result<(), Box<dyn std::error::Error>> {
    let arr = array![[1., 2., 3.], [4., 5., 6.]];
    let arr_slice = arr.into_shape((2, 3)).to_owned()?.into_raw_vec();
    let device = Device::new_cuda(0)?;
    let tensor = Tensor::from_slice(&arr_slice, (2, 3), &device)?;

    let tensor_ones = Tensor::ones(tensor.shape(), DType::F32, &device)?;
    println!("Ones tensor {tensor_ones}");

    let tensor_rand = Tensor::rand(0f32, 1., tensor.shape(), &device)?;
    println!("Rand tensor {tensor_rand}");

    Ok(())
}

#[allow(dead_code)]
fn example3() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let tensor = Tensor::rand(0f32, 1., (2, 4), &device)?;

    println!("Tensor Shape {:?}", tensor.shape());
    println!("Tensor DType {:?}", tensor.dtype());
    println!("Tensor Device {:?}", tensor.device());

    Ok(())
}

#[allow(dead_code)]
fn example4() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let tensor1 = Tensor::ones((4, 4), DType::F32, &device)?;
    let tensor2 = Tensor::zeros((4, 4), DType::F32, &device)?;
    let tensor = Tensor::cat(&[&tensor1, &tensor2], 0)?;
    println!("Tensor {tensor}");
    Ok(())
}

#[allow(dead_code)]
fn example5() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let tensor1 = Tensor::rand(0f32, 1., (4, 4), &device)?;
    let tensor2 = Tensor::rand(0f32, 1., (4, 4), &device)?;
    let tensor = Tensor::mul(&tensor1, &tensor2)?;
    println!("Tensor {tensor}");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    example0()?;
    Ok(())
}

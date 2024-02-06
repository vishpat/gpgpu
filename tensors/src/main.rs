use candle_core::{DType, Device, Tensor};
use ndarray::array;

#[allow(dead_code)]
fn scalar() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let scalar_tensor = Tensor::new(std::f32::consts::PI, &device)?;
    let scalar = scalar_tensor.to_scalar::<f32>()?;
    println!("{scalar}");
    println!("ndim {:?}", scalar_tensor.dims().len());
    println!("shape {:?}", scalar_tensor.shape());
    Ok(())
}

#[allow(dead_code)]
fn vector() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let vector = vec![1., 2., 3., 4.];
    let vector_tensor = Tensor::from_slice(&vector, (4,), &device)?;
    println!("{vector_tensor}");
    println!("ndim {:?}", vector_tensor.dims().len());
    println!("shape {:?}", vector_tensor.shape());
    Ok(())
}

#[allow(dead_code)]
fn init1() -> Result<(), Box<dyn std::error::Error>> {
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
fn init2() -> Result<(), Box<dyn std::error::Error>> {
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
fn info() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let tensor = Tensor::rand(0f32, 1., (2, 4), &device)?;

    println!("Tensor Shape {:?}", tensor.shape());
    println!("Tensor DType {:?}", tensor.dtype());
    println!("Tensor Device {:?}", tensor.device());

    Ok(())
}

#[allow(dead_code)]
fn concat() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let tensor1 = Tensor::ones((4, 4), DType::F32, &device)?;
    let tensor2 = Tensor::zeros((4, 4), DType::F32, &device)?;
    let tensor = Tensor::cat(&[&tensor1, &tensor2], 0)?;
    println!("Tensor {tensor}");
    Ok(())
}

#[allow(dead_code)]
fn multiple() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let tensor1 = Tensor::rand(0f32, 1., (4, 4), &device)?;
    let tensor2 = Tensor::rand(0f32, 1., (4, 4), &device)?;
    let tensor = Tensor::mul(&tensor1, &tensor2)?;
    println!("Tensor {tensor}");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    vector()?;
    Ok(())
}

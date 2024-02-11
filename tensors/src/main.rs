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
fn init_matrix1() -> Result<(), Box<dyn std::error::Error>> {
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
fn init_matrix2() -> Result<(), Box<dyn std::error::Error>> {
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

#[allow(dead_code)]
fn range() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let tensor1 = Tensor::arange(0f32, 6f32, &device)?;
    println!("Tensor {tensor1}");

    let tensor1 = Tensor::arange(0f32, 6f32, &device)?.reshape((2, 3))?;
    println!("Tensor {tensor1}");

    let tensor2 = Tensor::arange(0f32, 12f32, &device)?.reshape((3, 4))?;
    println!("Tensor {tensor2}");

    let tensor = tensor1.matmul(&tensor2)?;
    println!("Tensor {tensor}");
    Ok(())
}

#[allow(dead_code)]
fn broadcast_add() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let tensor1 = Tensor::arange(0f32, 6f32, &device)?.reshape((2, 3))?;
    println!("Tensor {tensor1}");

    let tensor = tensor1.broadcast_add(&Tensor::new(10f32, &device)?)?;
    println!("Tensor {tensor}");

    Ok(())
}

#[allow(dead_code)]
fn broadcast_mul() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let tensor1 = Tensor::arange(0f32, 6f32, &device)?.reshape((2, 3))?;
    println!("Tensor {tensor1}");

    let tensor = tensor1.broadcast_mul(&Tensor::new(10f32, &device)?)?;
    println!("Tensor {tensor}");

    Ok(())
}

#[allow(dead_code)]
fn indexing() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let rand_tensor = Tensor::rand(0f32, 1., (2, 2, 2), &device)?;
    println!("Tensor {rand_tensor}");

    let index_tensor = Tensor::new(&[0u32], &device)?;
    println!("Index tensor {index_tensor}");

    let selected = rand_tensor.index_select(&index_tensor, 2)?;
    println!("Selected tensor {selected}");
    Ok(())
}

#[allow(dead_code)]
fn argmax() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let tensor = Tensor::arange(0f32, 10f32, &device)?;
    println!("Tensor {tensor}");

    let argmax = tensor.argmax(0)?;
    println!("Argmax {argmax}");

    let tensor = Tensor::rand(0f32, 1., (3, 3), &device)?;
    println!("Tensor {tensor}");

    let argmax = tensor.argmax(0)?;
    println!("Argmax: dim 0  {argmax}");

    let argmax = tensor.argmax(1)?;
    println!("Argmax: dim 1  {argmax}");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    argmax()?;
    Ok(())
}

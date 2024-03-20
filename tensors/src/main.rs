use candle_core::{DType, Device, Tensor, D};
use ndarray::array;

fn cdist(x1: &Tensor, x2: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    let dim1 = x1.shape().dims2()?;
    let x1 = x1.unsqueeze(0)?;

    let dim2 = x2.shape().dims2()?;
    let x2 = x2.unsqueeze(1)?;
    Ok(x1
        .broadcast_sub(&x2)?
        .sqr()?
        .sum(D::Minus1)?
        .sqrt()?
        .transpose(D::Minus1, D::Minus2)?)
}

fn zscore_normalize(data: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mean = data.mean(0)?;
    let squared_diff = data.broadcast_sub(&mean)?.sqr()?;
    let variance = squared_diff.mean(0)?;
    let std_dev = variance.sqrt()?;
    let normalized = data.broadcast_sub(&mean)?.broadcast_div(&std_dev)?;
    Ok(normalized)
}

fn cov(data: &Tensor, device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mean = data.mean(0)?;
    let centered = data.broadcast_sub(&mean)?;
    let (m, n) = data.shape().dims2()?;
    let cov = centered
        .transpose(D::Minus1, D::Minus2)?
        .matmul(&centered)?
        .broadcast_div(&Tensor::new(m as f64, device)?)?;
    Ok(cov)
}

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

    let rand_tensor = Tensor::rand(0f32, 1., (20, 1), &device)?;
    println!("Tensor {rand_tensor}");

    let index_tensor = Tensor::new(&[0u32, 3u32], &device)?;
    println!("Index tensor {index_tensor}");

    let selected = rand_tensor.index_select(&index_tensor, 0)?;
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

#[allow(dead_code)]
fn cdist_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let data1 = vec![0.9041, 0.0196, -0.3108, -2.4423, -0.4821, 1.059];
    let tensor1 = Tensor::from_slice(&data1, (3, 2), &device)?;

    let data2 = vec![-2.1763, -0.4713, -0.6986, 1.3702];
    let tensor2 = Tensor::from_slice(&data2, (2, 2), &device)?;

    let tensor = cdist(&tensor1, &tensor2)?;
    println!("Tensor {tensor}");

    Ok(())
}

#[allow(dead_code)]
fn zscore_normalize_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let data = vec![3., 2., 15., 6., 0., 10., 1., 18.];
    let tensor = Tensor::from_slice(&data, (4, 2), &device)?;
    println!("Tensor {tensor}");
    let tensor = zscore_normalize(&tensor)?;
    println!("Normalized Tensor {tensor}");

    Ok(())
}

#[allow(dead_code)]
fn cov_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let data = vec![0., 2., 1., 1., 2., 0.];
    let tensor = Tensor::from_slice(&data, (3, 2), &device)?;
    let tensor = cov(&tensor, &device)?;
    println!("Cov Tensor {tensor}");

    let data = vec![68., 29., 60., 26., 58., 30., 40., 35.];
    let tensor = Tensor::from_slice(&data, (4, 2), &device)?;
    let tensor = cov(&tensor, &device)?;
    println!("Cov Tensor {tensor}");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cov_test()?;
    Ok(())
}

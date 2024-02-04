use candle_core::{Device, Tensor, DType};
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
    let arr = array![[1., 2., 3.], [4., 5., 6.]];
    let arr_slice = arr.into_shape((2, 3)).to_owned()?.into_raw_vec();
    let device = Device::new_cuda(0)?;
    let tensor = Tensor::from_slice(&arr_slice, (2, 3), &device)?;

    let tensor_ones = Tensor::ones(tensor.shape(), DType::F32, &device)?;
    let slice = tensor_ones.to_vec2::<f32>()?;
    println!("Ones tensor {:?}", slice);

    let tensor_rand = Tensor::rand(0f32, 1., tensor.shape(), &device)?;
    let slice = tensor_rand.to_vec2::<f32>()?;
    println!("Rand tensor {:?}", slice);

    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    example2()?;
    Ok(())
}

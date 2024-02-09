use candle_core::{Device, Tensor};

fn load_image_tensor() -> Result<Tensor, Box<dyn std::error::Error>> {
    let img = image::io::Reader::open("test.png")?.decode()?;
    let (h, w) = (img.height() as usize, img.width() as usize);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let device = Device::new_cuda(0)?;
    let tensor = Tensor::from_vec(data, (h, w, 3), &device)?.permute((2, 0, 1))?;
    Ok(tensor)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tensor = load_image_tensor()?;
    let dim = tensor.dims();
    println!("{:?}\n {tensor}", dim);
    Ok(())
}

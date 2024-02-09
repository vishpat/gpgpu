use candle_core::{Device, Tensor};

fn load_image_tensor(device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    let img = image::io::Reader::open("test.png")?.decode()?;
    let (h, w) = (img.height() as usize, img.width() as usize);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let tensor = Tensor::from_vec(data, (h, w, 3), &device)?.permute((2, 0, 1))?;
    Ok(tensor)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let weights = Tensor::new(&[0.299, 0.587, 0.114], &device)?
        .unsqueeze(1)?
        .unsqueeze(2)?;

    println!("Weight tensor: {weights}");

    let tensor = load_image_tensor(&device)?;
    let dim = tensor.dims();
    println!("{:?}\n Image tensor: {tensor}", dim);

    let gray_tensor = tensor.permute((1, 2, 0))?.matmul(&weights)?;
    println!("Gray tensor: {gray_tensor}");

    Ok(())
}

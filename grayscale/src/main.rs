use candle_core::{Device, Tensor};

fn load_image_tensor(
    device: &Device,
) -> Result<(Tensor, usize, usize), Box<dyn std::error::Error>> {
    let img = image::io::Reader::open("test.png")?.decode()?;
    let (h, w) = (img.height() as usize, img.width() as usize);
    let img = img.to_rgb8();
    let data = img.into_raw().iter().map(|x| (*x as f32 * 1.0)).collect();
    let tensor = Tensor::from_vec(data, (w * h, 3), device)?;
    Ok((tensor, h, w))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available(0)?;
    let weights = Tensor::new(&[0.299f32, 0.587f32, 0.114f32], &device)?.unsqueeze(1)?;
    let (img_tensor, h, w) = load_image_tensor(&device)?;
    let grayscale_tensor = img_tensor.broadcast_matmul(&weights)?.flatten(0, 1)?;
    let img_data: Vec<u8> = grayscale_tensor
        .to_vec1::<f32>()?
        .iter()
        .map(|x| (*x as u8))
        .collect();
    let img = image::GrayImage::from_vec(w as u32, h as u32, img_data).unwrap();
    img.save("grayscale.png")?;
    Ok(())
}

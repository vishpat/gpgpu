use candle_core::{Device, Tensor};

fn load_image_tensor(device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    let img = image::io::Reader::open("test.png")?.decode()?;
    let (h, w) = (img.height() as usize, img.width() as usize);
    let img = img.to_rgb8();
    let data = img.into_raw().iter().map(|x| (*x as f32 * 1.0) as f32).collect();
    let tensor = Tensor::from_vec(data, (h, w, 3), &device)?.permute((2, 0, 1))?;
    Ok(tensor)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let weights = Tensor::new(&[0.299f32, 0.587f32, 0.114f32], &device)?
        .unsqueeze(1)?;

    println!("Weight tensor: {weights}");

    let img_tensor = load_image_tensor(&device)?.permute((1, 2, 0))?;
    println!("Image tensor: {img_tensor}");

    let grayscale = img_tensor.broadcast_matmul(&weights)?; 
    println!("Grayscale tensor: {grayscale}");

    Ok(())
}

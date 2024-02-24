use anyhow::Result;
use candle_core::{Device, Tensor, D};

// Implement Linear Regression model using Gradient Descent
// https://www.youtube.com/watch?v=UVCFaaEBnTE
struct LinearRegression {
    thetas: Tensor,
    device: Device,
}

impl LinearRegression {
    fn new(&self, feature_cnt: usize, gpu: bool) -> Result<Self> {
        let device = if gpu {
            Device::cuda_if_available(0).unwrap()
        } else {
            Device::Cpu
        };
        let thetas: Vec<f32> = vec![0.0; feature_cnt];
        let thetas = Tensor::from_vec(thetas, (feature_cnt,), &device)?; 
        Ok(Self { thetas, device }) 
    }

    fn predict(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.matmul(&self.thetas.unsqueeze(1)?)?.squeeze(1)?)
    }

    fn train(&mut self, x: &Tensor, y: &Tensor, learning_rate: f32, iterations: i32) -> Result<()> {
        let m = y.shape().dims1()?;
        for _ in 0..iterations {
            let predictions = self.predict(x)?;
            let deltas = predictions.sub(y)?;
            let gradient = x
                .t()?
                .matmul(&deltas.unsqueeze(D::Minus1)?)?
                .broadcast_div(&Tensor::new(m as f32, &self.device)?)?;
            let gradient = gradient.squeeze(D::Minus1)?.squeeze(D::Minus1)?;
            self.thetas = self
                .thetas
                .sub(&gradient.broadcast_mul(&Tensor::new(learning_rate, &self.device)?)?)?;
        }
        Ok(())
    }
}

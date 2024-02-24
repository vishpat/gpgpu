use anyhow::Result;
use candle_core::{Device, Tensor, D};
use std::rc::Rc;

pub struct Dataset {
    pub training_data: Tensor,
    pub training_labels: Tensor,
    pub test_data: Tensor,
    pub test_labels: Tensor,
    pub feature_cnt: usize,
}

// Implement Linear Regression model using Gradient Descent
// https://www.youtube.com/watch?v=UVCFaaEBnTE
pub struct LinearRegression {
    thetas: Tensor,
    device: Rc<Device>,
}

impl LinearRegression {
    pub fn new(feature_cnt: usize, device: Rc<Device>) -> Result<Self> {
        let thetas: Vec<f32> = vec![0.0; feature_cnt];
        let thetas = Tensor::from_vec(thetas, (feature_cnt,), &device)?;
        Ok(Self { thetas, device })
    }

    pub fn predict(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.matmul(&self.thetas.unsqueeze(1)?)?.squeeze(1)?)
    }

    pub fn cost(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let m = y.shape().dims1()?;
        let predictions = self.predict(x)?;
        let deltas = predictions.sub(y)?;
        let cost = deltas
            .mul(&deltas)?
            .mean(D::Minus1)?
            .div(&Tensor::new(2.0 * m as f32, &self.device)?)?;
        Ok(cost)
    }

    pub fn train(&mut self, x: &Tensor, y: &Tensor, learning_rate: f32) -> Result<()> {
        let m = y.shape().dims1()?;
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
        Ok(())
    }
}

pub fn r2_score(predictions: &Tensor, labels: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
    let mean = labels.mean(D::Minus1)?;

    let rss = labels.sub(predictions)?;
    let rss = rss.mul(&rss)?.sum(D::Minus1)?;

    let sst = labels.broadcast_sub(&mean)?;
    let sst = sst.mul(&sst)?.sum(D::Minus1)?;

    let tmp = rss.div(&sst)?.to_scalar::<f32>()?;

    Ok(1.0 - tmp)
}

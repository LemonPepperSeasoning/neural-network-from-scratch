use crate::layer::Layer;
use crate::tensor::{RcTensor, Tensor};

pub struct Model {
    layers: Vec<Layer>,
}

impl Model {
    pub fn new(shape: Vec<usize>) -> Self {
        println!("model#init");
        let layers = shape
            .windows(2)
            .map(|window| Layer::new(window[0], window[1]))
            .collect();
        Model { layers }
    }

    pub fn parameters(&self) -> Vec<RcTensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    pub fn feed_foward(&self, input: Vec<RcTensor>) -> Vec<RcTensor> {
        println!("model#feed_foward");
        self.layers
            .iter()
            .fold(input, |x, layer| layer.feed_foward(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameters() {
        let model_a = Model::new(vec![3, 4, 4, 1]);
        let params: Vec<RcTensor> = model_a.parameters();

        // 3*4 + 4*4 + 4*1 = 12 + 16 + 4 = 32
        assert_eq!(params.len(), 32);
    }

    #[test]
    fn test_feed_forward() {
        let a: RcTensor = RcTensor::new(Tensor::new(-3f32));
        let b: RcTensor = RcTensor::new(Tensor::new(2f32));
        let c: RcTensor = RcTensor::new(Tensor::new(0f32));
        let x: Vec<RcTensor> = vec![
            RcTensor::clone(&a),
            RcTensor::clone(&b),
            RcTensor::clone(&c),
        ];

        let model_a = Model::new(vec![3, 4, 4, 1]);
        let output: Vec<RcTensor> = model_a.feed_foward(x);

        assert_eq!(output.len(), 1);
    }
}

use crate::neuron::Neuron;
use crate::tensor::{RcTensor, Tensor};
use std::vec::Vec;

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        println!("layer#init ({}, {})", nin, nout);
        let neurons: Vec<Neuron> = (0..nout).map(|_| Neuron::new(nin)).collect();
        Layer { neurons }
    }

    pub fn feed_foward(&self, input: Vec<RcTensor>) -> Vec<RcTensor> {
        println!("layer#feed_foward");
        self.neurons
            .iter()
            .map(|neuron| <Neuron as Clone>::clone(&neuron).feed_foward(&input))
            .collect()
    }

    pub fn parameters(&self) -> Vec<RcTensor> {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameters() {
        let layer_a = Layer::new(3, 4);
        let params: Vec<RcTensor> = layer_a.parameters();

        assert_eq!(params.len(), 12);
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

        let layer_a = Layer::new(3, 4);
        let output: Vec<RcTensor> = layer_a.feed_foward(x);

        assert_eq!(output.len(), 4);
    }
}

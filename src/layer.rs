use crate::neuron::Neuron;
use crate::scalar::RcScalar;
use std::vec::Vec;

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        //println!("layer#init ({}, {})", nin, nout);
        let neurons: Vec<Neuron> = (0..nout).map(|_| Neuron::new(nin)).collect();
        Layer { neurons }
    }

    pub fn feed_foward(&self, input: Vec<RcScalar>) -> Vec<RcScalar> {
        //println!("layer#feed_foward");
        self.neurons
            .iter()
            .map(|neuron| <Neuron as Clone>::clone(neuron).feed_foward(&input))
            .collect()
    }

    pub fn parameters(&self) -> Vec<RcScalar> {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::Scalar;

    #[test]
    fn test_parameters() {
        let layer_a = Layer::new(3, 4);
        let params: Vec<RcScalar> = layer_a.parameters();

        assert_eq!(params.len(), 16);
    }

    #[test]
    fn test_feed_forward() {
        let a: RcScalar = RcScalar::new(Scalar::new(-3f32));
        let b: RcScalar = RcScalar::new(Scalar::new(2f32));
        let c: RcScalar = RcScalar::new(Scalar::new(0f32));
        let x: Vec<RcScalar> = vec![
            RcScalar::clone(&a),
            RcScalar::clone(&b),
            RcScalar::clone(&c),
        ];

        let layer_a = Layer::new(3, 4);
        let output: Vec<RcScalar> = layer_a.feed_foward(x);

        assert_eq!(output.len(), 4);
    }
}

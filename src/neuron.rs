use rand::Rng;
use std::fmt;
use std::vec::Vec;
use crate::tensor::{RcTensor, Tensor};


#[derive(Debug, Clone)]
pub struct Neuron {
    pub w: Vec<RcTensor>,
    pub b: RcTensor,
}


impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Neuron(w: [")?;
        for (i, rc_tensor) in self.w.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", rc_tensor)?;
        }
        write!(f, "], b: {})", self.b)
    }
}


impl Neuron {
    pub fn new(nin: usize, nonlin: bool) -> Self {
        let mut rng = rand::thread_rng();
        let w: Vec<RcTensor> = (0..nin)
            .map(|_| RcTensor::new(Tensor::new(rng.gen_range(-1.0..1.0))))
            .collect();
        let b = RcTensor::new(Tensor::new(0.0));
        Self {w, b}
    }

    pub fn parameters(&self) -> Vec<RcTensor> {
        self.w.iter().map(|x| RcTensor::clone(x) + RcTensor::clone(&self.b)).collect()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameters() {
        let neuron_a = Neuron::new(3, true);
        let params = neuron_a.parameters();

        assert_eq!(params.len(), 3);
        assert_eq!(params[0].0.borrow().grad, 0.0);
        assert_eq!(params[1].0.borrow().grad, 0.0);
        assert_eq!(params[2].0.borrow().grad, 0.0);

        println!("{}", params[0].0.borrow().prev[0]);
        // assert_eq!(params[0].0.prev, Vec::new());
    }
}

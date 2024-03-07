use rand::Rng;
use std::fmt;
use std::vec::Vec;
use crate::tensor::Tensor;


pub struct Neuron {
    pub w: Vec<Tensor>,
    pub b: Tensor,
}


impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Neuron(w: [")?;
        for (i, tensor) in self.w.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", tensor)?;
        }
        write!(f, "], b: {})", self.b)
    }
}


impl Neuron {
    pub fn new(nin: usize, nonlin: bool) -> Self {
        let mut rng = rand::thread_rng();
        let w: Vec<Tensor> = (0..nin)
            .map(|_| Tensor::new(rng.gen_range(-1.0..1.0)))
            .collect();
        let b = Tensor::new(0.0);
        Self {w, b}
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.w.iter().map(|x| x + &self.b).collect()
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parameters() { 
        let neuron_a = Neuron::new(2, true);
        let params = neuron_a.parameters();
        assert_eq!(params.len(), 2);
    }
}

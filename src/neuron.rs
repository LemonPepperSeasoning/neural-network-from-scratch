use rand::Rng;
use std::fmt;
use std::vec::Vec;
use crate::tensor::Tensor;


#[derive(Debug, Clone)]
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
            .map(|_| Tensor{ 
                data: rng.gen_range(-1.0..1.0), 
                grad: 0.0, 
                prev: Vec::new()
            })
            .collect();
        let b = Tensor{
            data: 0.0, 
            grad: 0.0, 
            prev: Vec::new()
        };
        Self {w, b}
    }
}

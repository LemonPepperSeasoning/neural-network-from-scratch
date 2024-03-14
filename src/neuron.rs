use crate::scalar::{RcScalar, Scalar};
use rand::Rng;
use std::fmt;
use std::iter::zip;
use std::vec::Vec;


#[derive(Debug, Clone)]
pub struct Neuron {
    pub w: Vec<RcScalar>,
    pub b: RcScalar,
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Neuron(w: [")?;
        for (i, rc_scalar) in self.w.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", rc_scalar)?;
        }
        write!(f, "], b: {})", self.b)
    }
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng();
        let w: Vec<RcScalar> = (0..nin)
            .map(|_| RcScalar::new(Scalar::new(rng.gen_range(-1.0..1.0))))
            .collect();
        let b = RcScalar::new(Scalar::new(0.0));
        Self { w, b }
    }

    pub fn feed_foward(self, scalars: &Vec<RcScalar>) -> RcScalar {
        assert_eq!(self.w.len(), scalars.len());
        zip(self.w, scalars)
            .map(|(a, b)| a * b.clone())
            .fold(self.b, |acc, x| acc + x)
            .tanh()
    }

    pub fn parameters(&self) -> Vec<RcScalar> {
        let mut new_vec = self.w.clone(); // Clone the existing vector or use clone_from_slice if possible
        new_vec.push(RcScalar::clone(&self.b));
        new_vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameters() {
        let neuron_a = Neuron::new(3);
        let params = neuron_a.parameters();

        assert_eq!(params.len(), 4);
        assert_eq!(params[0].0.borrow().grad, 0.0);
        assert_eq!(params[1].0.borrow().grad, 0.0);
        assert_eq!(params[2].0.borrow().grad, 0.0);
        assert_eq!(params[3].0.borrow().data, 0.0);
        assert_eq!(params[3].0.borrow().grad, 0.0);
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

        let neuron_a = Neuron::new(3);
        let output: RcScalar = neuron_a.feed_foward(&x);

        println!("{}", output);
    }
}

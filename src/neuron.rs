use crate::scalar::Scalar;
use rand::Rng;
use std::fmt;
use std::iter::zip;
use std::vec::Vec;

#[derive(Debug, Clone)]
pub struct Neuron<'a> {
    pub w: Vec<&'a Scalar<'a>>,
    pub b: &'a Scalar<'a>,
}

impl fmt::Display for Neuron<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Neuron(w: [")?;
        for (i, scalar) in self.w.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", scalar)?;
        }
        write!(f, "], b: {})", self.b)
    }
}

impl Neuron<'_> {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng();
        let w: Vec<&Scalar> = (0..nin)
            .map(|_| &Scalar::new(rng.gen_range(-1.0..1.0)))
            .collect();
        let ref b = Scalar::new(0.0);
        Self { w, b }
    }

    pub fn feed_foward<'a>(&'a self, scalars: &'a Vec<Scalar>) -> Scalar<'a> {
        assert_eq!(self.w.len(), scalars.len());
        zip(self.w.iter(), scalars)
            .map(|(a, b)| *a * b)
            .fold(self.b, |acc, x| &(acc + &x))
            .tanh()
    }

    pub fn parameters(&self) -> Vec<&Scalar> {
        let mut new_vec = self.w.clone(); // Clone the existing vector or use clone_from_slice if possible
        new_vec.push(self.b);
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
        assert_eq!(params[0].grad.get(), 0.0);
        assert_eq!(params[1].grad.get(), 0.0);
        assert_eq!(params[2].grad.get(), 0.0);
        assert_eq!(params[3].data.get(), 0.0);
        assert_eq!(params[3].grad.get(), 0.0);
    }

    #[test]
    fn test_feed_forward() {
        let a = Scalar::new(-3f32);
        let b = Scalar::new(2f32);
        let c = Scalar::new(0f32);
        let x: Vec<Scalar> = vec![
            a,
            b,
            c,
        ];

        let neuron_a = Neuron::new(3);
        let output: Scalar = neuron_a.feed_foward(&x);

        println!("{}", output);
    }
}

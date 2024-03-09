use crate::tensor::Tensor;
use crate::neuron::Neuron;

pub struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        println!("layer#init");
        let neurons: Vec<Neuron> = (0..nout)
            .map(|_| Neuon::new(nin))
            .collect();
        layer { neurons }
    }

    pub fn feed_foward(self, input: Vec<RcTensor>) -> Vec<RcTensor>{
        println!("layer#feed_foward");
        self.neurons.iter().map(|x| x.feed_foward(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let layer_a = Layer::new(3);
        let output: Vec<RcTensor> = layer_a.feed_foward(x);

        assert_eq(output.len(), 3);
    }
}
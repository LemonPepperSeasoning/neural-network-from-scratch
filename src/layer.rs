use crate::tensor::Tensor;
use crate::neuron::Neuron;

pub struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    pub fn new() {
        println!("layer#init");
    }

    pub fn feed_foward(input: Vec<Tensor>) {
        println!("layer#feed_foward");
    }
}
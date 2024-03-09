use crate::tensor::Tensor;
use crate::layer::Layer;

pub struct Model {
    layers: Vec<Layer>,
}

impl Model {
    pub fn new() {
        println!("model#init");
    }

    pub fn feed_foward(input: Vec<Tensor>) {
        println!("model#feed_foward");
    }
}
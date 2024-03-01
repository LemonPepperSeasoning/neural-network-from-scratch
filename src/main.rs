mod tensor;
mod neuron;
use std::vec::Vec;
use crate::neuron::Neuron;
use crate::tensor::Tensor;


fn main() {
    println!("Hello, world!");

    let mut tensor_a = Tensor {
        data: 0.001,
        grad: 0.0,
        prev: Vec::new(),
    };

    let mut tensor_b = Tensor {
        data: 0.002,
        grad: 0.0,
        prev: Vec::new(),
    };
    
    let mut tensor_c = Tensor {
        data: 0.003,
        grad: 0.0,
        prev: Vec::new(),
    };

    tensor_a.backward();

    let tensor_d = tensor_a.clone() + tensor_b.clone();
    println!("{} + {} = {}", tensor_a, tensor_b, tensor_d);

    let tensor_e = tensor_a.clone() + tensor_c.clone();
    println!("{} + {} = {}", tensor_a, tensor_c, tensor_e);

    let tensor_f = tensor_b.clone() + tensor_c.clone();
    println!("{} + {} = {}", tensor_b, tensor_c, tensor_f);

    let tensor_g = tensor_a.clone() * tensor_b.clone();
    println!("{} * {} = {}", tensor_a, tensor_b, tensor_g);

    let tensor_h = tensor_a.clone() * tensor_c.clone();
    println!("{} * {} = {}", tensor_a, tensor_c, tensor_h);

    let tensor_i = tensor_b.clone() * tensor_c.clone();
    println!("{} * {} = {}", tensor_b, tensor_c, tensor_i);



    let neuron_a = Neuron::new(3, true);
    println!("{}", neuron_a);
}

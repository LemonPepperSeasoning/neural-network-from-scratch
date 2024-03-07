mod tensor;
mod neuron;
use crate::neuron::Neuron;
use crate::tensor::Tensor;


fn main() {
    println!("Hello, world!");

    let mut tensor_a = Tensor::new(0.001);    
    let mut tensor_b = Tensor::new(0.002);
    let mut tensor_c = Tensor::new(0.003);

    tensor_a.backward();

    let tensor_d = &tensor_a + &tensor_b;
    println!("{} + {} = {}", tensor_a, tensor_b, tensor_d);

    let tensor_e = &tensor_a + &tensor_c;
    println!("{} + {} = {}", tensor_a, tensor_c, tensor_e);

    let tensor_f = &tensor_b + &tensor_c;
    println!("{} + {} = {}", tensor_b, tensor_c, tensor_f);

    let tensor_g = &tensor_a * &tensor_b;
    println!("{} * {} = {}", tensor_a, tensor_b, tensor_g);

    let tensor_h = &tensor_a * &tensor_c;
    println!("{} * {} = {}", tensor_a, tensor_c, tensor_h);

    let tensor_i = &tensor_b * &tensor_c;
    println!("{} * {} = {}", tensor_b, tensor_c, tensor_i);



    let neuron_a = Neuron::new(3, true);
    println!("{}", neuron_a);
}

use std::ops;
use std::fmt;
use std::vec::Vec;

#[derive(Debug, Clone)]
struct Tensor {
    data: f32,
    grad: f32,
    prev: Vec<Tensor>,
}


impl Tensor {
    fn backward(&self) {
        println!("Tensor#backward() on {}", self);
    }
}


impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor(data={})", self.data)
    }
}


impl ops::Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Self) -> Self::Output {
        println!("Tensor#add() on ({}, {})", self, other);
        
        Tensor {
            data: self.data + other.data,
            grad: 0.0,
            prev: vec![self, other],
        }
    }
}


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
}

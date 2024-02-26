use std::ops;
use std::fmt;
use std::vec::Vec;


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

    fn add(self, other: Tensor) -> Tensor {
        println!("Tensor#add() on (Tensor(data={}), Tensor(data={}))", self.data, other.data);
        
        Tensor {
            data: self.data + other.data,
            grad: 0.0,
            prev: vec![self, other],
        }
    }
}


fn main() {
    println!("Hello, world!");

    let tensor_a = Tensor {
        data: 0.001,
        grad: 0.0,
        prev: Vec::new(),
    };

    let tensor_b = Tensor {
        data: 0.003,
        grad: 0.0,
        prev: Vec::new(),
    };
    

    tensor_a.backward();

    let tensor_c = tensor_a + tensor_b;

    println!("{}", tensor_c);
}

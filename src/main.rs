use std::ops;
use std::fmt;
use std::vec::Vec;
use std::rc::Rc;

#[derive(Debug, Clone)]
struct Tensor {
    data: f32,
    grad: f32,
    prev: Vec<Tensor>,
}

struct RcTensorWrapper(Rc<Tensor>);


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


impl ops::Add for RcTensorWrapper {
    type Output = Tensor;

    fn add(self, other: Self) -> Self::Output {
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
    
    let rc_tensor_a = Rc::new(tensor_a);
    let rc_tensor_b = Rc::new(tensor_b);



//    tensor_a.backward();
    

    let tensor_c = Rc::clone(&rc_tensor_a) + Rc::clone(&rc_tensor_b);
    println!("{}", tensor_c);

//    let tensor_d = tensor_a + tensor_b;
//    println!("{}", tensor_d);
}


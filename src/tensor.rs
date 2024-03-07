use std::ops;
use std::fmt;
use std::vec::Vec;

pub struct Tensor {
    pub data: f32,
    pub grad: f32,
    pub prev: Vec<Tensor>,
    pub _backward: Box<dyn FnMut()>,
}

fn default_backward_fn() {
    println!("running default_backward()");
}


impl Tensor {
    pub fn new(data: f32) -> Self {
        let _backward = Box::new(|| {});
        Self {data, grad: 0.0, prev: Vec::new(), _backward}
    }

    pub fn backward(&self) {
        println!("Tensor#backward() on {}", self);
        (self._backward)();
    }
}


impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor(data={})", self.data)
    }
}


impl ops::Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: Self) -> Self::Output {
        println!("Tensor#add() on ({}, {})", self, other);
        let out = Tensor {
            data: self.data + other.data,
            grad: 0.0,
            prev: vec![*self, *other],
            _backward: Box::new(|| {}),
        };

/*        fn _backward() {
            println!("running _backwards()");
            self.grad += out.grad;
            other.grad += out.grad;
        }
        out._backward = _backward;
*/

        let _backward = move || {
            println!("running _backward()");
            self.grad += out.grad;
            other.grad += out.grad;
        };

        // Assign the closure to _backward field
        out._backward = Box::new(_backward);
        out
    }
}


impl ops::Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: Self) -> Self::Output {
        println!("Tensor#mul() on ({}, {})", self, other);
        Tensor {
            data: self.data * other.data,
            grad: 0.0,
            prev: vec![*self, *other],
            _backward: Box::new(|| {}),
        }
    }
}


impl ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        println!("Tensor#neg() on {}", self);
        Tensor {
            data: -self.data,
            grad: 0.0,
            prev: vec![*self],
            _backward: Box::new(|| {}),
        }
    }
}


impl ops::Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: Self) -> Self::Output {
        println!("Tensor#sub() on ({}, {})", self, other);
        self + &(-other)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_a() -> Tensor {
        Tensor::new(0.001)
    }

    fn tensor_b() -> Tensor {
        Tensor::new(0.002) 
    }
    
    fn tensor_c() -> Tensor {
        Tensor::new(0.003)
    }

    #[test]
    fn test_add() {
        let tensor_1 = &tensor_a() + &tensor_b(); 
        let tensor_2 = &tensor_a() + &tensor_c();
        let tensor_3 = &tensor_b() + &tensor_c();
        
        assert_eq!(tensor_1.data, 0.003);
        assert_eq!(tensor_2.data, 0.004);
        assert_eq!(tensor_3.data, 0.005);

        //tensor_1._backward();
        assert_eq!(tensor_1.grad, 0.0);
    }


    #[test]
    fn test_mul() {
        let tensor_1 = &tensor_a() * &tensor_b(); 
        let tensor_2 = &tensor_a() * &tensor_c();
        let tensor_3 = &tensor_b() * &tensor_c();
        
        assert_eq!(tensor_1.data, 0.0000020000002);
        assert_eq!(tensor_2.data, 0.000003);
        assert_eq!(tensor_3.data, 0.000006);
    }


    #[test]
    fn test_neg() {
        let tensor_1 = -&tensor_a(); 
        let tensor_2 = -&tensor_b();
        let tensor_3 = -&tensor_c();
        
        assert_eq!(tensor_1.data, -0.001);
        assert_eq!(tensor_2.data, -0.002);
        assert_eq!(tensor_3.data, -0.003);
    }

    #[test]
    fn test_sub() {
        let tensor_1 = &tensor_a() - &tensor_b(); 
        let tensor_2 = &tensor_b() - &tensor_a();
        let tensor_3 = &tensor_c() - &tensor_a();
        
        assert_eq!(tensor_1.data, -0.001);
        assert_eq!(tensor_2.data, 0.001);
        assert_eq!(tensor_3.data, 0.0019999999);
    }   
}


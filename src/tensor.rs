use std::ops;
use std::fmt;
use std::vec::Vec;


#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: f32,
    pub grad: f32,
    pub prev: Vec<Tensor>,
}


impl Tensor {
    pub fn new(data: f32, grad: f32) -> Self {
        Tensor {
            data: data,
            grad: grad,
            prev: Vec::new() 
        }
    }

    pub fn backward(&self) {
        println!("Tensor#backward() on {}", self);
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
        Tensor {
            data: self.data + other.data,
            grad: 0.0,
            prev: vec![self.clone(), other.clone()],
        }
    }
}


impl ops::Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: Self) -> Self::Output {
        println!("Tensor#mul() on ({}, {})", self, other);
        Tensor {
            data: self.data * other.data,
            grad: 0.0,
            prev: vec![self.clone(), other.clone()],
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
            prev: vec![self.clone()],
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
        Tensor::new(0.001, 0.0)
    }

    fn tensor_b() -> Tensor {
        Tensor::new(0.002, 0.0) 
    }
    
    fn tensor_c() -> Tensor {
        Tensor::new(0.003, 0.0)
    }

    #[test]
    fn test_add() {
        let tensor_1 = &tensor_a() + &tensor_b(); 
        let tensor_2 = &tensor_a() + &tensor_c();
        let tensor_3 = &tensor_b() + &tensor_c();
        
        assert_eq!(tensor_1.data, 0.003);
        assert_eq!(tensor_2.data, 0.004);
        assert_eq!(tensor_3.data, 0.005);
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


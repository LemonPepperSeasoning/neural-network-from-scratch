use std::cell::RefCell;
use std::fmt;
use std::ops;
use std::rc::Rc;
use std::vec::Vec;

#[derive(Debug, Clone, PartialEq)]
enum Ops {
    ADD,
    MUL,
    NEG,
    NULL,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub data: f32,
    pub grad: f32,
    pub prev: Vec<RcTensor>,
    pub ops: Ops,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RcTensor(pub Rc<RefCell<Tensor>>);

impl RcTensor {
    pub fn new(tensor: Tensor) -> Self {
        RcTensor(Rc::new(RefCell::new(tensor)))
    }

    pub fn clone(self: &RcTensor) -> Self {
        RcTensor(Rc::clone(&self.0))
    }
}

impl Tensor {
    pub fn new(data: f32) -> Self {
        Tensor {
            data: data,
            grad: 0f32,
            prev: Vec::new(),
            ops: Ops::NULL,
        }
    }

    pub fn backwards(&mut self) {
        println!("Tensor#backward() on {}", self);
        // Sort in topological order
        // Iterate over list & for eaach, run backward()
    }

    pub fn backward(&mut self) {
        match self.ops {
            Ops::ADD => {
                for tensor in self.prev.iter() {
                    tensor.0.borrow_mut().grad += self.grad
                }
            }
            Ops::MUL => {
                assert_eq!(self.prev.len(), 2);
                let mut tensor_1 = self.prev[0].0.borrow_mut();
                let mut tensor_2 = self.prev[1].0.borrow_mut();
                tensor_1.grad += (self.grad * tensor_2.data);
                tensor_2.grad += (self.grad * tensor_1.data);
            }
            Ops::NEG => println!("Should not happen"),
            Ops::NULL => println!("backwards do nothing"),
            _ => println!("Should not happen"),
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor(data={})", self.data)
    }
}

impl fmt::Display for RcTensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RcTensor(data={})", self.0.borrow().data)
    }
}

impl ops::Add for RcTensor {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        println!("Tensor#add() on ({}, {})", self, other);
        RcTensor(Rc::new(RefCell::new(Tensor {
            data: self.0.borrow().data + other.0.borrow().data,
            grad: 0.0,
            prev: vec![RcTensor::clone(&self), RcTensor::clone(&other)],
            ops: Ops::ADD,
        })))
    }
}

impl ops::Mul for RcTensor {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        println!("Tensor#mul() on ({}, {})", self, other);
        RcTensor(Rc::new(RefCell::new(Tensor {
            data: self.0.borrow().data * other.0.borrow().data,
            grad: 0f32,
            prev: vec![RcTensor::clone(&self), RcTensor::clone(&other)],
            ops: Ops::MUL,
        })))
    }
}

impl ops::Neg for RcTensor {
    type Output = Self;

    fn neg(self) -> Self::Output {
        println!("Tensor#neg() on {}", self);

        RcTensor(Rc::new(RefCell::new(Tensor {
            data: -self.0.borrow().data,
            grad: 0f32,
            prev: vec![RcTensor::clone(&self)],
            ops: Ops::NEG,
        })))
    }
}

impl ops::Sub for RcTensor {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        println!("Tensor#sub() on ({}, {})", self, other);
        self + (-other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let tensor_a: RcTensor = RcTensor::new(Tensor::new(0.001));
        let tensor_b: RcTensor = RcTensor::new(Tensor::new(0.002));

        let a_add_b: RcTensor = RcTensor::clone(&tensor_a) + RcTensor::clone(&tensor_b);

        assert_eq!(a_add_b.0.borrow().data, 0.003);
        assert_eq!(a_add_b.0.borrow().grad, 0.0);
        assert_eq!(
            a_add_b.0.borrow().prev,
            vec![
                RcTensor::new(Tensor::new(0.001)),
                RcTensor::new(Tensor::new(0.002)),
            ]
        );
        assert_eq!(a_add_b.0.borrow().ops, Ops::ADD);

        tensor_a.0.borrow_mut().backward();
        assert_eq!(tensor_a.0.borrow().grad, 0.0);

        a_add_b.0.borrow_mut().grad = 1.0;
        a_add_b.0.borrow_mut().backward();

        assert_eq!(a_add_b.0.borrow().grad, 1.0);
        assert_eq!(tensor_a.0.borrow().grad, 1.0);
        assert_eq!(tensor_b.0.borrow().grad, 1.0);
    }

    #[test]
    fn test_mul() {
        let tensor_a: RcTensor = RcTensor::new(Tensor::new(0.001));
        let tensor_b: RcTensor = RcTensor::new(Tensor::new(0.002));
        let a_mul_b: RcTensor = RcTensor::clone(&tensor_a) * RcTensor::clone(&tensor_b);

        assert_eq!(a_mul_b.0.borrow().data, 0.0000020000002);
        assert_eq!(a_mul_b.0.borrow().grad, 0.0);
        assert_eq!(
            a_mul_b.0.borrow().prev,
            vec![
                RcTensor::new(Tensor::new(0.001)),
                RcTensor::new(Tensor::new(0.002)),
            ]
        );
        assert_eq!(a_mul_b.0.borrow().ops, Ops::MUL);

        tensor_a.0.borrow_mut().backward();
        assert_eq!(tensor_a.0.borrow().grad, 0.0);

        a_mul_b.0.borrow_mut().grad = 2.0;
        a_mul_b.0.borrow_mut().backward();
        assert_eq!(a_mul_b.0.borrow().grad, 2.0);
        assert_eq!(tensor_a.0.borrow().grad, 0.004);
        assert_eq!(tensor_b.0.borrow().grad, 0.002);
    }

    #[test]
    fn test_neg() {
        let tensor_a: RcTensor = RcTensor::new(Tensor::new(0.001));
        let neg_a: RcTensor = -RcTensor::clone(&tensor_a);

        assert_eq!(neg_a.0.borrow().data, -0.001);
        assert_eq!(neg_a.0.borrow().grad, 0.0);
        assert_eq!(
            neg_a.0.borrow().prev,
            vec![RcTensor::new(Tensor::new(0.001)),]
        );
        assert_eq!(neg_a.0.borrow().ops, Ops::NEG);
    }

    #[test]
    fn test_sub() {
        let tensor_a: RcTensor = RcTensor::new(Tensor::new(0.001));
        let tensor_b: RcTensor = RcTensor::new(Tensor::new(0.002));
        let a_sub_b: RcTensor = RcTensor::clone(&tensor_a) - RcTensor::clone(&tensor_b);

        assert_eq!(a_sub_b.0.borrow().data, -0.001);
        assert_eq!(a_sub_b.0.borrow().grad, 0.0);
        assert_eq!(
            a_sub_b.0.borrow().prev,
            vec![
                RcTensor::new(Tensor::new(0.001)),
                RcTensor::new(Tensor {
                    data: -0.002,
                    grad: 0.0,
                    prev: vec![RcTensor::new(Tensor::new(0.002))],
                    ops: Ops::NEG
                }),
            ]
        );
        assert_eq!(a_sub_b.0.borrow().ops, Ops::ADD);
    }
}

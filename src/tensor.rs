use std::ops;
use std::fmt;
use std::vec::Vec;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub data: f32,
    pub grad: f32,
    pub prev: Vec<RcTensor>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RcTensor(Rc<Tensor>);

impl RcTensor {
    pub fn new(tensor: Tensor) -> Self {
        RcTensor(Rc::new(tensor))
    }

    pub fn clone(self: &RcTensor) -> Self {
        RcTensor(Rc::clone(&self.0))
    }
}

impl Tensor {
    pub fn new(data: f32) -> Tensor {
        Tensor {
            data: data,
            grad: 0f32,
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

impl fmt::Display for RcTensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RcTensor(data={})", self.0.data)
    }
}


impl ops::Add for RcTensor {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        println!("Tensor#add() on ({}, {})", self, other);
        RcTensor(Rc::new(Tensor {
            data: self.0.data + other.0.data,
            grad: 0.0,
            prev: vec![RcTensor::clone(&self), RcTensor::clone(&other)],
        }))
    }
}


impl ops::Mul for RcTensor {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        println!("Tensor#mul() on ({}, {})", self, other);
        RcTensor(Rc::new(Tensor {
            data: self.0.data * other.0.data,
            grad: 0f32,
            prev: vec![RcTensor::clone(&self), RcTensor::clone(&other)],
        }))
    }
}


impl ops::Neg for RcTensor {
    type Output = Self;

    fn neg(self) -> Self::Output {
        println!("Tensor#neg() on {}", self);

        RcTensor(Rc::new(Tensor {
            data: -self.0.data,
            grad: 0f32,
            prev: vec![RcTensor::clone(&self)],
        }))
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
        let tensor_c: RcTensor = RcTensor::new(Tensor::new(0.003));

        let a_add_b: RcTensor = RcTensor::clone(&tensor_a) + RcTensor::clone(&tensor_b);
        let b_add_c: RcTensor = RcTensor::clone(&tensor_b) + RcTensor::clone(&tensor_c);
        let a_add_c: RcTensor = RcTensor::clone(&tensor_a) + RcTensor::clone(&tensor_c);

        assert_eq!(a_add_b.0.data, 0.003);
        assert_eq!(b_add_c.0.data, 0.005);
        assert_eq!(a_add_c.0.data, 0.004);

        assert_eq!(a_add_b.0.grad, 0.0);
        assert_eq!(b_add_c.0.grad, 0.0);
        assert_eq!(a_add_c.0.grad, 0.0);

        assert_eq!(a_add_b.0.prev, vec![
            RcTensor::new((Tensor::new(0.001))),
            RcTensor::new((Tensor::new(0.002))),
        ]);
        assert_eq!(b_add_c.0.prev, vec![
            RcTensor::new((Tensor::new(0.002))),
            RcTensor::new((Tensor::new(0.003))),
        ]);
        assert_eq!(a_add_c.0.prev, vec![
            RcTensor::new((Tensor::new(0.001))),
            RcTensor::new((Tensor::new(0.003))),
        ]);
    }


    #[test]
    fn test_mul() {
        let tensor_a: RcTensor = RcTensor::new(Tensor::new(0.001));
        let tensor_b: RcTensor = RcTensor::new(Tensor::new(0.002));
        let tensor_c: RcTensor = RcTensor::new(Tensor::new(0.003));

        let a_mul_b: RcTensor = RcTensor::clone(&tensor_a) * RcTensor::clone(&tensor_b);
        let b_mul_c: RcTensor = RcTensor::clone(&tensor_b) * RcTensor::clone(&tensor_c);
        let a_mul_c: RcTensor = RcTensor::clone(&tensor_a) * RcTensor::clone(&tensor_c);

        assert_eq!(a_mul_b.0.data, 0.0000020000002);
        assert_eq!(b_mul_c.0.data, 0.000006);
        assert_eq!(a_mul_c.0.data, 0.000003);

        assert_eq!(a_mul_b.0.grad, 0.0);
        assert_eq!(b_mul_c.0.grad, 0.0);
        assert_eq!(a_mul_c.0.grad, 0.0);

        assert_eq!(a_mul_b.0.prev, vec![
            RcTensor::new((Tensor::new(0.001))),
            RcTensor::new((Tensor::new(0.002))),
        ]);
        assert_eq!(b_mul_c.0.prev, vec![
            RcTensor::new((Tensor::new(0.002))),
            RcTensor::new((Tensor::new(0.003))),
        ]);
        assert_eq!(a_mul_c.0.prev, vec![
            RcTensor::new((Tensor::new(0.001))),
            RcTensor::new((Tensor::new(0.003))),
        ]);
    }


    #[test]
    fn test_neg() {
        let tensor_a: RcTensor = RcTensor::new(Tensor::new(0.001));
        let tensor_b: RcTensor = RcTensor::new(Tensor::new(0.002));
        let tensor_c: RcTensor = RcTensor::new(Tensor::new(0.003));

        let neg_a: RcTensor = - RcTensor::clone(&tensor_a);
        let neg_b: RcTensor = - RcTensor::clone(&tensor_b);
        let neg_c: RcTensor = - RcTensor::clone(&tensor_c);

        assert_eq!(neg_a.0.data, -0.001);
        assert_eq!(neg_b.0.data, -0.002);
        assert_eq!(neg_c.0.data, -0.003);

        assert_eq!(neg_a.0.grad, 0.0);
        assert_eq!(neg_b.0.grad, 0.0);
        assert_eq!(neg_c.0.grad, 0.0);

        assert_eq!(neg_a.0.prev, vec![
            RcTensor::new((Tensor::new(0.001))),
        ]);
        assert_eq!(neg_b.0.prev, vec![
            RcTensor::new((Tensor::new(0.002))),
        ]);
        assert_eq!(neg_c.0.prev, vec![
            RcTensor::new((Tensor::new(0.003))),
        ]);
    }


    #[test]
    fn test_sub() {
        let tensor_a: RcTensor = RcTensor::new(Tensor::new(0.001));
        let tensor_b: RcTensor = RcTensor::new(Tensor::new(0.002));
        let tensor_c: RcTensor = RcTensor::new(Tensor::new(0.003));

        let a_sub_b: RcTensor = RcTensor::clone(&tensor_a) - RcTensor::clone(&tensor_b);
        let c_sub_b: RcTensor = RcTensor::clone(&tensor_c) - RcTensor::clone(&tensor_b);
        let c_sub_a: RcTensor = RcTensor::clone(&tensor_c) - RcTensor::clone(&tensor_a);

        assert_eq!(a_sub_b.0.data, -0.001);
        assert_eq!(c_sub_b.0.data, 0.0009999999);
        assert_eq!(c_sub_a.0.data, 0.0019999999);

        assert_eq!(a_sub_b.0.grad, 0.0);
        assert_eq!(c_sub_b.0.grad, 0.0);
        assert_eq!(c_sub_a.0.grad, 0.0);

        assert_eq!(a_sub_b.0.prev, vec![
            RcTensor::new((Tensor::new(0.001))),
            RcTensor::new((Tensor { data: -0.002, grad: 0.0, prev: vec![RcTensor::new(Tensor::new(0.002))] })),
        ]);
        assert_eq!(c_sub_b.0.prev, vec![
            RcTensor::new((Tensor::new(0.003))),
            RcTensor::new((Tensor { data: -0.002, grad: 0.0, prev: vec![RcTensor::new(Tensor::new(0.002))] })),
        ]);
        assert_eq!(c_sub_a.0.prev, vec![
            RcTensor::new((Tensor::new(0.003))),
            RcTensor::new((Tensor { data: -0.001, grad: 0.0, prev: vec![RcTensor::new(Tensor::new(0.001))] })),
        ]);
    }

    // #[test]
    // fn test_neg() {
    //     let tensor_a = RcTensor::new(Tensor::new(0.001, 0.0));
    //     let tensor_b = RcTensor::new(Tensor::new(0.002, 0.0));
    //     let tensor_c = RcTensor::new(Tensor::new(0.003, 0.0));

    //     let tensor_1 = -&tensor_a();
    //     let tensor_2 = -&tensor_b();
    //     let tensor_3 = -&tensor_c();

    //     assert_eq!(tensor_1.data, -0.001);
    //     assert_eq!(tensor_2.data, -0.002);
    //     assert_eq!(tensor_3.data, -0.003);
    // }

    // #[test]
    // fn test_sub() {
    //     let tensor_a = RcTensor::new(Tensor::new(0.001, 0.0));
    //     let tensor_b = RcTensor::new(Tensor::new(0.002, 0.0));
    //     let tensor_c = RcTensor::new(Tensor::new(0.003, 0.0));


    //     let tensor_1 = &tensor_a() - &tensor_b();
    //     let tensor_2 = &tensor_b() - &tensor_a();
    //     let tensor_3 = &tensor_c() - &tensor_a();

    //     assert_eq!(tensor_1.data, -0.001);
    //     assert_eq!(tensor_2.data, 0.001);
    //     assert_eq!(tensor_3.data, 0.0019999999);
    // }
}


use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::ops;
use std::rc::Rc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::vec::Vec;

static GLOBAL_COUTER: AtomicU8 = AtomicU8::new(0);

fn get_id() -> usize {
    GLOBAL_COUTER.fetch_add(1, Ordering::SeqCst);
    GLOBAL_COUTER.load(Ordering::SeqCst) as usize
}

#[derive(Debug, Clone, PartialEq)]
pub enum Ops {
    ADD,
    MUL,
    TANH,
    POW2,
    NULL,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub uid: usize,
    pub data: f32,
    pub grad: f32,
    pub prev: Vec<RcTensor>,
    pub ops: Ops,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RcTensor(pub Rc<RefCell<Tensor>>);

impl Eq for RcTensor {}

impl std::hash::Hash for RcTensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.borrow().uid.hash(state);
    }
}

impl RcTensor {
    pub fn new(tensor: Tensor) -> Self {
        RcTensor(Rc::new(RefCell::new(tensor)))
    }

    pub fn clone(&self) -> Self {
        RcTensor(Rc::clone(&self.0))
    }

    pub fn square(&self) -> Self {
        RcTensor(Rc::new(RefCell::new(Tensor {
            uid: get_id(),
            data: self.0.borrow().data.powf(2f32),
            grad: 0.0,
            prev: vec![RcTensor::clone(&self)],
            ops: Ops::POW2,
        })))
    }

    pub fn tanh(&self) -> Self {
        //println!("Tensor#tanh() on ({})", self);
        RcTensor(Rc::new(RefCell::new(Tensor {
            uid: get_id(),
            data: self.0.borrow().data.tanh(),
            grad: 0.0,
            prev: vec![RcTensor::clone(&self)],
            ops: Ops::TANH,
        })))
    }

    pub fn backwards(&self) {
        //println!("Tensor#backward() on {}", self);
        // Sort in topological order
        let mut ordered_list: Vec<RcTensor> = Vec::new();
        let mut visited: HashSet<RcTensor> = HashSet::new();
        let mut to_visit: Vec<RcTensor> = vec![self.clone()];

        while let Some(c_tensor) = to_visit.pop() {
            if !visited.contains(&c_tensor) {
                ordered_list.push(c_tensor.clone());
                visited.insert(c_tensor.clone());
                // Assuming bfs is a function defined elsewhere in your code
                for child in c_tensor.0.borrow().prev.iter() {
                    to_visit.push(child.clone());
                }
            }
        }

        self.0.borrow_mut().grad = 1.0;
        // Iterate over list & for eaach, run backward()
        for rc_tensor in ordered_list {
            rc_tensor.0.borrow_mut().backward();
        }
    }
}

impl Tensor {
    pub fn new(data: f32) -> Self {
        Tensor {
            uid: get_id(),
            data: data,
            grad: 0f32,
            prev: Vec::new(),
            ops: Ops::NULL,
        }
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
                tensor_1.grad += self.grad * tensor_2.data;
                tensor_2.grad += self.grad * tensor_1.data;
                // println!("{}",tensor_1.grad);ca
            }
            Ops::POW2 => {
                assert_eq!(self.prev.len(), 1);
                let mut tensor_1 = self.prev[0].0.borrow_mut();
                tensor_1.grad += 2f32 * self.grad * tensor_1.data;
            }
            Ops::TANH => {
                assert_eq!(self.prev.len(), 1);
                let mut tensor_1 = self.prev[0].0.borrow_mut();
                tensor_1.grad += self.grad * (1f32 - tensor_1.data.tanh().powf(2f32));
            }
            _ => (),
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor(uid={},data={})", self.uid, self.data)
    }
}

impl fmt::Display for RcTensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "RcTensor(uid={}, data={}, grad={})",
            self.0.borrow().uid,
            self.0.borrow().data,
            self.0.borrow().grad
        )
    }
}

impl ops::Add for RcTensor {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        //println!("Tensor#add() on ({}, {})", self, other);
        RcTensor(Rc::new(RefCell::new(Tensor {
            uid: get_id(),
            data: self.0.borrow().data + other.0.borrow().data,
            grad: 0f32,
            prev: vec![RcTensor::clone(&self), RcTensor::clone(&other)],
            ops: Ops::ADD,
        })))
    }
}

impl ops::Mul for RcTensor {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        //println!("Tensor#mul() on ({}, {})", self, other);
        RcTensor(Rc::new(RefCell::new(Tensor {
            uid: get_id(),
            data: self.0.borrow().data * other.0.borrow().data,
            grad: 0f32,
            prev: vec![RcTensor::clone(&self), RcTensor::clone(&other)],
            ops: Ops::MUL,
        })))
    }
}

impl ops::Mul<f32> for RcTensor {
    type Output = Self;

    fn mul(self, other: f32) -> Self::Output {
        let other_rctensor = RcTensor::new(Tensor::new(other));
        RcTensor(Rc::new(RefCell::new(Tensor {
            uid: get_id(),
            data: self.0.borrow().data * other,
            grad: 0f32,
            prev: vec![RcTensor::clone(&self), RcTensor::clone(&other_rctensor)],
            ops: Ops::MUL,
        })))
    }
}

impl ops::Neg for RcTensor {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * -1f32
    }
}

impl ops::Sub for RcTensor {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        //println!("Tensor#sub() on ({}, {})", self, other);
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
        assert_eq!(a_add_b.0.borrow().ops, Ops::ADD);
        assert_eq!(a_add_b.0.borrow().prev.len(), 2);

        assert_eq!(a_add_b.0.borrow().prev[0].0.borrow().data, 0.001);
        assert_eq!(a_add_b.0.borrow().prev[0].0.borrow().grad, 0.0);
        assert_eq!(a_add_b.0.borrow().prev[0].0.borrow().ops, Ops::NULL);
        assert_eq!(a_add_b.0.borrow().prev[0].0.borrow().prev.len(), 0);

        assert_eq!(a_add_b.0.borrow().prev[1].0.borrow().data, 0.002);
        assert_eq!(a_add_b.0.borrow().prev[1].0.borrow().grad, 0.0);
        assert_eq!(a_add_b.0.borrow().prev[1].0.borrow().ops, Ops::NULL);
        assert_eq!(a_add_b.0.borrow().prev[1].0.borrow().prev.len(), 0);

        tensor_a.0.borrow_mut().backward();
        assert_eq!(tensor_a.0.borrow().grad, 0.0);

        a_add_b.backwards();
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
        assert_eq!(a_mul_b.0.borrow().ops, Ops::MUL);
        assert_eq!(a_mul_b.0.borrow().prev.len(), 2);

        assert_eq!(a_mul_b.0.borrow().prev[0].0.borrow().data, 0.001);
        assert_eq!(a_mul_b.0.borrow().prev[0].0.borrow().grad, 0.0);
        assert_eq!(a_mul_b.0.borrow().prev[0].0.borrow().ops, Ops::NULL);
        assert_eq!(a_mul_b.0.borrow().prev[0].0.borrow().prev.len(), 0);

        assert_eq!(a_mul_b.0.borrow().prev[1].0.borrow().data, 0.002);
        assert_eq!(a_mul_b.0.borrow().prev[1].0.borrow().grad, 0.0);
        assert_eq!(a_mul_b.0.borrow().prev[1].0.borrow().ops, Ops::NULL);
        assert_eq!(a_mul_b.0.borrow().prev[1].0.borrow().prev.len(), 0);

        tensor_a.0.borrow_mut().backward();
        assert_eq!(tensor_a.0.borrow().grad, 0.0);

        a_mul_b.backwards();
        assert_eq!(a_mul_b.0.borrow().grad, 1.0);
        assert_eq!(tensor_a.0.borrow().grad, 0.002);
        assert_eq!(tensor_b.0.borrow().grad, 0.001);
    }

    #[test]
    fn test_neg() {
        let tensor_a: RcTensor = RcTensor::new(Tensor::new(0.001));
        let neg_a: RcTensor = -RcTensor::clone(&tensor_a);

        assert_eq!(neg_a.0.borrow().data, -0.001);
        assert_eq!(neg_a.0.borrow().grad, 0.0);
        assert_eq!(neg_a.0.borrow().ops, Ops::MUL);
        assert_eq!(neg_a.0.borrow().prev.len(), 2);

        assert_eq!(neg_a.0.borrow().prev[0].0.borrow().data, 0.001);
        assert_eq!(neg_a.0.borrow().prev[0].0.borrow().grad, 0.0);
        assert_eq!(neg_a.0.borrow().prev[0].0.borrow().ops, Ops::NULL);
        assert_eq!(neg_a.0.borrow().prev[0].0.borrow().prev.len(), 0);

        assert_eq!(neg_a.0.borrow().prev[1].0.borrow().data, -1.0);
        assert_eq!(neg_a.0.borrow().prev[1].0.borrow().grad, 0.0);
        assert_eq!(neg_a.0.borrow().prev[1].0.borrow().ops, Ops::NULL);
        assert_eq!(neg_a.0.borrow().prev[1].0.borrow().prev.len(), 0);
    }

    #[test]
    fn test_sub() {
        let tensor_a: RcTensor = RcTensor::new(Tensor::new(0.001));
        let tensor_b: RcTensor = RcTensor::new(Tensor::new(0.002));
        let a_sub_b: RcTensor = RcTensor::clone(&tensor_a) - RcTensor::clone(&tensor_b);

        assert_eq!(a_sub_b.0.borrow().data, -0.001);
        assert_eq!(a_sub_b.0.borrow().grad, 0.0);
        assert_eq!(a_sub_b.0.borrow().ops, Ops::ADD);

        assert_eq!(a_sub_b.0.borrow().prev.len(), 2);
        assert_eq!(a_sub_b.0.borrow().prev[0].0.borrow().data, 0.001);
        assert_eq!(a_sub_b.0.borrow().prev[0].0.borrow().grad, 0.0);
        assert_eq!(a_sub_b.0.borrow().prev[0].0.borrow().ops, Ops::NULL);
        assert_eq!(a_sub_b.0.borrow().prev[0].0.borrow().prev.len(), 0);

        assert_eq!(a_sub_b.0.borrow().prev[1].0.borrow().data, -0.002);
        assert_eq!(a_sub_b.0.borrow().prev[1].0.borrow().grad, 0.0);
        assert_eq!(a_sub_b.0.borrow().prev[1].0.borrow().ops, Ops::MUL);
        assert_eq!(a_sub_b.0.borrow().prev[1].0.borrow().prev.len(), 2);

        assert_eq!(
            a_sub_b.0.borrow().prev[1].0.borrow().prev[0]
                .0
                .borrow()
                .data,
            0.002
        );
        assert_eq!(
            a_sub_b.0.borrow().prev[1].0.borrow().prev[0]
                .0
                .borrow()
                .grad,
            0.0
        );
        assert_eq!(
            a_sub_b.0.borrow().prev[1].0.borrow().prev[0]
                .0
                .borrow()
                .prev
                .len(),
            0
        );
        assert_eq!(
            a_sub_b.0.borrow().prev[1].0.borrow().prev[1].0.borrow().ops,
            Ops::NULL
        );
        assert_eq!(
            a_sub_b.0.borrow().prev[1].0.borrow().prev[1]
                .0
                .borrow()
                .data,
            -1.0
        );
        assert_eq!(
            a_sub_b.0.borrow().prev[1].0.borrow().prev[1]
                .0
                .borrow()
                .grad,
            0.0
        );
        assert_eq!(
            a_sub_b.0.borrow().prev[1].0.borrow().prev[1]
                .0
                .borrow()
                .prev
                .len(),
            0
        );
        assert_eq!(
            a_sub_b.0.borrow().prev[1].0.borrow().prev[0].0.borrow().ops,
            Ops::NULL
        );
    }

    #[test]
    fn integration() {
        let a: RcTensor = RcTensor::new(Tensor::new(-3f32));
        let b: RcTensor = RcTensor::new(Tensor::new(2f32));
        let c: RcTensor = RcTensor::new(Tensor::new(0f32));
        let d: RcTensor = RcTensor::new(Tensor::new(1f32));
        let e: RcTensor = RcTensor::new(Tensor::new(6.881f32));
        let ab = RcTensor::clone(&a) * RcTensor::clone(&b);
        let cd: RcTensor = RcTensor::clone(&c) * RcTensor::clone(&d);
        let ab_cd: RcTensor = RcTensor::clone(&ab) + RcTensor::clone(&cd);
        let ab_cd_e: RcTensor = RcTensor::clone(&ab_cd) + RcTensor::clone(&e);
        let ab_cd_e_tanh = RcTensor::clone(&ab_cd_e).tanh();

        assert_eq!(ab_cd_e_tanh.0.borrow().data, 0.70691997);

        ab_cd_e_tanh.backwards();
        assert_eq!(a.0.borrow().grad, 1.0005283);
        assert_eq!(b.0.borrow().grad, -1.5007925);
        assert_eq!(c.0.borrow().grad, 0.50026417);
        assert_eq!(d.0.borrow().grad, 0.0);
        assert_eq!(e.0.borrow().grad, 0.50026417);
    }
}

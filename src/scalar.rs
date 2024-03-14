use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::ops;
use std::rc::Rc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::vec::Vec;
use log::debug;


static GLOBAL_COUTER: AtomicU8 = AtomicU8::new(0);

fn get_id() -> usize {
    GLOBAL_COUTER.fetch_add(1, Ordering::SeqCst);
    GLOBAL_COUTER.load(Ordering::SeqCst) as usize
}

#[derive(Debug, Clone, PartialEq)]
pub enum Ops {
    Add,
    Mul,
    Tanh,
    Pow2,
    Null,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Scalar {
    pub uid: usize,
    pub data: f32,
    pub grad: f32,
    pub prev: Vec<RcScalar>,
    pub ops: Ops,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RcScalar(pub Rc<RefCell<Scalar>>);

impl Eq for RcScalar {}

impl std::hash::Hash for RcScalar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.borrow().uid.hash(state);
    }
}

impl RcScalar {
    pub fn new(scalar: Scalar) -> Self {
        RcScalar(Rc::new(RefCell::new(scalar)))
    }

    pub fn clone(&self) -> Self {
        RcScalar(Rc::clone(&self.0))
    }

    pub fn square(&self) -> Self {
        debug!("Scalar#debug() on ({})", self);
        RcScalar(Rc::new(RefCell::new(Scalar {
            uid: get_id(),
            data: self.0.borrow().data.powf(2f32),
            grad: 0.0,
            prev: vec![RcScalar::clone(self)],
            ops: Ops::Pow2,
        })))
    }

    pub fn tanh(&self) -> Self {
        debug!("Scalar#Tanh() on ({})", self);
        RcScalar(Rc::new(RefCell::new(Scalar {
            uid: get_id(),
            data: self.0.borrow().data.tanh(),
            grad: 0.0,
            prev: vec![RcScalar::clone(self)],
            ops: Ops::Tanh,
        })))
    }

    pub fn backwards(&self) {
        debug!("Scalar#backward() on {}", self);
        // Sort in topological order
        let mut ordered_list: Vec<RcScalar> = Vec::new();
        let mut visited: HashSet<RcScalar> = HashSet::new();
        let mut to_visit: Vec<RcScalar> = vec![self.clone()];

        while let Some(c_scalar) = to_visit.pop() {
            if !visited.contains(&c_scalar) {
                ordered_list.push(c_scalar.clone());
                visited.insert(c_scalar.clone());
                // Assuming bfs is a function defined elsewhere in your code
                for child in c_scalar.0.borrow().prev.iter() {
                    to_visit.push(child.clone());
                }
            }
        }

        self.0.borrow_mut().grad = 1.0;
        // Iterate over list & for eaach, run backward()
        for rc_scalar in ordered_list {
            rc_scalar.0.borrow_mut().backward();
        }
    }
}

impl Scalar {
    pub fn new(data: f32) -> Self {
        let new_scalar = Scalar {
            uid: get_id(),
            data,
            grad: 0f32,
            prev: Vec::new(),
            ops: Ops::Null,
        };
        debug!("Scalar#init() on ({})", new_scalar);
        new_scalar
    }

    pub fn backward(&mut self) {
        match self.ops {
            Ops::Add => {
                for scalar in self.prev.iter() {
                    scalar.0.borrow_mut().grad += self.grad
                }
            }
            Ops::Mul => {
                assert_eq!(self.prev.len(), 2);
                let mut scalar_1 = self.prev[0].0.borrow_mut();
                let mut scalar_2 = self.prev[1].0.borrow_mut();
                scalar_1.grad += self.grad * scalar_2.data;
                scalar_2.grad += self.grad * scalar_1.data;
                // println!("{}",scalar_1.grad);ca
            }
            Ops::Pow2 => {
                assert_eq!(self.prev.len(), 1);
                let mut scalar_1 = self.prev[0].0.borrow_mut();
                scalar_1.grad += 2f32 * self.grad * scalar_1.data;
            }
            Ops::Tanh => {
                assert_eq!(self.prev.len(), 1);
                let mut scalar_1 = self.prev[0].0.borrow_mut();
                scalar_1.grad += self.grad * (1f32 - scalar_1.data.tanh().powf(2f32));
            }
            _ => (),
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Scalar(uid={},data={})", self.uid, self.data)
    }
}

impl fmt::Display for RcScalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "RcScalar(uid={}, data={}, grad={})",
            self.0.borrow().uid,
            self.0.borrow().data,
            self.0.borrow().grad
        )
    }
}

impl ops::Add for RcScalar {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        debug!("Scalar#Add() on ({}, {})", self, other);
        RcScalar(Rc::new(RefCell::new(Scalar {
            uid: get_id(),
            data: self.0.borrow().data + other.0.borrow().data,
            grad: 0f32,
            prev: vec![RcScalar::clone(&self), RcScalar::clone(&other)],
            ops: Ops::Add,
        })))
    }
}

impl ops::Mul for RcScalar {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        debug!("Scalar#Mul() on ({}, {})", self, other);
        RcScalar(Rc::new(RefCell::new(Scalar {
            uid: get_id(),
            data: self.0.borrow().data * other.0.borrow().data,
            grad: 0f32,
            prev: vec![RcScalar::clone(&self), RcScalar::clone(&other)],
            ops: Ops::Mul,
        })))
    }
}

impl ops::Mul<f32> for RcScalar {
    type Output = Self;

    fn mul(self, other: f32) -> Self::Output {
        debug!("Scalar#Mul() on ({}, {})", self, other);
        let other_rcscalar = RcScalar::new(Scalar::new(other));
        RcScalar(Rc::new(RefCell::new(Scalar {
            uid: get_id(),
            data: self.0.borrow().data * other,
            grad: 0f32,
            prev: vec![RcScalar::clone(&self), RcScalar::clone(&other_rcscalar)],
            ops: Ops::Mul,
        })))
    }
}

impl ops::Neg for RcScalar {
    type Output = Self;

    fn neg(self) -> Self::Output {
        debug!("Scalar#neg() on ({})", self);
        self * -1f32
    }
}

impl ops::Sub for RcScalar {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        debug!("Scalar#sub() on ({}, {})", self, other);
        self + (-other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_Add() {
        let scalar_a: RcScalar = RcScalar::new(Scalar::new(0.001));
        let scalar_b: RcScalar = RcScalar::new(Scalar::new(0.002));

        let a_Add_b: RcScalar = RcScalar::clone(&scalar_a) + RcScalar::clone(&scalar_b);

        assert_eq!(a_Add_b.0.borrow().data, 0.003);
        assert_eq!(a_Add_b.0.borrow().grad, 0.0);
        assert_eq!(a_Add_b.0.borrow().ops, Ops::Add);
        assert_eq!(a_Add_b.0.borrow().prev.len(), 2);

        assert_eq!(a_Add_b.0.borrow().prev[0].0.borrow().data, 0.001);
        assert_eq!(a_Add_b.0.borrow().prev[0].0.borrow().grad, 0.0);
        assert_eq!(a_Add_b.0.borrow().prev[0].0.borrow().ops, Ops::Null);
        assert_eq!(a_Add_b.0.borrow().prev[0].0.borrow().prev.len(), 0);

        assert_eq!(a_Add_b.0.borrow().prev[1].0.borrow().data, 0.002);
        assert_eq!(a_Add_b.0.borrow().prev[1].0.borrow().grad, 0.0);
        assert_eq!(a_Add_b.0.borrow().prev[1].0.borrow().ops, Ops::Null);
        assert_eq!(a_Add_b.0.borrow().prev[1].0.borrow().prev.len(), 0);

        scalar_a.0.borrow_mut().backward();
        assert_eq!(scalar_a.0.borrow().grad, 0.0);

        a_Add_b.backwards();
        assert_eq!(a_Add_b.0.borrow().grad, 1.0);
        assert_eq!(scalar_a.0.borrow().grad, 1.0);
        assert_eq!(scalar_b.0.borrow().grad, 1.0);
    }

    #[test]
    fn test_Mul() {
        let scalar_a: RcScalar = RcScalar::new(Scalar::new(0.001));
        let scalar_b: RcScalar = RcScalar::new(Scalar::new(0.002));
        let a_Mul_b: RcScalar = RcScalar::clone(&scalar_a) * RcScalar::clone(&scalar_b);

        assert_eq!(a_Mul_b.0.borrow().data, 0.0000020000002);
        assert_eq!(a_Mul_b.0.borrow().grad, 0.0);
        assert_eq!(a_Mul_b.0.borrow().ops, Ops::Mul);
        assert_eq!(a_Mul_b.0.borrow().prev.len(), 2);

        assert_eq!(a_Mul_b.0.borrow().prev[0].0.borrow().data, 0.001);
        assert_eq!(a_Mul_b.0.borrow().prev[0].0.borrow().grad, 0.0);
        assert_eq!(a_Mul_b.0.borrow().prev[0].0.borrow().ops, Ops::Null);
        assert_eq!(a_Mul_b.0.borrow().prev[0].0.borrow().prev.len(), 0);

        assert_eq!(a_Mul_b.0.borrow().prev[1].0.borrow().data, 0.002);
        assert_eq!(a_Mul_b.0.borrow().prev[1].0.borrow().grad, 0.0);
        assert_eq!(a_Mul_b.0.borrow().prev[1].0.borrow().ops, Ops::Null);
        assert_eq!(a_Mul_b.0.borrow().prev[1].0.borrow().prev.len(), 0);

        scalar_a.0.borrow_mut().backward();
        assert_eq!(scalar_a.0.borrow().grad, 0.0);

        a_Mul_b.backwards();
        assert_eq!(a_Mul_b.0.borrow().grad, 1.0);
        assert_eq!(scalar_a.0.borrow().grad, 0.002);
        assert_eq!(scalar_b.0.borrow().grad, 0.001);
    }

    #[test]
    fn test_neg() {
        let scalar_a: RcScalar = RcScalar::new(Scalar::new(0.001));
        let neg_a: RcScalar = -RcScalar::clone(&scalar_a);

        assert_eq!(neg_a.0.borrow().data, -0.001);
        assert_eq!(neg_a.0.borrow().grad, 0.0);
        assert_eq!(neg_a.0.borrow().ops, Ops::Mul);
        assert_eq!(neg_a.0.borrow().prev.len(), 2);

        assert_eq!(neg_a.0.borrow().prev[0].0.borrow().data, 0.001);
        assert_eq!(neg_a.0.borrow().prev[0].0.borrow().grad, 0.0);
        assert_eq!(neg_a.0.borrow().prev[0].0.borrow().ops, Ops::Null);
        assert_eq!(neg_a.0.borrow().prev[0].0.borrow().prev.len(), 0);

        assert_eq!(neg_a.0.borrow().prev[1].0.borrow().data, -1.0);
        assert_eq!(neg_a.0.borrow().prev[1].0.borrow().grad, 0.0);
        assert_eq!(neg_a.0.borrow().prev[1].0.borrow().ops, Ops::Null);
        assert_eq!(neg_a.0.borrow().prev[1].0.borrow().prev.len(), 0);
    }

    #[test]
    fn test_sub() {
        let scalar_a: RcScalar = RcScalar::new(Scalar::new(0.001));
        let scalar_b: RcScalar = RcScalar::new(Scalar::new(0.002));
        let a_sub_b: RcScalar = RcScalar::clone(&scalar_a) - RcScalar::clone(&scalar_b);

        assert_eq!(a_sub_b.0.borrow().data, -0.001);
        assert_eq!(a_sub_b.0.borrow().grad, 0.0);
        assert_eq!(a_sub_b.0.borrow().ops, Ops::Add);

        assert_eq!(a_sub_b.0.borrow().prev.len(), 2);
        assert_eq!(a_sub_b.0.borrow().prev[0].0.borrow().data, 0.001);
        assert_eq!(a_sub_b.0.borrow().prev[0].0.borrow().grad, 0.0);
        assert_eq!(a_sub_b.0.borrow().prev[0].0.borrow().ops, Ops::Null);
        assert_eq!(a_sub_b.0.borrow().prev[0].0.borrow().prev.len(), 0);

        assert_eq!(a_sub_b.0.borrow().prev[1].0.borrow().data, -0.002);
        assert_eq!(a_sub_b.0.borrow().prev[1].0.borrow().grad, 0.0);
        assert_eq!(a_sub_b.0.borrow().prev[1].0.borrow().ops, Ops::Mul);
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
            Ops::Null
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
            Ops::Null
        );
    }

    #[test]
    fn integration() {
        let a: RcScalar = RcScalar::new(Scalar::new(-3f32));
        let b: RcScalar = RcScalar::new(Scalar::new(2f32));
        let c: RcScalar = RcScalar::new(Scalar::new(0f32));
        let d: RcScalar = RcScalar::new(Scalar::new(1f32));
        let e: RcScalar = RcScalar::new(Scalar::new(6.881f32));
        let ab = RcScalar::clone(&a) * RcScalar::clone(&b);
        let cd: RcScalar = RcScalar::clone(&c) * RcScalar::clone(&d);
        let ab_cd: RcScalar = RcScalar::clone(&ab) + RcScalar::clone(&cd);
        let ab_cd_e: RcScalar = RcScalar::clone(&ab_cd) + RcScalar::clone(&e);
        let ab_cd_e_Tanh = RcScalar::clone(&ab_cd_e).Tanh();

        assert_eq!(ab_cd_e_Tanh.0.borrow().data, 0.70691997);

        ab_cd_e_Tanh.backwards();
        assert_eq!(a.0.borrow().grad, 1.0005283);
        assert_eq!(b.0.borrow().grad, -1.5007925);
        assert_eq!(c.0.borrow().grad, 0.50026417);
        assert_eq!(d.0.borrow().grad, 0.0);
        assert_eq!(e.0.borrow().grad, 0.50026417);
    }
}

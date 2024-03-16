use log::debug;
use std::cell::Cell;
use std::collections::HashSet;
use std::fmt;
use std::ops;
use std::sync::atomic::{AtomicU8, Ordering};
use std::vec::Vec;

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
pub struct Scalar<'a> {
    pub uid: usize,
    pub data: Cell<f32>,
    pub grad: Cell<f32>,
    pub prev: Vec<&'a Scalar<'a>>,
    pub ops: Ops,
}

impl std::hash::Hash for Scalar<'_>{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.uid.hash(state);
    }
}

impl Eq for Scalar<'_> {}



impl Scalar<'_> {
    pub fn new(data: f32) -> Self {
        let new_scalar = Scalar {
            uid: get_id(),
            data: Cell::new(data),
            grad: Cell::new(0f32),
            prev: Vec::new(),
            ops: Ops::Null,
        };
        debug!("Scalar#init() on ({})", new_scalar);
        new_scalar
    }

    // pub fn clone(&self) -> Self {
    //     Scalar(Rc::clone(&self.0))
    // }

    pub fn square<'a>(&'a self) -> Scalar<'a> {
        debug!("Scalar#debug() on ({})", self);
        Scalar {
            uid: get_id(),
            data: Cell::new(self.data.get().powf(2f32)),
            grad: Cell::new(0f32),
            prev: vec![self],
            ops: Ops::Pow2,
        }
    }

    pub fn tanh<'a>(&'a self) -> Scalar<'a> {
        debug!("Scalar#Tanh() on ({})", self);
        Scalar {
            uid: get_id(),
            data: Cell::new(self.data.get().tanh()),
            grad: Cell::new(0f32),
            prev: vec![self],
            ops: Ops::Tanh,
        }
    }

    pub fn backwards(&self) {
        debug!("Scalar#backward() on {}", self);
        // Sort in topological order
        let mut ordered_list: Vec<&Scalar<'_>> = Vec::new();
        let mut visited: HashSet<&Scalar<'_>> = HashSet::new();
        let mut to_visit: Vec<&Scalar<'_>> = vec![self];

        while let Some(c_scalar) = to_visit.pop() {
            if !visited.contains(&c_scalar) {
                ordered_list.push(&c_scalar);
                visited.insert(&c_scalar);
                // Assuming bfs is a function defined elsewhere in your code
                for child in c_scalar.prev.iter() {
                    to_visit.push(&child);
                }
            }
        }

        self.grad.set(1f32);
        // Iterate over list & for eaach, run backward()
        for rc_scalar in ordered_list {
            rc_scalar.backward();
        }
    }

    pub fn backward(&self) {
        match self.ops {
            Ops::Add => {
                for scalar in self.prev.iter() {
                    scalar.grad.set(scalar.grad.get() + self.grad.get());
                }
            }
            Ops::Mul => {
                assert_eq!(self.prev.len(), 2);
                let scalar_1 = self.prev[0];
                let scalar_2 = self.prev[1];
                scalar_1.grad.set(scalar_1.grad.get() + self.grad.get() * scalar_2.data.get());
                scalar_2.grad.set(scalar_1.grad.get() + self.grad.get() * scalar_1.data.get());
                // println!("{}",scalar_1.grad);ca
            }
            Ops::Pow2 => {
                assert_eq!(self.prev.len(), 1);
                let mut scalar_1 = self.prev[0];
                scalar_1.grad.set(scalar_1.grad.get() + 2f32 * self.grad.get() * scalar_1.data.get());
            }
            Ops::Tanh => {
                assert_eq!(self.prev.len(), 1);
                let mut scalar_1 = self.prev[0];
                scalar_1.grad.set(scalar_1.grad.get() + self.grad.get() * (1f32 - scalar_1.data.get().tanh().powf(2f32)));
            }
            _ => (),
        }
    }
}

impl fmt::Display for Scalar<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Scalar(uid={},data={},grad={})",
            self.uid,
            self.data.get(),
            self.grad.get()
        )
    }
}

impl<'a> ops::Add for &'a Scalar<'a> {
    type Output = Scalar<'a>;

    fn add(self, other: Self) -> Self::Output {
        debug!("Scalar#Add() on ({}, {})", self, other);
        Scalar {
            uid: get_id(),
            data: Cell::new(self.data.get() + other.data.get()),
            grad: Cell::new(0f32),
            prev: vec![self, other],
            ops: Ops::Add,
        }
    }
}

impl<'a> ops::Mul for &'a Scalar<'a> {
    type Output = Scalar<'a>;

    fn mul(self, other: Self) -> Self::Output {
        debug!("Scalar#Mul() on ({}, {})", self, other);
        Scalar {
            uid: get_id(),
            data: Cell::new(self.data.get() * other.data.get()),
            grad: Cell::new(0f32),
            prev: vec![self, other],
            ops: Ops::Mul,
        }
    }
}

// impl<'a> ops::Mul<f32> for &'a Scalar<'a> {
//     type Output = Scalar<'static>;

//     fn mul(self, other: f32) -> Self::Output {
//         debug!("Scalar#Mul() on ({}, {})", self, other);
//         let other_scalar: Scalar<'a> = Scalar::new(other);
//         Scalar {
//             uid: get_id(),
//             data: Cell::new(self.data.get() * other),
//             grad: Cell::new(0f32),
//             prev: vec![self, &other_scalar],
//             ops: Ops::Mul,
//         }
//     }
// }

// impl<'a> ops::Neg for &'a Scalar<'a> {
//     type Output = Scalar<'a>;

//     fn neg(self) -> Self::Output {
//         debug!("Scalar#neg() on ({})", self);
//         self * -1f32
//     }
// }

impl<'a> ops::Sub for &'a Scalar<'a> {
    type Output = Scalar<'a>;

    fn sub(self, other: Self) -> Self::Output {
        debug!("Scalar#sub() on ({}, {})", self, other);
        other.data.set(other.data.get() * -1f32);
        self + other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_Add() {
        let scalar_a = Scalar::new(0.001);
        let scalar_b = Scalar::new(0.002);

        let a_add_b = &scalar_a + &scalar_b;

        assert_eq!(a_add_b.data.get(), 0.003);
        assert_eq!(a_add_b.grad.get(), 0.0);
        assert_eq!(a_add_b.ops, Ops::Add);
        assert_eq!(a_add_b.prev.len(), 2);

        assert_eq!(a_add_b.prev[0].data.get(), 0.001);
        assert_eq!(a_add_b.prev[0].grad.get(), 0.0);
        assert_eq!(a_add_b.prev[0].ops, Ops::Null);
        assert_eq!(a_add_b.prev[0].prev.len(), 0);

        assert_eq!(a_add_b.prev[1].data.get(), 0.002);
        assert_eq!(a_add_b.prev[1].grad.get(), 0.0);
        assert_eq!(a_add_b.prev[1].ops, Ops::Null);
        assert_eq!(a_add_b.prev[1].prev.len(), 0);

        scalar_a.backward();
        assert_eq!(scalar_a.grad.get(), 0.0);

        a_add_b.backwards();
        assert_eq!(a_add_b.grad.get(), 1.0);
        assert_eq!(scalar_a.grad.get(), 1.0);
        assert_eq!(scalar_b.grad.get(), 1.0);
    }

    #[test]
    fn test_Mul() {
        let scalar_a = Scalar::new(0.001);
        let scalar_b = Scalar::new(0.002);
        let a_mul_b = &scalar_a * &scalar_b;

        assert_eq!(a_mul_b.data.get(), 0.0000020000002);
        assert_eq!(a_mul_b.grad.get(), 0.0);
        assert_eq!(a_mul_b.ops, Ops::Mul);
        assert_eq!(a_mul_b.prev.len(), 2);

        assert_eq!(a_mul_b.prev[0].data.get(), 0.001);
        assert_eq!(a_mul_b.prev[0].grad.get(), 0.0);
        assert_eq!(a_mul_b.prev[0].ops, Ops::Null);
        assert_eq!(a_mul_b.prev[0].prev.len(), 0);

        assert_eq!(a_mul_b.prev[1].data.get(), 0.002);
        assert_eq!(a_mul_b.prev[1].grad.get(), 0.0);
        assert_eq!(a_mul_b.prev[1].ops, Ops::Null);
        assert_eq!(a_mul_b.prev[1].prev.len(), 0);

        scalar_a.backward();
        assert_eq!(scalar_a.grad.get(), 0.0);

        a_mul_b.backwards();
        assert_eq!(a_mul_b.grad.get(), 1.0);
        assert_eq!(scalar_a.grad.get(), 0.002);
        assert_eq!(scalar_b.grad.get(), 0.001);
    }

    // #[test]
    // fn test_neg() {
    //     let scalar_a = Scalar::new(0.001);
    //     let neg_a = -&scalar_a;

    //     assert_eq!(neg_a.data.get(), -0.001);
    //     assert_eq!(neg_a.grad.get(), 0.0);
    //     assert_eq!(neg_a.ops, Ops::Mul);
    //     assert_eq!(neg_a.prev.len(), 2);

    //     assert_eq!(neg_a.prev[0].data.get(), 0.001);
    //     assert_eq!(neg_a.prev[0].grad.get(), 0.0);
    //     assert_eq!(neg_a.prev[0].ops, Ops::Null);
    //     assert_eq!(neg_a.prev[0].prev.len(), 0);

    //     assert_eq!(neg_a.prev[1].data.get(), -1.0);
    //     assert_eq!(neg_a.prev[1].grad.get(), 0.0);
    //     assert_eq!(neg_a.prev[1].ops, Ops::Null);
    //     assert_eq!(neg_a.prev[1].prev.len(), 0);
    // }

    #[test]
    fn test_sub() {
        let scalar_a = Scalar::new(0.001);
        let scalar_b = Scalar::new(0.002);
        let a_sub_b = &scalar_a - &scalar_b;

        assert_eq!(a_sub_b.data.get(), -0.001);
        assert_eq!(a_sub_b.grad.get(), 0.0);
        assert_eq!(a_sub_b.ops, Ops::Add);

        assert_eq!(a_sub_b.prev.len(), 2);
        assert_eq!(a_sub_b.prev[0].data.get(), 0.001);
        assert_eq!(a_sub_b.prev[0].grad.get(), 0.0);
        assert_eq!(a_sub_b.prev[0].ops, Ops::Null);
        assert_eq!(a_sub_b.prev[0].prev.len(), 0);

        assert_eq!(a_sub_b.prev[1].data.get(), -0.002);
        assert_eq!(a_sub_b.prev[1].grad.get(), 0.0);
        assert_eq!(a_sub_b.prev[1].ops, Ops::Mul);
        assert_eq!(a_sub_b.prev[1].prev.len(), 2);

        assert_eq!(a_sub_b.prev[1].prev[0].data.get(), 0.002);
        assert_eq!(a_sub_b.prev[1].prev[0].grad.get(), 0.0);
        assert_eq!(a_sub_b.prev[1].prev[0].prev.len(), 0);
        assert_eq!(a_sub_b.prev[1].prev[1].ops, Ops::Null);
        assert_eq!(a_sub_b.prev[1].prev[1].data.get(), -1.0);
        assert_eq!(a_sub_b.prev[1].prev[1].grad.get(), 0.0);
        assert_eq!(a_sub_b.prev[1].prev[1].prev.len(), 0);
        assert_eq!(a_sub_b.prev[1].prev[0].ops, Ops::Null);
    }

    #[test]
    fn integration() {
        let a = Scalar::new(-3f32);
        let b = Scalar::new(2f32);
        let c = Scalar::new(0f32);
        let d = Scalar::new(1f32);
        let e = Scalar::new(6.881f32);
        let ab = &a * &b;
        let cd = &c * &d;
        let ab_cd = &ab + &cd;
        let ab_cd_e = &ab_cd + &e;
        let ab_cd_e_tanh = ab_cd_e.tanh();

        assert_eq!(ab_cd_e_tanh.data.get(), 0.70691997);

        ab_cd_e_tanh.backwards();
        assert_eq!(a.grad.get(), 1.0005283);
        assert_eq!(b.grad.get(), -1.5007925);
        assert_eq!(c.grad.get(), 0.50026417);
        assert_eq!(d.grad.get(), 0.0);
        assert_eq!(e.grad.get(), 0.50026417);
    }
}

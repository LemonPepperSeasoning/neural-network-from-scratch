use std::fmt;
use std::ops;
use std::sync::atomic::{AtomicU8, Ordering};
use std::vec::Vec;
use std::cell::Cell;


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
    Square,
    Null
}

#[derive(Debug, Clone, PartialEq)]
pub struct Scalar<'a>  {
    pub uid: usize,
    pub data: Cell<f32>,
    pub grad: Cell<f32>,
    pub prev: Vec<&'a Scalar<'a>>,
    pub ops: Ops,
}


impl Scalar<'_> {
    pub fn new(data: f32) -> Self {
        Scalar {
            uid: get_id(),
            data: Cell::new(data),
            grad: Cell::new(0f32),
            prev: Vec::new(),
            ops: Ops::Null,
        }
    }

    pub fn backwards(&self) {
        self.grad.set(1f32);
        self.backward();
    }

    fn backward(&self) {
        match self.ops {
            Ops::Add => {
                for scalar in self.prev.iter() {
                    scalar.grad.set(scalar.grad.get() + self.grad.get())
                }
            }
            Ops::Mul => {
                assert_eq!(self.prev.len(), 2);
                let scalar_1 = self.prev[0];
                let scalar_2 = self.prev[1];
                scalar_1.grad.set(self.grad.get() + scalar_2.data.get());
                scalar_2.grad.set(self.grad.get() + scalar_1.data.get());
            }
            Ops::Square => {
                assert_eq!(self.prev.len(), 1);
                let scalar_1 = self.prev[0];
                scalar_1.grad.set(2f32 * self.grad.get() * scalar_1.data.get());
            }
            Ops::Tanh => {
                assert_eq!(self.prev.len(), 1);
                let scalar_1 = self.prev[0];
                scalar_1.grad.set(self.grad.get() + (1f32 - scalar_1.data.get().tanh().powf(2f32)));
            }
            _ => (),
        }
    }
}

impl fmt::Display for Scalar<'_>  {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Scalar(uid={},data={},grad={})", self.uid, self.data.get(), self.grad.get())
    }
}

impl<'a> ops::Add for &'a Scalar<'a> {
    type Output = Scalar<'a>;

    fn add(self, other: Self) -> Self::Output {
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
        Scalar {
            uid: get_id(),
            data: Cell::new(self.data.get() * other.data.get()),
            grad: Cell::new(0f32),
            prev: vec![self, other],
            ops: Ops::Mul,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Scalar::new(1f32);
        let b = Scalar::new(2f32);
        let c = &a + &b;
        println!("{}", c);
        c.backwards();
        println!("{}", a);
        println!("{}", b);
        println!("{}", c);
    }

    #[test]
    fn test_mul() {
        let a = Scalar::new(1f32);
        let b = Scalar::new(2f32);
        let c = &a * &b;
        println!("{}", c);
        c.backwards();
        println!("{}", a);
        println!("{}", b);
        println!("{}", c);
    }

    // #[test]
    // fn test_mul_float() {
    //     let a = Scalar::new(1f32);
    //     let c = a * 3f32;
    //     println!("{}", c);
    //     println!("{}", c.prev[0]);
    //     println!("{}", c.prev[1]);
    // }

}
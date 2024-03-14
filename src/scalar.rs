use std::fmt;
use std::fmt::Pointer;
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
    ADD,
    MUL,
    TANH,
    POW2,
    NULL,
}

#[derive(Debug, Clone, PartialEq)]
enum ScalarPointer {
    Pointer(Box<Scalar>),
    Nil,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Scalar {
    pub uid: usize,
    pub data: f32,
    pub grad: f32,
    pub prev1: ScalarPointer,
    pub prev2: ScalarPointer,
    pub ops: Ops,
}


impl Scalar {
    pub fn new(data: f32) -> Self {
        Scalar {
            uid: get_id(),
            data: data,
            grad: 0f32,
            prev1: ScalarPointer::Nil,
            prev2: ScalarPointer::Nil,
            ops: Ops::NULL,
        }
    }

    pub fn get_data(&self) -> f32{
        self.data
    }

    pub fn get_grad(&self) -> f32{
        self.grad
    }

    pub fn add_grad(&mut self, grad:f32) {
        self.grad += grad;
    }

    pub fn backwards(&mut self) {
        self.grad = 1f32;
        self.backward();
    }

    fn backward(&mut self) {
        match self.ops {
            Ops::ADD => {
                match self.prev1 {
                    ScalarPointer::Pointer(ref mut scalar) => {
                        // scalar.add_grad(self.grad);
                        scalar.grad += self.grad;
                    },
                    ScalarPointer::Nil => {}
                }
                // for scalar in self.prev.iter() {
                //     // scalar.grad += self.grad
                //     scalar.add_grad(self.grad)
                // }
            }
            Ops::MUL => {
                // assert_eq!(self.prev.len(), 2);
                // let mut scalar_1 = self.prev1;
                // let mut scalar_2 = self.prev2;
                // let mut scalar_1 = self.prev[0];
                // let mut scalar_2 = self.prev[1];
                // Scalar_1.grad += self.grad * Scalar_2.data;
                // Scalar_2.grad += self.grad * Scalar_1.data;
                match self.prev1 {
                    ScalarPointer::Pointer(ref mut scalar) => {
                        scalar.add_grad(self.grad * 2f32);
                    },
                    ScalarPointer::Nil => {}
                }

                // scalar_1.add_grad(self.grad * scalar_2.get_data());

                // println!("{}",Scalar_1.grad);ca
            }
            // Ops::POW2 => {
            //     assert_eq!(self.prev.len(), 1);
            //     let mut Scalar_1 = self.prev[0];
            //     Scalar_1.grad += 2f32 * self.grad * Scalar_1.data;
            // }
            // Ops::TANH => {
            //     assert_eq!(self.prev.len(), 1);
            //     let mut Scalar_1 = self.prev[0];
            //     Scalar_1.grad += self.grad * (1f32 - Scalar_1.data.tanh().powf(2f32));
            // }
            _ => (),
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Scalar(uid={},data={},grad={})", self.uid, self.data, self.grad)
    }
}

impl ops::Add for &Scalar {
    type Output = Scalar ;

    fn add(self, other: &Scalar) -> Self::Output {
        Scalar {
            uid: get_id(),
            data: self.data + other.data,
            grad: 0f32,
            prev1: ScalarPointer::Pointer(Box::new(*self)),
            prev2: ScalarPointer::Pointer(Box::new(*other)),
            ops: Ops::NULL,
        }
    }
}

// impl ops::Add for Scalar<'_> {
//     type Output = Self;

//     fn add(self, other: Self) -> Self::Output {
//         Scalar {
//             uid: get_id(),
//             data: self.data + other.data,
//             grad: 0f32,
//             prev: vec![self, other],
//             ops: Ops::ADD,
//         }
//     }
// }

// impl ops::Mul for Scalar<'_> {
//     type Output = Self;

//     fn mul(self, other: Self) -> Self::Output {
//         Scalar {
//             uid: get_id(),
//             data: self.data * other.data,
//             grad: 0f32,
//             prev: vec![self, other],
//             ops: Ops::MUL,
//         }
//     }
// }

// impl ops::Mul<f32> for Scalar<'_> {
//     type Output = Self;

//     fn mul(self, other: f32) -> Self::Output {
//         let other = Scalar::new(other);
//         Scalar {
//             uid: get_id(),
//             data: self.data + other.data,
//             grad: 0f32,
//             prev: vec![self, other],
//             ops: Ops::MUL,
//         }
//     }
// }




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Scalar::new(1f32);
        let b = Scalar::new(2f32);
        let mut c = a + b;
        println!("{}", c);
        c.backwards();
        println!("{}", c);
    }

    // #[test]
    // fn test_mul() {
    //     let a = Scalar::new(1f32);
    //     let b = Scalar::new(2f32);
    //     let c = a * b;
    //     println!("{}", c);
    //     println!("{}", c.prev[0]);
    //     println!("{}", c.prev[1]);
    // }

    // #[test]
    // fn test_mul_float() {
    //     let a = Scalar::new(1f32);
    //     let c = a * 3f32;
    //     println!("{}", c);
    //     println!("{}", c.prev[0]);
    //     println!("{}", c.prev[1]);
    // }

}
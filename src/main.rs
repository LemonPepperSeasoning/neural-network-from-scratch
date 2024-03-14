mod scalar;
use crate::scalar::Scalar;



fn main() {
    println!("hello world");

    let a = Scalar::new(1f32);
    let b = Scalar::new(2f32);
    let x = Scalar::new(3f32);
    let y = Scalar::new(4f32);
    let mut c = a + b;
    let mut d = a + b;



    println!("{}", c);
    c.backwards();
    println!("{}", c);
}


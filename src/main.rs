mod layer;
mod model;
mod neuron;
mod scalar;
// use crate::model::Model;
use crate::scalar::Scalar;
use log::debug;

fn main() {
    env_logger::init();
    debug!("Starting application...");

    let xs: Vec<Vec<Scalar<'_>>> = vec![
        vec![
            Scalar::new(2f32),
            Scalar::new(3f32),
            Scalar::new(-1f32),
        ],
        vec![
            Scalar::new(3f32),
            Scalar::new(-1f32),
            Scalar::new(0.5f32),
        ],
        vec![
            Scalar::new(0.5f32),
            Scalar::new(1f32),
            Scalar::new(1f32),
        ],
        vec![
            Scalar::new(1f32),
            Scalar::new(1f32),
            Scalar::new(-1f32),
        ],
    ];

    let ys: Vec<Vec<Scalar<'_>>> = vec![
        vec![Scalar::new(1f32)],
        vec![Scalar::new(-1f32)],
        vec![Scalar::new(-1f32)],
        vec![Scalar::new(1f32)],
    ];

    // let model_a = Model::new(vec![3, 4, 4, 1]);

    // for i in 0..100 {
    //     let y_preds: Vec<Vec<Scalar<'_>>> = xs
    //         .iter()
    //         .map(|x: &Vec<Scalar>| model_a.feed_foward(x.clone()))
    //         .collect();

    //     let mut loss: Scalar = Scalar::new(0f32);
    //     for (row1, row2) in ys.iter().zip(y_preds.iter()) {
    //         for (elem1, elem2) in row1.iter().zip(row2.iter()) {
    //             loss = &loss + &(elem1 - elem2).square();
    //         }
    //     }

    //     for p in model_a.parameters() {
    //         p.0.borrow_mut().grad = 0f32;
    //     }

    //     loss.backwards();

    //     for p in model_a.parameters() {
    //         p.data.set(0.01 * p.grad.get());
    //     }

    //     println!("Iter: {}, loss: {}", i, loss.data.get());
    // }
}

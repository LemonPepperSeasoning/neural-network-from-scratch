mod layer;
mod model;
mod neuron;
mod tensor;
use crate::model::Model;
use crate::tensor::{RcTensor, Tensor};

fn main() {
    let xs: Vec<Vec<RcTensor>> = vec![
        vec![
            RcTensor::clone(&RcTensor::new(Tensor::new(2f32))),
            RcTensor::clone(&RcTensor::new(Tensor::new(3f32))),
            RcTensor::clone(&RcTensor::new(Tensor::new(-1f32))),
        ],
        vec![
            RcTensor::clone(&RcTensor::new(Tensor::new(3f32))),
            RcTensor::clone(&RcTensor::new(Tensor::new(-1f32))),
            RcTensor::clone(&RcTensor::new(Tensor::new(0.5f32))),
        ],
        vec![
            RcTensor::clone(&RcTensor::new(Tensor::new(0.5f32))),
            RcTensor::clone(&RcTensor::new(Tensor::new(1f32))),
            RcTensor::clone(&RcTensor::new(Tensor::new(1f32))),
        ],
        vec![
            RcTensor::clone(&RcTensor::new(Tensor::new(1f32))),
            RcTensor::clone(&RcTensor::new(Tensor::new(1f32))),
            RcTensor::clone(&RcTensor::new(Tensor::new(-1f32))),
        ],
    ];

    let ys: Vec<Vec<RcTensor>> = vec![
        vec![RcTensor::new(Tensor::new(1f32))],
        vec![RcTensor::new(Tensor::new(-1f32))],
        vec![RcTensor::new(Tensor::new(-1f32))],
        vec![RcTensor::new(Tensor::new(1f32))],
    ];

    let model_a = Model::new(vec![3, 4, 4, 1]);

    for i in 0..100 {
        let y_preds: Vec<Vec<RcTensor>> = xs
            .iter()
            .map(|x: &Vec<RcTensor>| model_a.feed_foward(x.clone()))
            .collect();

        let mut loss: RcTensor = RcTensor::new(Tensor::new(0f32));
        for (row1, row2) in ys.iter().zip(y_preds.iter()) {
            for (&ref elem1, elem2) in row1.iter().zip(row2.iter()) {
                loss = loss + (RcTensor::clone(elem1) - RcTensor::clone(elem2)).square();
            }
        }

        for p in model_a.parameters() {
            p.0.borrow_mut().grad = 0f32;
        }

        loss.backwards();

        for p in model_a.parameters() {
            let mut borrowed = p.0.borrow_mut();
            borrowed.data -= 0.01 * borrowed.grad;
        }

        println!("Iter: {}, loss: {}", i, loss.0.borrow().data);
    }
}

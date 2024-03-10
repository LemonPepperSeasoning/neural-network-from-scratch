use crate::layer::Layer;
use crate::scalar::RcScalar;

pub struct Model {
    layers: Vec<Layer>,
}

impl Model {
    pub fn new(shape: Vec<usize>) -> Self {
        //println!("model#init");
        let layers = shape
            .windows(2)
            .map(|window: &[usize]| Layer::new(window[0], window[1]))
            .collect();
        Model { layers }
    }

    pub fn parameters(&self) -> Vec<RcScalar> {
        self.layers
            .iter()
            .flat_map(|layer: &Layer| layer.parameters())
            .collect()
    }

    pub fn feed_foward(&self, input: Vec<RcScalar>) -> Vec<RcScalar> {
        //println!("model#feed_foward");
        self.layers
            .iter()
            .fold(input, |x: Vec<RcScalar>, layer: &Layer| {
                layer.feed_foward(x)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::Scalar;

    #[test]
    fn test_parameters() {
        let model_a = Model::new(vec![3, 4, 4, 1]);
        let params: Vec<RcScalar> = model_a.parameters();

        // (3+1)*4 + (4+1)*4 + (4+1)*1 = 16 + 20 + 5 = 41
        assert_eq!(params.len(), 41);
    }

    #[test]
    fn test_feed_forward() {
        let a: RcScalar = RcScalar::new(Scalar::new(-3f32));
        let b: RcScalar = RcScalar::new(Scalar::new(2f32));
        let c: RcScalar = RcScalar::new(Scalar::new(0f32));
        let x: Vec<RcScalar> = vec![
            RcScalar::clone(&a),
            RcScalar::clone(&b),
            RcScalar::clone(&c),
        ];

        let model_a = Model::new(vec![3, 4, 4, 1]);
        let output: Vec<RcScalar> = model_a.feed_foward(x);

        assert_eq!(output.len(), 1);
    }
}

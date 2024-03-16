use crate::layer::Layer;
use crate::scalar::Scalar;

pub struct Model<'a> {
    layers: Vec<Layer<'a>>,
}

impl Model<'_> {
    pub fn new(shape: Vec<usize>) -> Self {
        //println!("model#init");
        let layers = shape
            .windows(2)
            .map(|window: &[usize]| Layer::new(window[0], window[1]))
            .collect();
        Model { layers }
    }

    pub fn feed_foward(&self, input: Vec<Scalar>) -> Vec<Scalar> {
        //println!("model#feed_foward");
        self.layers
            .iter()
            .fold(input, |x, layer| {
                layer.feed_foward(x)
            })
    }

    pub fn parameters(&self) -> Vec<Scalar> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::Scalar;

    #[test]
    fn test_parameters() {
        let model_a = Model::new(vec![3, 4, 4, 1]);
        let params: Vec<Scalar> = model_a.parameters();

        // (3+1)*4 + (4+1)*4 + (4+1)*1 = 16 + 20 + 5 = 41
        assert_eq!(params.len(), 41);
    }

    #[test]
    fn test_feed_forward() {
        let a = Scalar::new(-3f32);
        let b = Scalar::new(2f32);
        let c = Scalar::new(0f32);
        let x: Vec<Scalar> = vec![
            a,
            b,
            c,
        ];

        let model_a = Model::new(vec![3, 4, 4, 1]);
        let output: Vec<Scalar> = model_a.feed_foward(x);

        assert_eq!(output.len(), 1);
    }
}

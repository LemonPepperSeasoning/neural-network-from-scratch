# neural-network-from-scratch

Neural network from Scratch with Rust.

More specifically, this repository is a implementation of Multi-Layer Perceptron (MLP).

### How to use

Simple create N layer MLP by

```
mod model;
use crate::model::Model;

// this will create mlp with first layer having Input size 3, 2 hidden layers both size of 4 and output size of 1
let model_a = Model::new(vec![3, 4, 4, 1]);
```

### Development

Run with:

```
cargo run

// To see log
RUST_LOG=DEBUG cargo run
```

Run unit test:

```
cargo test
```

Auto format:

```
cargo fmt
```

!note: For release use: `cargo build --release`

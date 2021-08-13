use opencv::core::Scalar;

pub fn red() -> Scalar {
    Scalar::new(0.0, 0.0, 255.0, 1.0)
}

pub fn green() -> Scalar {
    Scalar::new(0.0, 255.0, 0.0, 1.0)
}

pub fn blue() -> Scalar {
    Scalar::new(255.0, 0.0, 0.0, 1.0)
}

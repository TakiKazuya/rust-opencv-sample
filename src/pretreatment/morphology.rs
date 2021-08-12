use opencv::imgproc::{get_structuring_element, morphology_ex, MORPH_RECT, MORPH_CLOSE, MORPH_OPEN};
use opencv::core::{Size, Point, Mat, BORDER_REPLICATE, Scalar};

/// オープニング処理
pub fn opening(image: Mat) -> Mat {
    let kernel = define_kernel();

    // オープニング処理の出力先を定義
    let mut dst_img_open = Mat::default();

    // オープニング処理
    let result_morphology_opening = morphology_ex(&image,
                                                  &mut dst_img_open,
                                                  MORPH_OPEN,
                                                  &kernel,
                                                  Point::default(),
                                                  1, BORDER_REPLICATE,
                                                  Scalar::default());
    if let Err(code) = result_morphology_opening {
        println!("オープニング処理に失敗しました。 Message: {}", code);
        panic!();
    }

    dst_img_open
}

/// クロージング処理
pub fn closing(image: Mat) -> Mat {
    let kernel = define_kernel();

    // クロージング処理の出力先を定義
    let mut dst_img_close = Mat::default();

    // クロージング処理
    let result_morphology_closing = morphology_ex(&image,
                                                  &mut dst_img_close,
                                                  MORPH_CLOSE,
                                                  &kernel,
                                                  Point::default(),
                                                  1, BORDER_REPLICATE,
                                                  Scalar::default());

    if let Err(code) = result_morphology_closing {
        println!("クロージング処理に失敗しました。 Message: {}", code);
        panic!();
    }

    dst_img_close
}

/// カーネルを定義
fn define_kernel() -> Mat {
    get_structuring_element(MORPH_RECT, Size::new(5, 5), Point::default()).unwrap()
}
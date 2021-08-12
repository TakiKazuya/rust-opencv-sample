use opencv::core::{Mat, Vector, Size, Point, BORDER_REPLICATE, Scalar};
use opencv::imgproc::{threshold, get_structuring_element, morphology_ex, THRESH_OTSU, MORPH_RECT, MORPH_CLOSE, MORPH_OPEN};
use opencv::imgcodecs::imwrite;

pub fn run(image: Mat) -> Mat {
    println!("前処理開始");

    // 出力先を用意
    let mut dst_img_threshold = Mat::default();

    // ２値化処理
    let result_threshold = threshold(&image, &mut dst_img_threshold, 0.0, 255.0, THRESH_OTSU);
    if let Err(code) = result_threshold {
        println!("２値化処理に失敗しました。 Message: {}", code);
        panic!();
    }

    // カーネルを定義
    let mut kernel = get_structuring_element(MORPH_RECT, Size::new(5, 5), Point::default()).unwrap();

    // クロージング処理の出力先を定義
    let mut dst_img_close = Mat::default();

    // クロージング処理
    let result_morphology_closing = morphology_ex(&dst_img_threshold,
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

    // オープニング処理の出力先を定義
    let mut dst_img_open = Mat::default();

    // オープニング処理
    let result_morphology_opening = morphology_ex(&dst_img_close,
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

    imwrite("output_pretreatment.jpg", &dst_img_open, &Vector::new());

    dst_img_open
}
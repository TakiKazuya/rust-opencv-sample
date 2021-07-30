use opencv::imgcodecs::{IMREAD_GRAYSCALE, imwrite};
use opencv::imgproc::{threshold,THRESH_OTSU, morphology_ex, MORPH_OPEN, MORPH_CLOSE, MORPH_RECT, morphology_default_border_value};
use opencv::core::{Mat, Vector, Size, Point, BORDER_WRAP, Scalar, BORDER_TRANSPARENT,BORDER_REPLICATE};
use opencv::imgproc::get_structuring_element;
use opencv::imgproc::ThresholdTypes::THRESH_BINARY;

fn main(){
    // 元画像を読み込み
    let path: String = String::from("image.jpg");
    let mut src_img = opencv::imgcodecs::imread(&path, IMREAD_GRAYSCALE).unwrap();

    // 出力先を用意
    let mut dst_img_threshold = Mat::default();

    // ２値化処理
    threshold(&src_img, &mut dst_img_threshold, 0.0, 255.0, THRESH_OTSU);

    // カーネルを定義
    let mut kernel = get_structuring_element(MORPH_RECT, Size::new(5, 5), Point::default()).unwrap();

    // クロージング処理の出力先を定義
    let mut dst_img_close = Mat::default();

    // クロージング処理
    morphology_ex(&dst_img_threshold,
                  &mut dst_img_close,
                  MORPH_CLOSE,
                  &kernel,
                  Point::default(),
                  1, BORDER_REPLICATE,
                  Scalar::default());

    // オープニング処理の出力先を定義
    let mut dst_img_open = Mat::default();

    // オープニング処理
    morphology_ex(&dst_img_close,
                  &mut dst_img_open,
                  MORPH_OPEN,
                  &kernel,
                  Point::default(),
                  1, BORDER_REPLICATE,
                  Scalar::default());

    imwrite("output.jpg", &dst_img_open, &Vector::new());
}

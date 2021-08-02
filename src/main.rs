use opencv::imgcodecs::{IMREAD_GRAYSCALE, imwrite};
use opencv::imgproc::{threshold, THRESH_OTSU, morphology_ex, MORPH_OPEN, MORPH_CLOSE, MORPH_RECT, morphology_default_border_value, RETR_LIST};
use opencv::core::{Mat, Vector, Size, Point, BORDER_WRAP, Scalar, BORDER_TRANSPARENT,BORDER_REPLICATE};
use opencv::imgproc::{get_structuring_element, find_contours};
use opencv::imgproc::ThresholdTypes::THRESH_BINARY;

fn main(){
    // 元画像を読み込み
    let path: String = String::from("image.jpg");

    // 処理元の画像を定義
    let mut src_img;
    let result_read_img = opencv::imgcodecs::imread(&path, IMREAD_GRAYSCALE);
    match result_read_img {
        Ok(img) => src_img = img,
        Err(code) => {
            print!("code: {:?}", code);
            return;
        }
    };

    ////// 前処理ここから //////

    // 出力先を用意
    let mut dst_img_threshold = Mat::default();

    // ２値化処理
    let result_threshold = threshold(&src_img, &mut dst_img_threshold, 0.0, 255.0, THRESH_OTSU);
    if result_threshold.is_err() {
        return;
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
    if result_morphology_closing.is_err() {
        return;
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
    if result_morphology_opening.is_err() {
        return;
    }

    imwrite("output_pretreatment.jpg", &dst_img_open, &Vector::new());

    ////// 前処理ここまで //////

    ////// 輪郭の抽出ここから//////

    imwrite("output.jpg", &dst_img_open, &Vector::new());
}

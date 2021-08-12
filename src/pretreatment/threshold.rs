use opencv::core::Mat;
use opencv::imgproc::{threshold, THRESH_OTSU};

pub fn run(image: Mat) -> Mat {
    // 出力先を定義
    let mut dst_img_threshold = Mat::default();

    // ２値化処理
    let result_threshold = threshold(&image, &mut dst_img_threshold, 0.0, 255.0, THRESH_OTSU);
    if let Err(code) = result_threshold {
        println!("２値化処理に失敗しました。 Message: {}", code);
        panic!();
    }

    dst_img_threshold
}
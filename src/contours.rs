use opencv::core::{Mat, Point, no_array, Vector};
use opencv::types::VectorOfVectorOfPoint;
use opencv::imgproc::{find_contours, RETR_LIST, CHAIN_APPROX_SIMPLE, LINE_8, INTER_MAX, draw_contours};
use opencv::imgcodecs::{IMREAD_COLOR, imwrite};

use crate::{SOURCE_IMAGE_PATH, colors};

pub fn extract(img: Mat) -> VectorOfVectorOfPoint {
    println!("輪郭抽出処理開始");

    // 前処理後の画像
    let mut src_img_pretreatment = img.clone();

    // 抽出した輪郭の出力先を定義
    let mut contours = VectorOfVectorOfPoint::default();

    // 輪郭の抽出
    let result_find_contours = find_contours(&src_img_pretreatment, &mut contours, RETR_LIST, CHAIN_APPROX_SIMPLE, Point::default());
    if let Err(code) = result_find_contours {
        println!("輪郭の抽出に失敗しました。 Message: {}", code);
        panic!();
    }

    // 輪郭を描画した画像の出力先(元画像に輪郭を描画して出力する)
    let mut dst_img_draw_contours;
    let result_read_img = opencv::imgcodecs::imread(SOURCE_IMAGE_PATH, IMREAD_COLOR);
    match result_read_img {
        Ok(img) => dst_img_draw_contours = img,
        Err(code) => {
            print!("code: {:?}", code);
            panic!();
        }
    };

    // 輪郭の描画
    let result_draw_contours = draw_contours(&mut dst_img_draw_contours, &contours, -1, colors::green(), 5, LINE_8, &no_array().unwrap(), INTER_MAX, Point::default());
    if let Err(code) = result_draw_contours {
        println!("輪郭の描画に失敗しました。 Message: {}", code);
        panic!();
    }

    //
    let result_write = imwrite("output_contours.jpg", &dst_img_draw_contours, &Vector::new());
    if let Err(code) = result_write {
        println!("輪郭描画後の出力に失敗しました。 Message: {}", code);
        panic!();
    }

    println!("輪郭抽出処理終了");
    contours
}
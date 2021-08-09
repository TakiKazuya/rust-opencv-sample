use opencv::core::{Mat, Vector, Size, Point, BORDER_WRAP, Scalar, BORDER_TRANSPARENT, BORDER_REPLICATE, CV_8UC3};
use opencv::imgcodecs::{IMREAD_GRAYSCALE, IMREAD_COLOR, imwrite};
use opencv::imgproc::{get_structuring_element, find_contours, threshold, THRESH_OTSU, morphology_ex, MORPH_OPEN, MORPH_CLOSE, MORPH_RECT, morphology_default_border_value, RETR_CCOMP, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, draw_contours, FILLED, INTER_MAX, LINE_8, INTER_NEAREST, RETR_LIST, RETR_TREE};
use opencv::types::{VectorOfVectorOfPoint, VectorOfVec4i};

fn main(){
    // 元画像を読み込み
    println!("画像の読み込みを開始します。");
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
    if let Err(code) = result_threshold {
        println!("２値化処理に失敗しました。 Message: {}", code);
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

    ////// 前処理ここまで //////

    ////// 輪郭の抽出ここから//////

    // 前処理後の画像
    let mut src_img_pretreatment = dst_img_open.clone();

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
    let result_read_img = opencv::imgcodecs::imread(&path, IMREAD_COLOR);
    match result_read_img {
        Ok(img) => dst_img_draw_contours = img,
        Err(code) => {
            print!("code: {:?}", code);
            panic!();
        }
    };
    // 輪郭の階層情報
    let hierarchy = VectorOfVec4i::default();

    // 描画する輪郭の色
    let green = Scalar::new(0.0, 255.0, 0.0, 1.0);

    // 輪郭の描画
    let result_draw_contours = draw_contours(&mut dst_img_draw_contours, &contours, -1, green, 5, LINE_8, &hierarchy, INTER_MAX, Point::new(5, 5));
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

    ////// 輪郭の抽出ここまで //////

    // 全ての処理が終わったあと、画像を出力する
    println!("画像を出力します。");
    imwrite("output.jpg", &src_img_pretreatment, &Vector::new());
}

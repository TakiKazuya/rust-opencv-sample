use opencv::core::{Mat, Vector, Size, Point, Scalar, BORDER_WRAP, BORDER_TRANSPARENT, BORDER_REPLICATE, CV_8UC3};
use opencv::imgcodecs::{IMREAD_GRAYSCALE, IMREAD_COLOR, imwrite};
use opencv::imgproc::{get_structuring_element, find_contours, threshold, morphology_ex, contour_area, draw_contours};
use opencv::imgproc::{THRESH_OTSU, MORPH_OPEN, MORPH_CLOSE, MORPH_RECT, RETR_CCOMP, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, FILLED, INTER_MAX, LINE_8, INTER_NEAREST, RETR_LIST, RETR_TREE};
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
            panic!();
        }
    };

    ////// 前処理ここから //////

    println!("前処理開始");

    // 出力先を用意
    let mut dst_img_threshold = Mat::default();

    // ２値化処理
    let result_threshold = threshold(&src_img, &mut dst_img_threshold, 0.0, 255.0, THRESH_OTSU);
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

    println!("前処理終了");

    ////// 前処理ここまで //////

    ////// 輪郭の抽出ここから//////

    println!("輪郭抽出処理開始");

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

    println!("輪郭抽出処理終了");

    ////// 輪郭の抽出ここまで //////

    ////// 面積が最大になる輪郭を取得 //////

    println!("面積が最大になる輪郭を取得する処理開始");

    // 輪郭の面積を保存するベクタを定義する。 要素の型はf64
    // 抽出した輪郭(contours)から面積を取得し、配列に追加していく。
    let contour_areas: Vec<f64> = contours.iter().map(|contour| {
        contour_area(&contour, false).unwrap_or(0.0)
    }).collect();

    println!("contour_areas: {:?}", contour_areas);

    // 最大値を取得する。
    let max_area = contour_areas.iter().fold(0.0/0.0, |m, v| v.max(m));

    // インデックスを取得
    let index = contour_areas.iter().position(|&area| area == max_area).unwrap();

    // 取得したインデックスから輪郭の情報を取得する。
    let max_contour = contours.iter().nth(index).unwrap();
    println!("面積が最大になる輪郭 -> {:?}", max_contour);

    println!("面積が最大になる輪郭を取得する処理終了");

    ////// 面積が最大になる輪郭を取得ここまで //////

    // 全ての処理が終わったあと、画像を出力する
    println!("画像を出力します。");
    imwrite("output.jpg", &src_img_pretreatment, &Vector::new());
}

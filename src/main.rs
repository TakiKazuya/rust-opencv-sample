use opencv::core::{Mat, Vector, Size, Point, Scalar, BORDER_WRAP, BORDER_TRANSPARENT, BORDER_REPLICATE, CV_8UC3, no_array, VectorExtern, norm, Point2f, DECOMP_LU};
use opencv::imgcodecs::{IMREAD_GRAYSCALE, IMREAD_COLOR, imwrite};
use opencv::imgproc::{get_structuring_element, find_contours, threshold, morphology_ex, contour_area, draw_contours, arc_length, approx_poly_dp, circle, get_perspective_transform, warp_perspective};
use opencv::imgproc::{THRESH_OTSU, MORPH_OPEN, MORPH_CLOSE, MORPH_RECT, RETR_CCOMP, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, INTER_MAX, LINE_8, INTER_NEAREST, RETR_LIST};
use opencv::types::{VectorOfVectorOfPoint, VectorOfPoint, VectorOfPoint2f};
use opencv::core::NormTypes::NORM_L1;
use std::error::Error;

mod pretreatment;

mod colors;

const SOURCE_IMAGE_PATH: &str = "image.jpg";

fn main(){
    // 元画像を読み込み
    println!("画像の読み込みを開始します。");

    // 処理元の画像を定義
    let mut src_img;
    let result_read_img = opencv::imgcodecs::imread(SOURCE_IMAGE_PATH, IMREAD_GRAYSCALE);
    match result_read_img {
        Ok(img) => src_img = img,
        Err(code) => {
            print!("code: {:?}", code);
            panic!();
        }
    };

    let mut output_img;
    let result_read_img = opencv::imgcodecs::imread(SOURCE_IMAGE_PATH, IMREAD_COLOR);
    match result_read_img {
        Ok(img) => output_img = img,
        Err(code) => {
            print!("code: {:?}", code);
            panic!();
        }
    };

    ////// 前処理ここから //////

    let img_pretreatment = pretreatment::run(src_img);

    ////// 前処理ここまで //////

    ////// 輪郭の抽出ここから//////

    println!("輪郭抽出処理開始");

    // 前処理後の画像
    let mut src_img_pretreatment = img_pretreatment.clone();

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
    let max_contour = contours.get(index).unwrap();

    println!("面積が最大になる輪郭 -> {:?}", max_contour);

    println!("面積が最大になる輪郭を取得する処理終了");

    ////// 面積が最大になる輪郭を取得ここまで //////

    ////// 図形の周囲の長さ取得ここから //////

    let result_arc_length = arc_length(&max_contour, true);
    let arc_len;
    match result_arc_length {
        Ok(length) => {
            arc_len = length;
            println!("arc_len: {}", arc_len)
        },
        Err(code) => {
            print!("図形の周囲の長さの取得に失敗しました。 Message: {}", code);
            panic!();
        }
    };

    ////// 図形の周囲の長さ取得ここまで //////

    ////// 図形の頂点抽出ここから //////

    let mut vertex_points = VectorOfPoint::default();
    let result_approx_contour = approx_poly_dp(&max_contour, &mut vertex_points, 0.1 * arc_len, true);
    if let Err(code) = result_approx_contour {
        println!("頂点抽出に失敗しました。 Message: {}", code);
        panic!();
    }

    println!("vertex_points: {:?}", &vertex_points);

    // 頂点を描画した画像の出力先(元画像に頂点を描画して出力する)
    let mut dst_img_draw_vertex;
    let result_read_img = opencv::imgcodecs::imread(SOURCE_IMAGE_PATH, IMREAD_COLOR);
    match result_read_img {
        Ok(img) => dst_img_draw_vertex = img,
        Err(code) => {
            print!("code: {:?}", code);
            panic!();
        }
    };

    // 頂点を一つずつ取り出して描画していく
    for point in vertex_points.iter() {
        circle(&mut dst_img_draw_vertex, point, 3, colors::red(), 5, 0, 0);
    }

    let result_write = imwrite("output_vertex.jpg", &dst_img_draw_vertex, &Vector::new());
    if let Err(code) = result_write {
        println!("頂点描画後の出力に失敗しました。 Message: {}", code);
        panic!();
    }

    ////// 図形の頂点抽出ここまで //////

    ////// 座標を左上、右上、右下、左下に分類する ここから //////

    // 抽出した頂点が４つであること。4つではない場合はエラーを返す
    if vertex_points.len() != 4 {
        panic!("頂点の数は4つである必要があります。抽出した頂点の数：{}", vertex_points.len());
    }

    // Vector<Point>型からVec<Point>型に変換
    let mut vec_vertex_points = vertex_points.to_vec();
    // 座標のxの値でソート
    vec_vertex_points.sort_by(|p, p1| {
        p.x.cmp(&p1.x)
    });

    // 左右で座標を分割
    let mut left = vec_vertex_points.get(0..2).unwrap().to_owned();
    let mut right = vec_vertex_points.get(2..4).unwrap().to_owned();

    // y軸でソートし、0番目を上側、1番目を下側とする
    // 左側
    left.sort_by(|p, p1| {
        p.y.cmp(&p1.y)
    });
    let left_up = left.first().unwrap().to_owned();
    let left_down = left.last().unwrap().to_owned();

    // 右側
    right.sort_by(|p, p1| {
        p.y.cmp(&p1.y)
    });
    let right_up = right.first().unwrap().to_owned();
    let right_down = right.last().unwrap().to_owned();

    println!("left_up: {:?}", left_up);
    println!("left_down: {:?}", left_down);
    println!("right_up: {:?}", right_up);
    println!("right_down: {:?}", right_down);

    ////// 座標を左上、右上、右下、左下に分類する ここまで //////

    ////// 台形補正処理 ここから //////

    let upper_line = ((right_up.x - left_up.x).abs() ^ 2) + ((right_up.y - left_up.y).abs() ^ 2);
    let downer_line = ((right_down.x - left_down.x).abs() ^ 2) + ((right_down.y - left_down.y).abs() ^ 2);
    let left_line = ((left_up.x - left_down.x).abs() ^ 2) + ((left_up.y - left_down.y).abs() ^ 2);
    let right_line = ((right_up.x - right_down.x).abs() ^ 2) + ((right_up.y - right_down.y).abs() ^ 2);

    let max_x = if upper_line > downer_line {
        upper_line
    } else {
        downer_line
    };

    let max_y = if left_line > right_line {
        left_line
    } else {
        right_line
    };

    // 元となる座標をVector<Point2f>に変換する

    let mut vec_vertex_points2f: Vec<Point2f> = vec![];
    for p in vertex_points.iter() {
        vec_vertex_points2f.push(Point2f::new(p.x as f32, p.y as f32));
    };

    let vertex_points= VectorOfPoint2f::from(vec_vertex_points2f);

    let left_up = Point2f::new(0.0,0.0);
    let left_down = Point2f::new(0.0, max_y as f32);
    let right_down = Point2f::new(max_x as f32, max_y as f32);
    let right_up = Point2f::new(max_x as f32, 0.0);

    let coordinate: Vector<Point2f> = Vector::from(vec![left_up, left_down, right_down, right_up]);

    let mut m;
    match get_perspective_transform(&vertex_points, &coordinate, DECOMP_LU) {
        Ok(mat) => {
            println!("{:?}", mat);
            m= mat;
        },
        Err(code) => {
            panic!("Error. Message: {}", code)
        }
    };

    let mut src_img;
    let result_read_img = opencv::imgcodecs::imread(SOURCE_IMAGE_PATH, IMREAD_COLOR);
    match result_read_img {
        Ok(img) => src_img = img,
        Err(code) => {
            print!("code: {:?}", code);
            panic!();
        }
    };

    let mut img_corrected = Mat::default();
    match warp_perspective(&src_img, &mut img_corrected, &m, Default::default(), 0, 0, Default::default()) {
        Ok(_) => {
            println!("success");
        },
        Err(code) => {
            panic!("{}", code);
        }
    }

    ////// 台形補正処理 ここまで //////

    // 全ての処理が終わったあと、画像を出力する
    println!("画像を出力します。");
    imwrite("output.jpg", &img_corrected, &Vector::new());
}

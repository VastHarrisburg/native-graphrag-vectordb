pub fn dot_product(x: &[f32], y: &[f32]) -> Option<f32>{
    let length1 = x.len();
    let length2 = y.len();
    if length1 != length2 {
        return None;
    }
    let mut sum = 0.0;
    for i in 0..length1 {
        sum = sum + (x[i] * y[i]);
    }
    return Some(sum);
}
pub fn norm(x: &[f32]) -> f32{
    if x.len() < 1 {
        return 0.0;
    }
    let mut mag = 0.0;
    for i in 0..x.len() {
        let squared = x[i].powf(2.0);
        mag = mag + squared;
    }
    return mag.sqrt();
}
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32>{
    let dot = match dot_product(a,b){
        None => return None,
        Some(sum) => sum,
    };

    let mag_a = norm(a);
    let mag_b = norm(b);
    if mag_a == 0.0 {
        return None;
    }
    if mag_b == 0.0 {
        return None;
    }
    let similarity = dot/(mag_a*mag_b);
    return Some(similarity);
}
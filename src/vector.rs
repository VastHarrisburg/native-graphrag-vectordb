pub fn dot_product(x: &[f32], y: &[f32]) -> Option<f32> {
    if x.len() != y.len() {
        return None;
    }

    Some(x.iter().zip(y).map(|(left, right)| left * right).sum())
}

pub fn norm(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }

    x.iter().map(|value| value.powi(2)).sum::<f32>().sqrt()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    let dot = dot_product(a, b)?;

    let mag_a = norm(a);
    let mag_b = norm(b);
    if mag_a == 0.0 || mag_b == 0.0 {
        return None;
    }

    Some(dot / (mag_a * mag_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_rejects_invalid_vectors() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), None);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), None);
    }

    #[test]
    fn cosine_similarity_scores_equal_vectors() {
        let score = cosine_similarity(&[1.0, 2.0], &[1.0, 2.0]).unwrap();
        assert!((score - 1.0).abs() < f32::EPSILON);
    }
}

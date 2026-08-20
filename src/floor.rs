//! Secondary occupancy family floor on a folded map.
//!
//! A 2-means split of the low-D points is two communities only when
//! the sides rematch to distinct packing-family labels *and* the
//! centroids do not overlap. Leftover wells of one packing stay one
//! community. This is a descriptor-distance floor, not a hop-graph
//! Fiedler split.

/// Occupancy family floor from a 2-D folding plus packing-family labels.
///
/// Returns 2 when a 2-means split separates two labelled packings.
/// Returns 1 when there are fewer than two points, one family, an
/// empty side, or the two sides overlap in the map.
pub fn occupancy_map_floor(xy: &[[f64; 2]], family: &[usize]) -> usize {
    if xy.len() < 2 || xy.len() != family.len() {
        return 1;
    }
    if xy
        .iter()
        .flat_map(|point| point.iter())
        .any(|value| !value.is_finite())
    {
        return 1;
    }
    if family.iter().all(|&f| f == family[0]) {
        return 1;
    }
    let Some((left, right)) = two_means(xy) else {
        return 1;
    };
    if left.is_empty() || right.is_empty() {
        return 1;
    }
    let Some(left_family) = majority_family(family, &left) else {
        return 1;
    };
    let Some(right_family) = majority_family(family, &right) else {
        return 1;
    };
    if left_family == right_family {
        return 1;
    }
    let (c0, r0) = centroid_radius(xy, &left);
    let (c1, r1) = centroid_radius(xy, &right);
    let gap = ((c0[0] - c1[0]).powi(2) + (c0[1] - c1[1]).powi(2)).sqrt();
    if gap > r0 + r1 { 2 } else { 1 }
}

fn majority_family(family: &[usize], members: &[usize]) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for &i in members {
        let f = *family.get(i)?;
        let count = members
            .iter()
            .filter(|&&j| family.get(j) == Some(&f))
            .count();
        match best {
            Some((_, held)) if count <= held => {}
            _ => best = Some((f, count)),
        }
    }
    best.map(|(f, _)| f)
}

fn centroid_radius(xy: &[[f64; 2]], members: &[usize]) -> ([f64; 2], f64) {
    let n = members.len().max(1) as f64;
    let mut c = [0.0, 0.0];
    for &i in members {
        c[0] += xy[i][0];
        c[1] += xy[i][1];
    }
    c[0] /= n;
    c[1] /= n;
    let radius = members
        .iter()
        .map(|&i| {
            let dx = xy[i][0] - c[0];
            let dy = xy[i][1] - c[1];
            (dx * dx + dy * dy).sqrt()
        })
        .fold(0.0, f64::max);
    (c, radius)
}

fn two_means(xy: &[[f64; 2]]) -> Option<(Vec<usize>, Vec<usize>)> {
    let (a, b) = farthest_pair(xy)?;
    let mut c0 = xy[a];
    let mut c1 = xy[b];
    let mut left = Vec::new();
    let mut right = Vec::new();
    for _ in 0..16 {
        left.clear();
        right.clear();
        for (i, point) in xy.iter().enumerate() {
            if dist2(*point, c0) <= dist2(*point, c1) {
                left.push(i);
            } else {
                right.push(i);
            }
        }
        if left.is_empty() || right.is_empty() {
            return None;
        }
        c0 = mean(xy, &left);
        c1 = mean(xy, &right);
    }
    Some((left, right))
}

fn farthest_pair(xy: &[[f64; 2]]) -> Option<(usize, usize)> {
    let n = xy.len();
    if n < 2 {
        return None;
    }
    let mut best = (0, 1);
    let mut best_d = -1.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist2(xy[i], xy[j]);
            if d > best_d {
                best_d = d;
                best = (i, j);
            }
        }
    }
    Some(best)
}

fn mean(xy: &[[f64; 2]], members: &[usize]) -> [f64; 2] {
    let n = members.len().max(1) as f64;
    let mut c = [0.0, 0.0];
    for &i in members {
        c[0] += xy[i][0];
        c[1] += xy[i][1];
    }
    c[0] /= n;
    c[1] /= n;
    c
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

#[cfg(test)]
mod tests {
    use super::occupancy_map_floor;

    #[test]
    fn one_family_is_one_community() {
        let xy = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [8.0, 8.0]];
        assert_eq!(occupancy_map_floor(&xy, &[0, 0, 0, 0]), 1);
    }

    #[test]
    fn two_separated_families_are_two_communities() {
        let xy = [[0.0, 0.0], [0.1, 0.0], [8.0, 8.0], [8.1, 8.0]];
        assert_eq!(occupancy_map_floor(&xy, &[0, 0, 1, 1]), 2);
    }

    #[test]
    fn overlapping_families_are_one_community() {
        let xy = [[0.0, 0.0], [0.1, 0.0], [0.05, 0.05], [0.08, 0.02]];
        assert_eq!(occupancy_map_floor(&xy, &[0, 0, 1, 1]), 1);
    }

    #[test]
    fn invalid_coordinates_are_one_community() {
        assert_eq!(
            occupancy_map_floor(&[[0.0, 0.0], [f64::NAN, 1.0]], &[0, 1]),
            1
        );
    }
}

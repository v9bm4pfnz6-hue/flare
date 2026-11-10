/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Request-local speculative acceptance and copy proposal metadata.

use crate::{Result, execution::BATCH_CAPACITIES};

/// Copy provenance uses the nonnegative bits of its signed 64-bit mask.
pub const COPY_DRAFT_CAPACITY: usize = i64::BITS as usize - 1;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProposalSource {
    #[default]
    Mtp,
    Copy,
}

impl ProposalSource {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "mtp" => Ok(Self::Mtp),
            "copy" => Ok(Self::Copy),
            _ => Err("speculative_proposal_source must be mtp or copy".into()),
        }
    }
}

/// Host-validated continuation and masks for the corresponding committed prefix.
#[derive(Clone, Debug)]
pub struct CopyProposal {
    pub consumed: u64,
    pub tokens: Vec<i64>,
    pub count: usize,
    pub grammar_masks: Vec<i32>,
}

pub fn graph_catalog(widths: &[usize], fixed: Option<usize>, capacity: usize) -> [Vec<usize>; 7] {
    BATCH_CAPACITIES.map(|batch| if batch <= capacity {
        fixed.map_or_else(|| widths.to_vec(), |width| vec![width])
    } else { Vec::new() })
}

pub fn validate_widths(widths: &[usize]) -> Result<()> {
    if !widths.is_empty()
        && (widths[0] != 1
            || widths.windows(2).any(|w| w[0] >= w[1])
            || widths.iter().any(|w| *w == 0))
    {
        return Err(
            "speculative widths must be sorted unique positive integers including 1".into(),
        );
    }
    Ok(())
}

/// A bounded observation mean whose information ages as the request advances.
#[derive(Clone, Copy, Debug)]
pub(super) struct PrefixEstimate {
    mean: f64,
    weight: f64,
    observed_at: u32,
}
impl PrefixEstimate {
    pub fn new(mean: f64, generated: u32) -> Self {
        Self { mean, weight: 2.0, observed_at: generated }
    }

    pub fn weight_at(&self, generated: u32) -> f64 {
        self.weight * (-f64::from(generated.saturating_sub(self.observed_at)) / 128.0).exp2()
    }

    pub fn mean(&self) -> f64 { self.mean }

    /// Seed the prior mean of a new request from a deployment-wide estimate; the
    /// observation weight and age stay unchanged.
    pub(super) fn set_mean(&mut self, mean: f64) {
        debug_assert!(mean.is_finite() && (0.0..=1.0).contains(&mean));
        self.mean = mean.clamp(0.0, 1.0);
    }

    pub fn upper(&self, generated: u32) -> f64 {
        (self.mean + (20.0f64.ln() / (2.0 * self.weight_at(generated))).sqrt()).min(1.0)
    }

    pub fn observe(&mut self, value: f64, generated: u32) {
        debug_assert!(value.is_finite() && (0.0..=1.0).contains(&value));
        self.weight = (self.weight_at(generated) + 1.0).min(128.0);
        self.mean += (value - self.mean) / self.weight;
        self.observed_at = generated;
    }
}

/// Prefix survival of the complete proposal program, including copy overrides.
#[derive(Clone)]
pub struct AcceptanceProfile {
    prefixes: [Vec<PrefixEstimate>; 2],
    rounds: u64,
}

impl Default for AcceptanceProfile {
    fn default() -> Self {
        Self::new(8)
    }
}

impl AcceptanceProfile {
    pub fn new(token_capacity: usize) -> Self {
        Self { prefixes: std::array::from_fn(|_| (0..token_capacity.saturating_sub(1))
            .map(|i| PrefixEstimate::new(0.5f64.powi(i as i32 + 1), 0)).collect()), rounds: 0 }
    }

    /// Replace the means of one request profile with a deployment-wide estimate.
    /// The per-depth observation weight and age are unchanged.
    pub(super) fn seed_means(&mut self, source: ProposalSource, means: &[f64]) {
        for (estimate, mean) in self.prefixes[source as usize].iter_mut().zip(means) {
            estimate.set_mean(*mean);
        }
    }

    /// Depth zero is the first draft after the pending target token.
    pub fn prefix_probability(&self, source: ProposalSource, depth: usize) -> Option<f64> {
        self.prefixes[source as usize]
            .get(depth)
            .map(|_| self.curve(source, 0, false)[depth])
    }

    pub(super) fn curve(&self, source: ProposalSource, generated: u32, optimistic: bool) -> Vec<f64> {
        let mut bound = 1.0f64;
        self.prefixes[source as usize].iter().map(|p| {
            bound = bound.min(if optimistic { p.upper(generated) } else { p.mean() });
            bound
        }).collect()
    }

    pub fn expected_tokens(&self, source: ProposalSource, width: usize) -> f64 {
        f64::from(width != 0) + self.curve(source, 0, false)[..width.saturating_sub(1).min(self.prefixes[source as usize].len())].iter().sum::<f64>()
    }

    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    pub(super) fn information(&self, source: ProposalSource, depth: usize, generated: u32) -> f64 {
        self.prefixes[source as usize][depth].weight_at(generated)
    }

    /// Rejection falsifies every longer offered prefix; termination censors its suffix.
    pub(super) fn observe(
        &mut self,
        source: ProposalSource,
        generated: u32,
        offered: usize,
        accepted: usize,
        rejected: bool,
        survival: Option<&[f32]>,
    ) {
        debug_assert!(accepted <= offered && offered <= self.prefixes[source as usize].len());
        if offered == 0 {
            return;
        }
        let observed = if rejected { offered } else { accepted };
        for depth in 0..observed {
            let value = survival.map_or(f64::from(depth < accepted), |p| f64::from(p[depth]));
            self.prefixes[source as usize][depth].observe(value, generated);
        }
        self.rounds = self.rounds.saturating_add(1);
    }
}

/// Deployment-wide per-depth prefix survival learned from completed speculative
/// rounds, used to seed the fixed prior of every new request profile. The
/// estimate is bounded: each depth keeps its mean and at most
/// `DEPLOYMENT_SAMPLE_CAP` observations, and censored depths (never offered, or
/// longer than the observed prefix) are never updated.
pub(super) struct DeploymentAcceptance {
    means: [Vec<f64>; 2],
    weights: [Vec<f64>; 2],
}

// Prior pseudo-observations of the fixed 0.5^(depth+1) prior. Two matches
// the per-request PrefixEstimate prior weight, so one round contributes at most
// one third of the estimate and a single rejection leaves the depth-0 mean at
// 1/3, above the 0.28 break-even used by the scheduler simulation.
pub(super) const DEPLOYMENT_PRIOR_WEIGHT: f64 = 2.0;
pub(super) const DEPLOYMENT_SAMPLE_CAP: f64 = 1024.0;

impl DeploymentAcceptance {
    pub(super) fn new(token_capacity: usize) -> Self {
        Self {
            means: std::array::from_fn(|_| (0..token_capacity.saturating_sub(1))
                .map(|i| 0.5f64.powi(i as i32 + 1)).collect()),
            weights: std::array::from_fn(|_| {
                vec![DEPLOYMENT_PRIOR_WEIGHT; token_capacity.saturating_sub(1)]
            }),
        }
    }

    /// Means to seed a new request profile; the fixed prior until any observation.
    pub(super) fn means(&self, source: ProposalSource) -> &[f64] {
        &self.means[source as usize]
    }

    #[cfg(test)]
    pub(super) fn observations(&self, source: ProposalSource) -> &[f64] {
        &self.weights[source as usize]
    }

    /// Record one completed round with the same censoring rule as
    /// `AcceptanceProfile::observe`: only depths `0..observed` are updated, where
    /// `observed = offered` on a rejection and `accepted` otherwise.
    pub(super) fn observe(&mut self, source: ProposalSource, offered: usize,
        accepted: usize, rejected: bool, survival: Option<&[f32]>) {
        if offered == 0 {
            return;
        }
        let index = source as usize;
        let observed = (if rejected { offered } else { accepted })
            .min(offered).min(self.means[index].len());
        for depth in 0..observed {
            let value = survival.map_or(f64::from(depth < accepted), |p| f64::from(p[depth]));
            let weight = (self.weights[index][depth] + 1.0).min(DEPLOYMENT_SAMPLE_CAP);
            self.means[index][depth] += (value - self.means[index][depth]) / weight;
            self.weights[index][depth] = weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_ages_information_without_changing_the_mean() {
        let mut estimate = PrefixEstimate::new(0.5, 0);
        for _ in 0..256 { estimate.observe(0.0, 0); }
        let mean = estimate.mean();
        let upper = estimate.upper(0);
        assert!(upper < 0.2);
        assert_eq!(estimate.mean(), mean);
        assert!(estimate.upper(128) > upper);
        assert_eq!(estimate.upper(u32::MAX), 1.0);
        estimate.observe(1.0, 4096);
        assert!(estimate.mean() > 0.99);
    }

    #[test]
    fn shallow_rounds_do_not_refresh_unobserved_suffixes() {
        let mut profile = AcceptanceProfile::default();
        for _ in 0..256 { profile.observe(ProposalSource::Mtp, 0, 7, 0, true, None); }
        let before = profile.curve(ProposalSource::Mtp, 0, true);
        for generated in 1..=1024 {
            profile.observe(ProposalSource::Mtp, generated, 1, 1, true, None);
        }
        let means = profile.curve(ProposalSource::Mtp, 1024, false);
        let upper = profile.curve(ProposalSource::Mtp, 1024, true);
        assert!(means[0] > 0.99);
        assert!(means[1] < 0.01);
        assert!(upper[1] > before[1] + 0.5);
        assert!(upper.windows(2).all(|p| p[0] >= p[1]));
    }

    #[test]
    fn probability_feedback_replaces_the_binary_sample_and_respects_censoring() {
        let mut profile = AcceptanceProfile::default();
        let survival = [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.01, -1., -1., -1., -1., -1., -1., -1., -1.];
        profile.observe(ProposalSource::Mtp, 0, 7, 0, true, Some(&survival));
        for depth in 0..7 {
            let expected = (2.0 * 0.5f64.powi(depth as i32 + 1) + f64::from(survival[depth])) / 3.0;
            assert!((profile.prefix_probability(ProposalSource::Mtp, depth).unwrap() - expected).abs() < 1e-12);
        }
        let before = profile.curve(ProposalSource::Mtp, 0, false);
        profile.observe(ProposalSource::Mtp, 0, 7, 1, false, Some(&survival));
        assert_eq!(profile.curve(ProposalSource::Mtp, 0, false)[1..], before[1..]);
    }

    #[test]
    fn deployment_acceptance_learns_across_requests_with_bounded_memory() {
        let mut deployment = DeploymentAcceptance::new(8);
        for _ in 0..2048 {
            deployment.observe(ProposalSource::Mtp, 5, 4, true, None);
        }
        let means = deployment.means(ProposalSource::Mtp);
        for depth in 0..4 {
            assert!((means[depth] - 1.0).abs() < 1e-3);
        }
        assert!(means[4].abs() < 1e-3);
        assert!(deployment.observations(ProposalSource::Mtp).iter()
            .all(|weight| *weight <= DEPLOYMENT_SAMPLE_CAP));
    }

    #[test]
    fn deployment_acceptance_ignores_censored_depths() {
        let mut deployment = DeploymentAcceptance::new(8);
        let before = deployment.means(ProposalSource::Mtp).to_vec();
        deployment.observe(ProposalSource::Mtp, 2, 1, false, None);
        let means = deployment.means(ProposalSource::Mtp);
        assert!((means[0] - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(means[1], before[1]);
        assert_eq!(means[2..], before[2..]);
        deployment.observe(ProposalSource::Mtp, 1, 0, true, None);
        assert!((deployment.means(ProposalSource::Mtp)[0] - 0.5).abs() < 1e-12);
        assert_eq!(deployment.means(ProposalSource::Mtp)[1], before[1]);
    }

    #[test]
    fn deployment_acceptance_uses_survival_probabilities() {
        let mut deployment = DeploymentAcceptance::new(8);
        let survival = [1.0f32, 0.5, 0.25, -1.0, -1.0, -1.0, -1.0, -1.0];
        deployment.observe(ProposalSource::Mtp, 3, 0, true, Some(&survival));
        let means = deployment.means(ProposalSource::Mtp);
        assert!((means[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!((means[1] - 1.0 / 3.0).abs() < 1e-12);
        assert!((means[2] - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn seeded_request_keeps_the_prior_weight_and_stays_request_local() {
        let mut deployment = DeploymentAcceptance::new(8);
        for _ in 0..32 {
            deployment.observe(ProposalSource::Mtp, 5, 4, true, None);
        }
        let seeded = deployment.means(ProposalSource::Mtp).to_vec();
        let mut profile = AcceptanceProfile::new(8);
        profile.seed_means(ProposalSource::Mtp, &seeded);
        for (depth, mean) in seeded.iter().enumerate() {
            assert_eq!(profile.prefixes[0][depth].mean, *mean);
            assert_eq!(profile.prefixes[0][depth].weight, 2.0);
        }
        profile.observe(ProposalSource::Mtp, 0, 5, 1, true, None);
        assert_eq!(profile.prefixes[0][0].weight, 3.0);
        assert_eq!(deployment.means(ProposalSource::Mtp), seeded.as_slice());
    }
}

#[cfg(test)]
#[test]
fn arbitrary_width_catalog_preserves_declared_values() {
    validate_widths(&[1, 8, 13, 16, 32]).unwrap();
    assert!(validate_widths(&[1, 0]).is_err());
    assert_eq!(graph_catalog(&[1, 8, 13, 16, 32], None, 1)[0], vec![1, 8, 13, 16, 32]);
    assert_eq!(graph_catalog(&[1, 8, 32], Some(32), 1)[0], vec![32]);
    assert!(graph_catalog(&[1, 8, 32], None, 1)[1].is_empty());
}

#[cfg(test)]
#[test]
fn acceptance_profiles_follow_package_capacity() {
    for capacity in [1, 8, 13, 16, 32, 257] {
        let mut profile = AcceptanceProfile::new(capacity);
        assert_eq!(profile.curve(ProposalSource::Mtp, 0, false).len(), capacity - 1);
        profile.observe(ProposalSource::Mtp, 0, capacity - 1, capacity - 1, false, None);
        assert_eq!(profile.rounds(), u64::from(capacity > 1));
        assert!(profile.prefix_probability(ProposalSource::Mtp, capacity - 1).is_none());
    }
}

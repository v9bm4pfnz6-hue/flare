/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scheduler simulation over acceptance traces recorded by window A-20261110-8.
//!
//! The traces are the per-depth offered/accepted counts from
//! optimization-team/evidence/A-20261110-8/calibration/alpha1.0-cold.json,
//! aggregated over the three runs of each prompt kind and context. They capture
//! the production under-verification: depth 1 is offered in almost every round,
//! while depths 2-5 are offered only in the prose and some reasoning cases.
//! The simulation replays these survival curves through the real request
//! profiles and the deployment-wide estimator, so it exercises the seeded-prior
//! behaviour without a GPU.

use super::speculation::{AcceptanceProfile, DeploymentAcceptance, ProposalSource};

const BREAK_EVEN_SURVIVAL: f64 = 0.28;
const MAX_WIDTH: usize = 6;

struct Trace {
    offered: [u32; 5],
    accepted: [u32; 5],
}

const TRACES: [Trace; 9] = [
    Trace { offered: [381, 10, 10, 9, 9], accepted: [341, 9, 9, 6, 5] },
    Trace { offered: [354, 82, 82, 3, 3], accepted: [309, 52, 39, 0, 0] },
    Trace { offered: [397, 4, 4, 3, 3], accepted: [348, 4, 4, 1, 0] },
    Trace { offered: [397, 3, 3, 3, 3], accepted: [343, 1, 1, 1, 1] },
    Trace { offered: [418, 1, 1, 1, 1], accepted: [338, 0, 0, 0, 0] },
    Trace { offered: [415, 3, 2, 2, 2], accepted: [340, 0, 0, 0, 0] },
    Trace { offered: [438, 19, 19, 19, 19], accepted: [287, 9, 6, 4, 2] },
    Trace { offered: [387, 94, 94, 94, 94], accepted: [261, 42, 24, 20, 18] },
    Trace { offered: [389, 83, 83, 83, 83], accepted: [269, 42, 24, 16, 11] },
];

fn survival(trace: &Trace) -> [f64; 5] {
    std::array::from_fn(|depth| {
        if trace.offered[depth] == 0 {
            0.0
        } else {
            f64::from(trace.accepted[depth]) / f64::from(trace.offered[depth])
        }
    })
}

fn next_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 33) as f64) / ((1u64 << 31) as f64)
}

/// Production adaptive width choice under the break-even rule: offer one more
/// draft while its forecast prefix survival stays at or above break-even.
fn chosen_width(profile: &AcceptanceProfile) -> usize {
    let curve = profile.curve(ProposalSource::Mtp, 0, false);
    let mut width = 1;
    for depth in 0..MAX_WIDTH.saturating_sub(1) {
        if curve[depth] >= BREAK_EVEN_SURVIVAL {
            width = depth + 2;
        } else {
            break;
        }
    }
    width
}

#[test]
fn seeded_request_from_recorded_deep_acceptance_verifies_more() {
    let mut deployment = DeploymentAcceptance::new(MAX_WIDTH + 1);
    let prose = survival(&TRACES[7]);
    for _ in 0..64 {
        let offered = 5;
        // Sequential prefix survival: accepted is the first depth below 0.2.
        let accepted = prose.iter().take(offered).take_while(|p| **p >= 0.2).count();
        deployment.observe(ProposalSource::Mtp, offered, accepted, accepted < offered, None);
    }
    let mut legacy = AcceptanceProfile::new(MAX_WIDTH + 1);
    let mut seeded = AcceptanceProfile::new(MAX_WIDTH + 1);
    seeded.seed_means(ProposalSource::Mtp, deployment.means(ProposalSource::Mtp));
    assert_eq!(chosen_width(&legacy), 2);
    assert!(chosen_width(&seeded) > chosen_width(&legacy));
    // A seeded profile keeps the fixed observation weight of a new request.
    assert_eq!(seeded.information(ProposalSource::Mtp, 0, 0), 2.0);
    legacy.observe(ProposalSource::Mtp, 0, 2, 2, false, None);
}

#[test]
fn seeded_request_from_low_acceptance_trace_stays_shallow() {
    let mut deployment = DeploymentAcceptance::new(MAX_WIDTH + 1);
    let code = survival(&TRACES[4]);
    for _ in 0..64 {
        let offered = 5;
        let accepted = code.iter().take(offered).take_while(|p| **p >= 0.1).count();
        deployment.observe(ProposalSource::Mtp, offered, accepted, accepted < offered, None);
    }
    let mut seeded = AcceptanceProfile::new(MAX_WIDTH + 1);
    seeded.seed_means(ProposalSource::Mtp, deployment.means(ProposalSource::Mtp));
    assert!(chosen_width(&seeded) <= 2);
}

#[test]
fn single_rejection_after_startup_keeps_next_request_above_break_even() {
    let mut deployment = DeploymentAcceptance::new(MAX_WIDTH + 1);
    deployment.observe(ProposalSource::Mtp, 1, 0, true, None);
    let mut profile = AcceptanceProfile::new(MAX_WIDTH + 1);
    profile.seed_means(ProposalSource::Mtp, deployment.means(ProposalSource::Mtp));
    assert!(profile.curve(ProposalSource::Mtp, 0, false)[0] > BREAK_EVEN_SURVIVAL,
        "depth-0 mean after one rejection must stay above break-even");
}

#[test]
fn simulation_over_recorded_traces_improves_tokens_with_bounded_memory() {
    let simulate = |seeded: bool| {
        let mut deployment = DeploymentAcceptance::new(MAX_WIDTH + 1);
        let mut state = 0x2026_1110u64;
        let mut tokens = 0u64;
        for request in 0..18u64 {
            let mut profile = AcceptanceProfile::new(MAX_WIDTH + 1);
            if seeded {
                profile.seed_means(ProposalSource::Mtp, deployment.means(ProposalSource::Mtp));
            }
            for round in 0..64u64 {
                let trace = &TRACES[((request + round) % TRACES.len() as u64) as usize];
                let truth = survival(trace);
                let width = chosen_width(&profile);
                // Prefix survival: acceptance is sequential, so the accepted
                // count is the first failing depth (deterministic trace draw).
                let mut accepted = 0usize;
                for depth in 0..width {
                    if next_unit(&mut state) < truth[depth] {
                        accepted += 1;
                    } else {
                        break;
                    }
                }
                let rejected = accepted < width;
                profile.observe(ProposalSource::Mtp, 0, width, accepted, rejected, None);
                deployment.observe(ProposalSource::Mtp, width, accepted, rejected, None);
                tokens += accepted as u64 + 1;
            }
        }
        (tokens, deployment)
    };
    let (legacy_tokens, _) = simulate(false);
    let (seeded_tokens, deployment) = simulate(true);
    assert!(seeded_tokens > legacy_tokens,
        "seeded {seeded_tokens} must exceed legacy {legacy_tokens}");
    assert!(deployment.observations(ProposalSource::Mtp).iter()
        .all(|weight| *weight <= super::speculation::DEPLOYMENT_SAMPLE_CAP));
    // The simulation samples every trace equally, so the deployment mean
    // converges to the unweighted mean of the per-trace depth-0 survival.
    let expected = TRACES.iter()
        .map(|trace| f64::from(trace.accepted[0]) / f64::from(trace.offered[0]))
        .sum::<f64>() / TRACES.len() as f64;
    let estimated = deployment.means(ProposalSource::Mtp)[0];
    assert!((estimated - expected).abs() < 0.05, "{estimated} vs {expected}");
}

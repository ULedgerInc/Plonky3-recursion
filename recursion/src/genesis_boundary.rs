//! In-circuit genesis boundary (ULedger Plan C).
//!
//! Binds the exposed `genesis_id` (the leading-8 public values of a genesis
//! block's wrap proof) to the off-circuit `block_commitment(genesis)`. The
//! recursion verifier circuit runs over an extension challenge field, but
//! `block_commitment` is a base-field Poseidon2 `PaddingFreeSponge<_, 16, 8, 8>`;
//! the D=1 base permutation (enabled by the embedding application's recursion
//! config) lets us hash base-field values in-circuit.
//!
//! ## Layout is hardcoded here (and cross-checked downstream)
//!
//! This crate cannot depend on the embedding application's commitment crate, so
//! the `DomainTag` discriminants, the digest length, and the AIR public-value
//! layout are duplicated as constants below. A downstream cross-check test
//! (`genesis_boundary_parity` in the ULedger node) asserts the in-circuit digest
//! equals the off-circuit `commitment::block_commitment`, catching any drift.

use alloc::vec::Vec;

use p3_circuit::ops::poseidon2_perm::Poseidon2PermCallBase;
use p3_circuit::ops::Poseidon2Config;
use p3_circuit::{CircuitBuilder, CircuitBuilderError, ExprId};
use p3_field::Field;

/// Poseidon2 digest length (base-field elements). Mirrors
/// `plonky3_core::commitment::DIGEST_LEN`.
pub const DIGEST_LEN: usize = 8;

/// `DomainTag::Meta` discriminant (`plonky3_core::commitment::DomainTag`).
pub const META_TAG: u32 = 0x50_33_00_30;
/// `DomainTag::Block` discriminant (`plonky3_core::commitment::DomainTag`).
pub const BLOCK_TAG: u32 = 0x50_33_00_40;

// AIR public-value layout (mirrors `plonky3_core::air`, post Plan C 14->30 growth).
/// First limb of the exposed `genesis_id` (leading-8) slot.
pub const PI_XRC: usize = 0;
/// First limb of `tx_root`.
pub const PI_TX_ROOT: usize = 8;
/// First limb of `xm_root`.
pub const PI_XM_ROOT: usize = 16;
/// `block_height` scalar.
pub const PI_BLOCK_HEIGHT: usize = 24;
/// `previous_block_height` scalar.
pub const PI_PREV_HEIGHT: usize = 25;
/// `transaction_count` scalar.
pub const PI_TX_COUNT: usize = 26;
/// `cross_reference_count` scalar.
pub const PI_XM_COUNT: usize = 27;

/// In-circuit `PaddingFreeSponge<_, 16, 8, 8>` over D=1 base-field inputs,
/// mirroring [`p3_symmetric::PaddingFreeSponge::hash_iter`] (overwrite mode).
///
/// `buf` holds the full preimage ExprIds (already `(tag, len)`-prefixed by the
/// caller, e.g. via [`in_circuit_hash_fields`]). The D=1 base Poseidon2 op must
/// already be enabled on `builder`.
///
/// The base permutation op is itself a sponge primitive: capacity is never
/// witness-fed (chained between rows, zeroed on `new_start`), and unsupplied
/// rate slots inherit the previous row's output. We feed only the rate block per
/// row and toggle `new_start` on the first row, reproducing overwrite mode.
pub fn in_circuit_padding_free_sponge<F: Field>(
    builder: &mut CircuitBuilder<F>,
    buf: &[ExprId],
) -> Result<[ExprId; DIGEST_LEN], CircuitBuilderError> {
    let mut last: Option<[Option<ExprId>; 16]> = None;
    let mut pos = 0usize;
    let mut new_start = true;
    loop {
        let mut inputs: [Option<ExprId>; 16] = [None; 16];
        let mut i = 0;
        while i < 8 && pos < buf.len() {
            inputs[i] = Some(buf[pos]);
            pos += 1;
            i += 1;
        }
        if i == 0 {
            // Exact multiple of RATE (or empty): hash_iter breaks at the top of
            // the loop without a trailing permutation.
            break;
        }
        let is_full = i == 8;
        let is_final = !is_full || pos >= buf.len();
        // Only the final row's rate output is consumed (the digest). Exposing
        // intermediate rows would leave unmatched CTL receives
        // (GlobalCumulativeMismatch); intermediate rows chain via the internal
        // capacity carry, so they need no exposure.
        let call = Poseidon2PermCallBase {
            config: Poseidon2Config::BabyBearD1Width16,
            new_start,
            inputs,
            out_ctl: if is_final { [true; 8] } else { [false; 8] },
            return_all_outputs: false,
        };
        let (_op, outputs) = builder.add_poseidon2_perm_base(&call)?;
        new_start = false;
        if is_final {
            last = Some(outputs);
            break;
        }
    }
    let outputs = last.expect("non-empty preimage yields at least one permutation");
    Ok(core::array::from_fn(|i| {
        outputs[i].expect("final-row rate output exposed via out_ctl")
    }))
}

/// In-circuit analog of `plonky3_core::commitment::hash_fields`: prepends the
/// `(tag, len)` prefix as in-circuit constants and sponges the result.
pub fn in_circuit_hash_fields<F: Field>(
    builder: &mut CircuitBuilder<F>,
    tag: u32,
    inputs: &[ExprId],
) -> Result<[ExprId; DIGEST_LEN], CircuitBuilderError> {
    let mut buf = Vec::with_capacity(inputs.len() + 2);
    buf.push(builder.define_const(F::from_u64(u64::from(tag))));
    buf.push(builder.define_const(F::from_u64(inputs.len() as u64)));
    buf.extend_from_slice(inputs);
    in_circuit_padding_free_sponge(builder, &buf)
}

/// In-circuit analog of `plonky3_core::commitment::meta_commitment`:
/// `hash_fields(Meta, [block_height, prev_height, tx_count, xm_count, params(8)])`.
pub fn in_circuit_meta_commitment<F: Field>(
    builder: &mut CircuitBuilder<F>,
    block_height: ExprId,
    prev_height: ExprId,
    tx_count: ExprId,
    xm_count: ExprId,
    params: &[ExprId; DIGEST_LEN],
) -> Result<[ExprId; DIGEST_LEN], CircuitBuilderError> {
    let mut payload = Vec::with_capacity(4 + DIGEST_LEN);
    payload.push(block_height);
    payload.push(prev_height);
    payload.push(tx_count);
    payload.push(xm_count);
    payload.extend_from_slice(params);
    in_circuit_hash_fields(builder, META_TAG, &payload)
}

/// In-circuit analog of `plonky3_core::commitment::block_commitment`:
/// `hash_fields(Block, [height, prev_chain(8), tx_root(8), xm_root(8), meta(8),
/// prev_proof(8)])`.
pub fn in_circuit_block_commitment<F: Field>(
    builder: &mut CircuitBuilder<F>,
    block_height: ExprId,
    prev_chain: &[ExprId; DIGEST_LEN],
    tx_root: &[ExprId; DIGEST_LEN],
    xm_root: &[ExprId; DIGEST_LEN],
    meta: &[ExprId; DIGEST_LEN],
    prev_proof: &[ExprId; DIGEST_LEN],
) -> Result<[ExprId; DIGEST_LEN], CircuitBuilderError> {
    let mut payload = Vec::with_capacity(1 + 5 * DIGEST_LEN);
    payload.push(block_height);
    payload.extend_from_slice(prev_chain);
    payload.extend_from_slice(tx_root);
    payload.extend_from_slice(xm_root);
    payload.extend_from_slice(meta);
    payload.extend_from_slice(prev_proof);
    in_circuit_hash_fields(builder, BLOCK_TAG, &payload)
}

/// Apply the genesis boundary: re-derive `meta` and `block_commitment`
/// in-circuit from the bound base public values (`air_public_targets`) plus the
/// fixed `params` digest limbs, then constrain the exposed leading-8
/// (`genesis_id`) equal to `block_commitment`. Genesis-only (`prev_chain` /
/// `prev_proof` are EMPTY constants).
///
/// `params` are the `params_digest()` limbs already embedded into the circuit
/// field. `air_public_targets` must follow the Plan C 30-wide layout.
pub fn apply_genesis_boundary<F: Field>(
    builder: &mut CircuitBuilder<F>,
    air_public_targets: &[ExprId],
    params: &[F],
) -> Result<(), CircuitBuilderError> {
    assert!(
        air_public_targets.len() >= PI_XM_COUNT + 1,
        "genesis boundary needs the 30-wide public-value layout"
    );
    assert_eq!(params.len(), DIGEST_LEN, "params digest must be 8 limbs");

    let block_height = air_public_targets[PI_BLOCK_HEIGHT];
    let prev_height = air_public_targets[PI_PREV_HEIGHT];
    let tx_count = air_public_targets[PI_TX_COUNT];
    let xm_count = air_public_targets[PI_XM_COUNT];
    let tx_root: [ExprId; DIGEST_LEN] =
        core::array::from_fn(|i| air_public_targets[PI_TX_ROOT + i]);
    let xm_root: [ExprId; DIGEST_LEN] =
        core::array::from_fn(|i| air_public_targets[PI_XM_ROOT + i]);
    let params_e: [ExprId; DIGEST_LEN] =
        core::array::from_fn(|i| builder.define_const(params[i]));

    let zero = builder.define_const(F::ZERO);
    let empty = [zero; DIGEST_LEN];

    let meta = in_circuit_meta_commitment(
        builder,
        block_height,
        prev_height,
        tx_count,
        xm_count,
        &params_e,
    )?;
    let block = in_circuit_block_commitment(
        builder,
        block_height,
        &empty,
        &tx_root,
        &xm_root,
        &meta,
        &empty,
    )?;

    // Constrain genesis_id (leading-8) == block_commitment. Consume each
    // CTL-exposed digest limb via an ALU `sub`+`assert_zero` (not `connect`):
    // the exposed output is a CTL receive from the Poseidon2 table, and it must
    // be used by a live witness so its contribution is matched (else
    // GlobalCumulativeMismatch). `connect` would alias the witness away.
    for i in 0..DIGEST_LEN {
        let diff = builder.sub(block[i], air_public_targets[PI_XRC + i]);
        builder.assert_zero(diff);
    }
    Ok(())
}

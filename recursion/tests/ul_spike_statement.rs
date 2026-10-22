use std::time::Instant;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::BabyBear;
use p3_circuit::ops::NpoTypeId;
use p3_circuit::tables::Traces;
use p3_circuit::{Circuit, CircuitBuilder, StatementExport, StatementSchema};
use p3_circuit_prover::{
    AirVariant, BatchStarkProof, BatchStarkProver, CircuitVerifier, PreparedCircuitProver,
    StatementAirBuilder, StatementPreprocessor, StatementProver,
};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_lookup::logup::LogUpGadget;
use p3_matrix::dense::RowMajorMatrix;
use p3_recursion::backend::fri::{FriRecursionBackendForExt, FriVerifierResult};
use p3_recursion::builtin_config::{
    BabyBearD4Poseidon2BinaryConfig, FriConfigV1, SuiteIdV1, baby_bear_d4_poseidon2_binary,
};
use p3_recursion::verifier::{verify_p3_uni_proof_circuit, verify_trusted_p3_batch_proof_circuit};
use p3_recursion::{
    BatchOnly, FriRecursionBackend, FriRecursionConfig, PcsRecursionBackend, Poseidon2Config,
    ProveNextLayerParams, RecursionInput, StarkVerifierInputsBuilder, Target,
    TrustedPcsRecursionBackend, VerifierCircuitResult, VerifierLimits,
};
use p3_uni_stark::Proof;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type C = BabyBearD4Poseidon2BinaryConfig;
type Backend = FriRecursionBackendForExt<4, 16, 8, Poseidon2Config>;
type Comm = <C as FriRecursionConfig>::Commitment;
type InputProof = <C as FriRecursionConfig>::InputProof;
type OpeningProof = <C as FriRecursionConfig>::OpeningProof;

const PERM: Poseidon2Config = Poseidon2Config::BABY_BEAR_D4_W16;
const GENESIS_LEN: usize = 8;
const HEIGHT: usize = 8;
const WINDOW: usize = 9;
const ROWS: usize = 32;

struct BlockAir;

impl<T> BaseAir<T> for BlockAir {
    fn width(&self) -> usize {
        WINDOW
    }

    fn num_public_values(&self) -> usize {
        WINDOW
    }
}

impl<AB: AirBuilder> Air<AB> for BlockAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice().to_vec();
        let next = main.next_slice().to_vec();
        let public = builder.public_values().to_vec();
        for i in 0..WINDOW {
            builder.when_first_row().assert_eq(local[i], public[i]);
            builder.when_transition().assert_eq(local[i], next[i]);
        }
    }
}

struct PreparedShape {
    circuit: Circuit<EF>,
    prover: PreparedCircuitProver<C>,
    verifier: CircuitVerifier<C>,
}

struct WrapLayer {
    shape: PreparedShape,
    block: FriVerifierResult<C>,
}

struct FoldLayer {
    shape: PreparedShape,
    head: FriVerifierResult<C>,
    block: FriVerifierResult<C>,
}

const fn descriptor(log_blowup: u32, num_queries: u32, pow_bits: u32) -> FriConfigV1 {
    FriConfigV1::new(
        SuiteIdV1::BabyBearD4Poseidon2BinaryFri,
        log_blowup,
        0,
        2,
        num_queries,
        pow_bits,
        pow_bits,
        0,
        0,
        0,
        0,
    )
}

fn config(descriptor: &FriConfigV1) -> C {
    baby_bear_d4_poseidon2_binary(descriptor, &VerifierLimits::default())
        .expect("built-in BabyBear D4 config")
}

fn backend() -> Backend {
    FriRecursionBackend::<16, 8, _>::new(PERM).for_extension_degree::<4>()
}

fn block_values(genesis: u32, height: u32) -> Vec<F> {
    (0..GENESIS_LEN as u32)
        .map(|i| F::from_u32(genesis + i))
        .chain([F::from_u32(height)])
        .collect()
}

fn prove_block(config: &C, values: &[F]) -> Proof<C> {
    let trace = RowMajorMatrix::new(values.repeat(ROWS), WINDOW);
    let proof = p3_uni_stark::prove(config, &BlockAir, trace, values).expect("block proof");
    p3_uni_stark::verify(config, &BlockAir, &proof, values).expect("native block verify");
    proof
}

fn uni_input<'a>(proof: &'a Proof<C>, values: &[F]) -> RecursionInput<'a, C, BlockAir> {
    RecursionInput::UniStark {
        proof,
        air: &BlockAir,
        public_inputs: values.to_vec(),
        preprocessed_commit: None,
    }
}

fn batch_input<'a>(
    verifier: &'a CircuitVerifier<C>,
    proof: &'a BatchStarkProof<C>,
    statement: &[F],
) -> RecursionInput<'a, C, BatchOnly> {
    RecursionInput::BatchStark {
        proof,
        common_data: verifier.common_data(),
        table_public_inputs: verifier
            .table_public_values(statement)
            .expect("statement matches the retained schema"),
    }
}

fn verify_block_child(
    builder: &mut CircuitBuilder<EF>,
    config: &C,
    proof: &Proof<C>,
) -> (Vec<Target>, FriVerifierResult<C>) {
    let inputs = StarkVerifierInputsBuilder::<C, Comm, OpeningProof>::try_allocate(
        builder, proof, None, WINDOW,
    )
    .expect("allocate block child");
    let op_ids =
        verify_p3_uni_proof_circuit::<BlockAir, C, Comm, InputProof, OpeningProof, _, 16, 8>(
            config,
            &BlockAir,
            builder,
            &inputs.proof_targets,
            &inputs.air_public_targets,
            &inputs.preprocessed_commit,
            config.pcs_verifier_params(),
            PERM,
        )
        .expect("verify block child in circuit");
    (
        inputs.air_public_targets.clone(),
        FriVerifierResult::UniStark(inputs, op_ids, VerifierLimits::default()),
    )
}

fn verify_head_child(
    builder: &mut CircuitBuilder<EF>,
    verifier: &CircuitVerifier<C>,
    proof: &BatchStarkProof<C>,
    statement: &[F],
) -> (Vec<Target>, FriVerifierResult<C>) {
    let (inputs, op_ids) = verify_trusted_p3_batch_proof_circuit::<
        C,
        Comm,
        InputProof,
        OpeningProof,
        LogUpGadget,
        Poseidon2Config,
        16,
        8,
        4,
    >(
        verifier,
        builder,
        proof,
        statement,
        verifier.config().pcs_verifier_params(),
        &LogUpGadget::new(),
        PERM,
    )
    .expect("verify head child in circuit");
    let statement_table = verifier
        .statement_layout()
        .table_instance()
        .expect("head exports a statement");
    (
        inputs.air_public_targets[statement_table].clone(),
        FriVerifierResult::BatchStark(inputs, op_ids, VerifierLimits::default()),
    )
}

fn prepare_shape(
    config: &C,
    backend: &Backend,
    circuit: Circuit<EF>,
    schema: &StatementSchema,
) -> PreparedShape {
    let params = ProveNextLayerParams::default();
    let mut preprocessors =
        <Backend as PcsRecursionBackend<C, BatchOnly, 4>>::non_primitive_preprocessors(backend);
    let mut air_builders =
        <Backend as PcsRecursionBackend<C, BatchOnly, 4>>::non_primitive_air_builders(backend);
    let mut provers =
        <Backend as PcsRecursionBackend<C, BatchOnly, 4>>::non_primitive_provers(backend, 4);
    preprocessors.push(Box::new(StatementPreprocessor::new(schema.clone())));
    air_builders.push(Box::new(StatementAirBuilder::<4>::new(schema.clone())));
    provers.push(Box::new(StatementProver::<4>::new(schema.clone())));
    let mut prover = BatchStarkProver::new(config.clone())
        .with_table_packing(params.table_packing)
        .with_alu_variant(AirVariant::Baseline);
    for table in provers {
        prover.register_table_prover(table);
    }
    let prover = prover
        .prepare_circuit::<EF, 4>(
            &circuit,
            &preprocessors,
            &air_builders,
            params.constraint_profile,
        )
        .expect("prepare circuit");
    let verifier = prover.verifier();
    PreparedShape {
        circuit,
        prover,
        verifier,
    }
}

fn build_wrap(config: &C, backend: &Backend, block: &Proof<C>) -> WrapLayer {
    let mut builder = CircuitBuilder::<EF>::new();
    <Backend as PcsRecursionBackend<C, BlockAir, 4>>::prepare_circuit(
        backend,
        config,
        &mut builder,
    )
    .expect("enable verifier ops");
    let (block_targets, block_result) = verify_block_child(&mut builder, config, block);
    let exports = block_targets
        .iter()
        .map(|&target| StatementExport::Base(target))
        .collect::<Vec<_>>();
    let schema = builder
        .set_statement_exports::<F>(&exports)
        .expect("export block window");
    let circuit = builder.build().expect("build wrap circuit");
    WrapLayer {
        shape: prepare_shape(config, backend, circuit, &schema),
        block: block_result,
    }
}

fn prove_wrap(
    wrap: &WrapLayer,
    config: &C,
    backend: &Backend,
    block: &Proof<C>,
    values: &[F],
) -> BatchStarkProof<C> {
    let prev = uni_input(block, values);
    let mut runner = wrap.shape.circuit.runner();
    runner
        .set_public_inputs(&wrap.block.pack_public_inputs(&prev).expect("pack public"))
        .expect("set public");
    runner
        .set_private_inputs(&wrap.block.pack_private_inputs(&prev).expect("pack private"))
        .expect("set private");
    <Backend as PcsRecursionBackend<C, BlockAir, 4>>::set_private_data(
        backend,
        config,
        &mut runner,
        <FriVerifierResult<C> as VerifierCircuitResult<C, BlockAir>>::op_ids(&wrap.block),
        &prev,
    )
    .expect("block FRI private data");
    let traces = runner.run().expect("run wrap circuit");
    wrap.shape.prover.prove(&traces).expect("prove wrap")
}

fn build_fold(
    config: &C,
    backend: &Backend,
    head_verifier: &CircuitVerifier<C>,
    head: &BatchStarkProof<C>,
    head_statement: &[F],
    block: &Proof<C>,
) -> FoldLayer {
    let mut builder = CircuitBuilder::<EF>::new();
    <Backend as PcsRecursionBackend<C, BatchOnly, 4>>::prepare_circuit(
        backend,
        config,
        &mut builder,
    )
    .expect("enable verifier ops");
    let (head_targets, head_result) =
        verify_head_child(&mut builder, head_verifier, head, head_statement);
    let (block_targets, block_result) = verify_block_child(&mut builder, config, block);
    for i in 0..GENESIS_LEN {
        let difference = builder.sub(head_targets[i], block_targets[i]);
        builder.assert_zero(difference);
    }
    let one = builder.define_const(EF::ONE);
    let next_height = builder.add(head_targets[HEIGHT], one);
    let difference = builder.sub(block_targets[HEIGHT], next_height);
    builder.assert_zero(difference);
    let exports = block_targets
        .iter()
        .map(|&target| StatementExport::Base(target))
        .chain([StatementExport::Base(next_height)])
        .collect::<Vec<_>>();
    let schema = builder
        .set_statement_exports::<F>(&exports)
        .expect("export block window and derived height");
    let circuit = builder.build().expect("build fold circuit");
    FoldLayer {
        shape: prepare_shape(config, backend, circuit, &schema),
        head: head_result,
        block: block_result,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_fold(
    fold: &FoldLayer,
    config: &C,
    backend: &Backend,
    head_verifier: &CircuitVerifier<C>,
    head: &BatchStarkProof<C>,
    head_statement: &[F],
    block: &Proof<C>,
    block_values: &[F],
) -> Result<Traces<EF>, String> {
    head_verifier
        .verify(head, head_statement)
        .map_err(|e| format!("native head verify: {e:?}"))?;
    let head_prev = batch_input(head_verifier, head, head_statement);
    let block_prev = uni_input(block, block_values);
    let mut public = fold
        .head
        .pack_public_inputs(&head_prev)
        .map_err(|e| format!("{e:?}"))?;
    public.extend(
        fold.block
            .pack_public_inputs(&block_prev)
            .map_err(|e| format!("{e:?}"))?,
    );
    let mut private = fold
        .head
        .pack_private_inputs(&head_prev)
        .map_err(|e| format!("{e:?}"))?;
    private.extend(
        fold.block
            .pack_private_inputs(&block_prev)
            .map_err(|e| format!("{e:?}"))?,
    );
    let mut runner = fold.shape.circuit.runner();
    runner
        .set_public_inputs(&public)
        .map_err(|e| format!("{e:?}"))?;
    runner
        .set_private_inputs(&private)
        .map_err(|e| format!("{e:?}"))?;
    <Backend as TrustedPcsRecursionBackend<C, BatchOnly, 4>>::set_private_data_for_trusted_batch(
        backend,
        head_verifier,
        head,
        head_statement,
        &mut runner,
        <FriVerifierResult<C> as VerifierCircuitResult<C, BatchOnly>>::op_ids(&fold.head),
    )
    .map_err(|e| format!("{e:?}"))?;
    <Backend as PcsRecursionBackend<C, BlockAir, 4>>::set_private_data(
        backend,
        config,
        &mut runner,
        <FriVerifierResult<C> as VerifierCircuitResult<C, BlockAir>>::op_ids(&fold.block),
        &block_prev,
    )
    .map_err(|e| format!("{e:?}"))?;
    runner.run().map_err(|e| format!("{e:?}"))
}

fn expected_fold_statement(head_statement: &[F], block_values: &[F]) -> Vec<F> {
    block_values
        .iter()
        .copied()
        .chain([head_statement[HEIGHT] + F::ONE])
        .collect()
}

fn assert_every_limb_bound(
    verifier: &CircuitVerifier<C>,
    proof: &BatchStarkProof<C>,
    expected: &[F],
) {
    for limb in 0..expected.len() {
        let mut tampered = expected.to_vec();
        tampered[limb] += F::ONE;
        assert!(
            verifier.verify(proof, &tampered).is_err(),
            "statement limb {limb} is not bound"
        );
    }
}

fn run_spike(descriptor: &FriConfigV1) {
    let started = Instant::now();
    let config = config(descriptor);
    let backend = backend();

    let a1 = block_values(10, 1);
    let a2 = block_values(10, 2);
    let a3 = block_values(10, 3);
    let b1 = block_values(50, 1);
    let b2 = block_values(50, 2);
    let a1_proof = prove_block(&config, &a1);
    let a2_proof = prove_block(&config, &a2);
    let a3_proof = prove_block(&config, &a3);
    let b1_proof = prove_block(&config, &b1);
    let b2_proof = prove_block(&config, &b2);

    let wrap = build_wrap(&config, &backend, &a1_proof);
    let proving = Instant::now();
    let wrap_a = prove_wrap(&wrap, &config, &backend, &a1_proof, &a1);
    let wrap_elapsed = proving.elapsed();
    let wrap_b = prove_wrap(&wrap, &config, &backend, &b1_proof, &b1);
    wrap.shape
        .verifier
        .verify(&wrap_a, &a1)
        .expect("wrap A control");
    wrap.shape
        .verifier
        .verify(&wrap_b, &b1)
        .expect("wrap B control");
    assert_every_limb_bound(&wrap.shape.verifier, &wrap_a, &a1);

    let head_verifier = wrap.shape.verifier.clone();
    drop(wrap);
    let fold = build_fold(&config, &backend, &head_verifier, &wrap_a, &a1, &a2_proof);

    let traces_a = run_fold(
        &fold,
        &config,
        &backend,
        &head_verifier,
        &wrap_a,
        &a1,
        &a2_proof,
        &a2,
    )
    .expect("run fold A");
    let proving = Instant::now();
    let mut fold_a = fold.shape.prover.prove(&traces_a).expect("prove fold A");
    let fold_elapsed = proving.elapsed();
    let expected_a = expected_fold_statement(&a1, &a2);
    fold.shape
        .verifier
        .verify(&fold_a, &expected_a)
        .expect("fold A control");

    let traces_b = run_fold(
        &fold,
        &config,
        &backend,
        &head_verifier,
        &wrap_b,
        &b1,
        &b2_proof,
        &b2,
    )
    .expect("run fold B on the cached shape");
    let fold_b = fold.shape.prover.prove(&traces_b).expect("prove fold B");
    let expected_b = expected_fold_statement(&b1, &b2);
    fold.shape
        .verifier
        .verify(&fold_b, &expected_b)
        .expect("fold B control");

    assert_every_limb_bound(&fold.shape.verifier, &fold_a, &expected_a);
    assert!(fold.shape.verifier.verify(&fold_a, &expected_b).is_err());
    assert!(fold.shape.verifier.verify(&fold_b, &expected_a).is_err());

    let genesis_mismatch = run_fold(
        &fold,
        &config,
        &backend,
        &head_verifier,
        &wrap_a,
        &a1,
        &b2_proof,
        &b2,
    );
    assert!(
        matches!(&genesis_mismatch, Err(error) if error.contains("WitnessConflict")),
        "genesis equality must bite in the circuit"
    );
    let height_gap = run_fold(
        &fold,
        &config,
        &backend,
        &head_verifier,
        &wrap_a,
        &a1,
        &a3_proof,
        &a3,
    );
    assert!(
        matches!(&height_gap, Err(error) if error.contains("WitnessConflict")),
        "height + 1 must bite in the circuit"
    );
    let wrong_head_statement = run_fold(
        &fold,
        &config,
        &backend,
        &head_verifier,
        &wrap_a,
        &b1,
        &a2_proof,
        &a2,
    );
    assert!(
        wrong_head_statement.is_err(),
        "head statement must be checked natively"
    );

    let mut forged = expected_a.clone();
    forged[HEIGHT] += F::ONE;
    forged[WINDOW] += F::ONE;
    fold_a
        .non_primitives
        .iter_mut()
        .find(|entry| entry.op_type == NpoTypeId::statement())
        .expect("statement table")
        .public_values
        .clone_from(&forged);
    assert!(
        fold.shape.verifier.verify(&fold_a, &forged).is_err(),
        "a rewritten attached statement must fail the STARK check"
    );

    println!(
        "descriptor {descriptor:?}\n\
         wrap: statement len {}, rows {:?}, npo {:?}, prove {wrap_elapsed:?}\n\
         fold: statement len {}, rows {:?}, npo {:?}, prove {fold_elapsed:?}\n\
         genesis mismatch: {}\nheight gap: {}\nwrong head statement: {}\ntotal {:?}",
        a1.len(),
        wrap_a.rows,
        npo_rows(&wrap_a),
        expected_a.len(),
        fold_b.rows,
        npo_rows(&fold_b),
        genesis_mismatch.err().unwrap_or_default(),
        height_gap.err().unwrap_or_default(),
        wrong_head_statement.err().unwrap_or_default(),
        started.elapsed(),
    );
}

fn npo_rows(proof: &BatchStarkProof<C>) -> Vec<(String, usize)> {
    proof
        .non_primitives
        .iter()
        .map(|entry| (format!("{:?}", entry.op_type), entry.rows))
        .collect()
}

#[test]
fn spike_statement_export_over_batch_and_uni_children() {
    run_spike(&descriptor(1, 2, 0));
}

#[test]
fn spike_statement_export_with_frozen_fri_params() {
    run_spike(&descriptor(3, 28, 16));
}

#[test]
fn spike_prepared_owners_are_cacheable() {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PreparedCircuitProver<C>>();
    assert_send_sync::<CircuitVerifier<C>>();
    assert_send_sync::<Circuit<EF>>();
    assert_send_sync::<FriVerifierResult<C>>();
}

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::BabyBear;
use p3_circuit::tables::Traces;
use p3_circuit::{Circuit, CircuitBuilder, StatementExport, StatementSchema};
use p3_circuit_prover::{
    AirVariant, BatchStarkProof, BatchStarkProver, CircuitVerifier, PreparedCircuitProver,
    StatementAirBuilder, StatementPreprocessor, StatementProver,
};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use p3_recursion::backend::fri::{CheckedFriVerifierResult, FriRecursionBackendForExt};
use p3_recursion::builtin_config::{
    BabyBearD4Poseidon2BinaryConfig, FriConfigV1, SuiteIdV1, baby_bear_d4_poseidon2_binary,
};
use p3_recursion::{
    BatchOnly, FriRecursionBackend, ObservableCommitment, PcsRecursionBackend, Poseidon2Config,
    ProveNextLayerParams, RecursionInput, Target, TrustedPcsRecursionBackend,
    VerifierCircuitResult, VerifierLimits,
};
use p3_uni_stark::Proof;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type C = BabyBearD4Poseidon2BinaryConfig;
type Backend = FriRecursionBackendForExt<4, 16, 8, Poseidon2Config>;
type Checked = CheckedFriVerifierResult<C>;

const PERM: Poseidon2Config = Poseidon2Config::BABY_BEAR_D4_W16;
const WINDOW: usize = 9;
const ROWS: usize = 32;
const KEY_LEN: usize = 8;

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

fn config() -> C {
    baby_bear_d4_poseidon2_binary(
        &FriConfigV1::new(
            SuiteIdV1::BabyBearD4Poseidon2BinaryFri,
            1,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        &VerifierLimits::default(),
    )
    .expect("built-in BabyBear D4 config")
}

fn block_values(seed: u32) -> Vec<F> {
    (0..WINDOW as u32).map(|i| F::from_u32(seed + i)).collect()
}

fn prove_block(config: &C, values: &[F]) -> Proof<C> {
    let trace = RowMajorMatrix::new(values.repeat(ROWS), WINDOW);
    p3_uni_stark::prove(config, &BlockAir, trace, values).expect("block proof")
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

fn tag_all(builder: &mut CircuitBuilder<EF>, prefix: &str, targets: &[Target]) {
    for (i, &target) in targets.iter().enumerate() {
        builder
            .tag(target, format!("{prefix}{i}"))
            .expect("unique tag");
    }
}

fn probe_all(traces: &Traces<EF>, prefix: &str, len: usize) -> Vec<EF> {
    (0..len)
        .map(|i| {
            *traces
                .probe(&format!("{prefix}{i}"))
                .expect("tagged target")
        })
        .collect()
}

fn lift(values: &[F]) -> Vec<EF> {
    values.iter().copied().map(EF::from).collect()
}

fn native_key(verifier: &CircuitVerifier<C>) -> Vec<F> {
    verifier
        .common_data()
        .preprocessed
        .as_ref()
        .expect("child has preprocessed columns")
        .commitment
        .roots()
        .concat()
}

#[test]
fn child_preprocessed_targets_match_native_commitment() {
    let config = config();
    let backend = FriRecursionBackend::<16, 8, _>::new(PERM).for_extension_degree::<4>();
    let a = block_values(10);
    let b = block_values(50);
    let a_proof = prove_block(&config, &a);
    let b_proof = prove_block(&config, &b);

    let mut builder = CircuitBuilder::<EF>::new();
    <Backend as PcsRecursionBackend<C, BlockAir, 4>>::prepare_circuit(
        &backend,
        &config,
        &mut builder,
    )
    .expect("enable verifier ops");
    let block: Checked = <Backend as PcsRecursionBackend<C, BlockAir, 4>>::build_verifier_circuit(
        &backend,
        &uni_input(&a_proof, &a),
        &config,
        &mut builder,
    )
    .expect("verify block child in circuit");
    assert!(block.preprocessed_commitment().is_none());
    assert!(block.air_public_targets(1).is_none());
    let block_targets = block
        .air_public_targets(0)
        .expect("uni child exposes instance 0")
        .to_vec();
    assert_eq!(block_targets.len(), WINDOW);
    tag_all(&mut builder, "block", &block_targets);
    let exports = block_targets
        .iter()
        .map(|&target| StatementExport::Base(target))
        .collect::<Vec<_>>();
    let schema = builder
        .set_statement_exports::<F>(&exports)
        .expect("export block window");
    let wrap = prepare_shape(
        &config,
        &backend,
        builder.build().expect("build wrap"),
        &schema,
    );

    let prove_wrap = |proof: &Proof<C>, values: &[F]| -> BatchStarkProof<C> {
        let prev = uni_input(proof, values);
        let mut runner = wrap.circuit.runner();
        runner
            .set_public_inputs(&block.pack_public_inputs(&prev).expect("pack public"))
            .expect("set public");
        runner
            .set_private_inputs(&block.pack_private_inputs(&prev).expect("pack private"))
            .expect("set private");
        <Backend as PcsRecursionBackend<C, BlockAir, 4>>::set_private_data_for_result(
            &backend,
            &config,
            &mut runner,
            &block,
            &prev,
        )
        .expect("block FRI private data");
        let traces = runner.run().expect("run wrap");
        assert_eq!(probe_all(&traces, "block", WINDOW), lift(values));
        let proof = wrap.prover.prove(&traces).expect("prove wrap");
        wrap.verifier.verify(&proof, values).expect("wrap verifies");
        proof
    };
    let wrap_a = prove_wrap(&a_proof, &a);
    let wrap_b = prove_wrap(&b_proof, &b);

    let key = native_key(&wrap.verifier);
    assert_eq!(key.len(), KEY_LEN);

    let mut builder = CircuitBuilder::<EF>::new();
    <Backend as PcsRecursionBackend<C, BatchOnly, 4>>::prepare_circuit(
        &backend,
        &config,
        &mut builder,
    )
    .expect("enable verifier ops");
    let head: Checked =
        <Backend as TrustedPcsRecursionBackend<C, BatchOnly, 4>>::build_trusted_batch_verifier_circuit(
            &backend,
            &wrap.verifier,
            &wrap_a,
            &a,
            &mut builder,
        )
        .expect("verify head child in circuit");
    let key_targets = head
        .preprocessed_commitment()
        .expect("head child has a preprocessed commitment")
        .to_observation_targets();
    assert_eq!(key_targets.len(), KEY_LEN);
    let statement_table = wrap
        .verifier
        .statement_layout()
        .table_instance()
        .expect("head exports a statement");
    let statement_targets = head
        .air_public_targets(statement_table)
        .expect("statement table instance")
        .to_vec();
    assert_eq!(statement_targets.len(), WINDOW);
    tag_all(&mut builder, "key", &key_targets);
    tag_all(&mut builder, "statement", &statement_targets);
    let exports = statement_targets
        .iter()
        .chain(&key_targets)
        .map(|&target| StatementExport::Base(target))
        .collect::<Vec<_>>();
    let schema = builder
        .set_statement_exports::<F>(&exports)
        .expect("export child statement and key");
    let parent = prepare_shape(
        &config,
        &backend,
        builder.build().expect("build parent"),
        &schema,
    );

    let prove_parent = |proof: &BatchStarkProof<C>, statement: &[F]| -> BatchStarkProof<C> {
        wrap.verifier
            .verify(proof, statement)
            .expect("native head verify");
        let prev = batch_input(&wrap.verifier, proof, statement);
        let mut runner = parent.circuit.runner();
        runner
            .set_public_inputs(&head.pack_public_inputs(&prev).expect("pack public"))
            .expect("set public");
        runner
            .set_private_inputs(&head.pack_private_inputs(&prev).expect("pack private"))
            .expect("set private");
        <Backend as TrustedPcsRecursionBackend<C, BatchOnly, 4>>::set_private_data_for_trusted_batch(
            &backend,
            &wrap.verifier,
            proof,
            statement,
            &mut runner,
            <Checked as VerifierCircuitResult<C, BatchOnly>>::op_ids(&head),
        )
        .expect("head FRI private data");
        let traces = runner.run().expect("run parent");
        assert_eq!(probe_all(&traces, "key", KEY_LEN), lift(&key));
        assert_eq!(probe_all(&traces, "statement", WINDOW), lift(statement));
        parent.prover.prove(&traces).expect("prove parent")
    };
    let parent_a = prove_parent(&wrap_a, &a);
    let parent_b = prove_parent(&wrap_b, &b);

    let expected_a = [a.as_slice(), key.as_slice()].concat();
    let expected_b = [b.as_slice(), key.as_slice()].concat();
    parent
        .verifier
        .verify(&parent_a, &expected_a)
        .expect("parent A exports the child statement and key");
    parent
        .verifier
        .verify(&parent_b, &expected_b)
        .expect("parent B exports the child statement and key");
    for limb in 0..expected_a.len() {
        let mut tampered = expected_a.clone();
        tampered[limb] += F::ONE;
        assert!(
            parent.verifier.verify(&parent_a, &tampered).is_err(),
            "exported limb {limb} is not bound"
        );
    }
    assert_ne!(native_key(&parent.verifier), key);
}

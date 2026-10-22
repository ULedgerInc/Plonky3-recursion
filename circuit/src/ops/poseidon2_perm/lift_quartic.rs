use p3_field::BasedVectorSpace;
use p3_field::extension::{BinomialExtensionField, BinomiallyExtendable};
use p3_symmetric::Permutation;

/// Runs a base-field permutation on the constant coefficient of each quartic lane and re-embeds
/// the result; high limbs are ignored here because the D=1 AIR binds them to zero on the bus.
#[derive(Clone, Debug)]
pub struct LiftPermToQuartic<P> {
    perm: P,
}

impl<P> LiftPermToQuartic<P> {
    pub const fn new(perm: P) -> Self {
        Self { perm }
    }
}

impl<F, P, const W: usize> Permutation<[BinomialExtensionField<F, 4>; W]> for LiftPermToQuartic<P>
where
    F: BinomiallyExtendable<4>,
    P: Permutation<[F; W]>,
{
    fn permute(
        &self,
        input: [BinomialExtensionField<F, 4>; W],
    ) -> [BinomialExtensionField<F, 4>; W] {
        let bases = core::array::from_fn(|i| input[i].as_basis_coefficients_slice()[0]);
        let output = self.perm.permute(bases);
        core::array::from_fn(|i| BinomialExtensionField::from(output[i]))
    }
}

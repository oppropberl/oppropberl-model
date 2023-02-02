package units.solvers.backend.z3smt.encoder;

import backend.z3smt.Z3SmtFormatTranslator;
import backend.z3smt.encoder.Z3SmtAbstractConstraintEncoder;

import checkers.inference.model.CombVariableSlot;
import checkers.inference.model.ConstantSlot;
import checkers.inference.model.VariableSlot;
import checkers.inference.solver.backend.encoder.combine.CombineConstraintEncoder;
import checkers.inference.solver.frontend.Lattice;

import com.microsoft.z3.BoolExpr;
import com.microsoft.z3.Context;

import org.checkerframework.javacutil.AnnotationUtils;

import units.representation.TypecheckUnit;
import units.representation.UnitsRepresentationUtils;
import units.solvers.backend.z3smt.representation.Z3InferenceUnit;

public class UnitsZ3SmtCombineConstraintEncoder
        extends Z3SmtAbstractConstraintEncoder<Z3InferenceUnit, TypecheckUnit>
        implements CombineConstraintEncoder<BoolExpr> {

    UnitsRepresentationUtils unitsRepUtils;

    public UnitsZ3SmtCombineConstraintEncoder(
            Lattice lattice,
            Context ctx,
            Z3SmtFormatTranslator<Z3InferenceUnit, TypecheckUnit> z3SmtFormatTranslator) {
        super(lattice, ctx, z3SmtFormatTranslator);
        unitsRepUtils = UnitsRepresentationUtils.getInstance();
    }

    @Override
    public BoolExpr encodeVariable_Variable(
            VariableSlot target, VariableSlot declared, CombVariableSlot result) {
        Z3InferenceUnit decl = declared.serialize(z3SmtFormatTranslator);
        Z3InferenceUnit res = result.serialize(z3SmtFormatTranslator);
        return UnitsZ3SmtEncoderUtils.equality(ctx, decl, res);
    }

    @Override
    public BoolExpr encodeVariable_Constant(
            VariableSlot target, ConstantSlot declared, CombVariableSlot result) {
        Z3InferenceUnit tar = target.serialize(z3SmtFormatTranslator);
        Z3InferenceUnit decl = declared.serialize(z3SmtFormatTranslator);
        Z3InferenceUnit res = result.serialize(z3SmtFormatTranslator);
        if (AnnotationUtils.areSame(declared.getValue(), unitsRepUtils.RECEIVER_DEPENDANT_UNIT)) {
            return UnitsZ3SmtEncoderUtils.equality(ctx, tar, res);
        }
        return UnitsZ3SmtEncoderUtils.equality(ctx, decl, res);
    }

    @Override
    public BoolExpr encodeConstant_Variable(
            ConstantSlot target, VariableSlot declared, CombVariableSlot result) {
        Z3InferenceUnit decl = declared.serialize(z3SmtFormatTranslator);
        Z3InferenceUnit res = result.serialize(z3SmtFormatTranslator);
        return UnitsZ3SmtEncoderUtils.equality(ctx, decl, res);
    }

    @Override
    public BoolExpr encodeConstant_Constant(
            ConstantSlot target, ConstantSlot declared, CombVariableSlot result) {
        Z3InferenceUnit tar = target.serialize(z3SmtFormatTranslator);
        Z3InferenceUnit decl = declared.serialize(z3SmtFormatTranslator);
        Z3InferenceUnit res = result.serialize(z3SmtFormatTranslator);
        if (AnnotationUtils.areSame(declared.getValue(), unitsRepUtils.RECEIVER_DEPENDANT_UNIT)) {
            return UnitsZ3SmtEncoderUtils.equality(ctx, tar, res);
        }
        return UnitsZ3SmtEncoderUtils.equality(ctx, decl, res);
    }
}

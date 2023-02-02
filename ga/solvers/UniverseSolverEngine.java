package universe.solvers;

import checkers.inference.solver.SolverEngine;
import checkers.inference.solver.backend.SolverFactory;
import checkers.inference.solver.backend.maxsat.MaxSatFormatTranslator;
import checkers.inference.solver.backend.maxsat.MaxSatSolverFactory;
import checkers.inference.solver.frontend.Lattice;
import org.checkerframework.javacutil.BugInCF;

import universe.solvers.z3smt.UniverseZ3SmtSolverFactory;

public class UniverseSolverEngine extends SolverEngine {
    @Override
    protected SolverFactory createSolverFactory() {
        if (solverName.contentEquals("Z3smt")) { // TODO: add solverName arg
            return new UniverseZ3SmtSolverFactory();
        } else if (solverName.contentEquals("MaxSAT")) {
            return new MaxSatSolverFactory(){
                @Override
                public MaxSatFormatTranslator createFormatTranslator(Lattice lattice) {
                    return new UniverseFormatTranslator(lattice);
                }
            };
        } else {
            throw new BugInCF(
                    "A back end solver (Z3smt, MaxSAT) must be supplied in solverArgs: solver=Z3smt");
        }

    }
}

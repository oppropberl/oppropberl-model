package universe.solver;

import checkers.inference.model.Constraint;
import checkers.inference.model.Slot;
import checkers.inference.solver.SolverEngine;
import checkers.inference.solver.backend.Solver;
import checkers.inference.solver.backend.SolverFactory;
import checkers.inference.solver.backend.geneticmaxsat.GeneticMaxSatSolver;
import checkers.inference.solver.backend.geneticmaxsat.GeneticMaxSatSolverFactory;
import checkers.inference.solver.backend.maxsat.MaxSatFormatTranslator;
import checkers.inference.solver.backend.maxsat.MaxSatSolverFactory;
import checkers.inference.solver.frontend.Lattice;
import checkers.inference.solver.util.SolverEnvironment;
import org.sat4j.maxsat.WeightedMaxSatDecorator;
import org.sat4j.maxsat.reader.WDimacsReader;
import org.sat4j.pb.IPBSolver;
import org.sat4j.reader.ParseFormatException;
import org.sat4j.specs.ContradictionException;
import org.sat4j.specs.TimeoutException;

import javax.lang.model.element.AnnotationMirror;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

public class UniverseSolverEngine extends SolverEngine {
    @Override
    protected SolverFactory createSolverFactory() {
        return new GeneticMaxSatSolverFactory() {
            @Override
            public Solver<?> createSolver(SolverEnvironment solverEnvironment, Collection<Slot> slots,
                                          Collection<Constraint> constraints, Lattice lattice) {
                MaxSatFormatTranslator formatTranslator = createFormatTranslator(lattice);
                return new GeneticMaxSatSolver(solverEnvironment, slots, constraints, formatTranslator, lattice){
                    // fitness & fit
                };
            }

            @Override
            public MaxSatFormatTranslator createFormatTranslator(Lattice lattice) {
                return new UniverseFormatTranslator(lattice);
            }
        };
    }
}

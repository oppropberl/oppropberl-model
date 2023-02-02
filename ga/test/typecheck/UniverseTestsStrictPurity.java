package universe;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.checkerframework.framework.test.CheckerFrameworkPerFileTest;
import org.checkerframework.framework.test.TestUtilities;
import org.junit.runners.Parameterized.Parameters;

public class UniverseTestsStrictPurity extends CheckerFrameworkPerFileTest {
    public UniverseTestsStrictPurity(File testFile) {
        super(testFile, UniverseInferenceChecker.class, "",
                "-Anomsgtext", "-Alint=checkStrictPurity", "-d", "testTmp");
    }

    @Parameters
    public static List<File> getTestFiles(){
        List<File> testfiles = new ArrayList<>();
        testfiles.addAll(TestUtilities.findRelativeNestedJavaFiles("tests", "typecheck/strictpurity"));
        return testfiles;
    }
}

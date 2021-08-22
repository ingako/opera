package moa.classifiers.transfer;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.GaussianNumericAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NominalAttributeClassObserver;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.classifiers.meta.AdaptiveRandomForest;
import moa.classifiers.patching.Patching;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.options.ClassOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.options.WEKAClassOption;
import weka.classifiers.Classifier;

import java.util.ArrayDeque;
import java.util.Random;

public class TransferFramework extends AbstractClassifier implements MultiClassClassifier, CapabilitiesHandler {

    private static final long serialVersionUID = 1L;

    public ClassOption patchingClassifierOption = new ClassOption("patchingClassifierOption", 'p',
            "Patching classifier options", Patching.class, "Patching");

    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
            "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-3");

    public ClassOption phantomTreeOption = new ClassOption("phantomTree", 't',
            "Phantom Tree for measuring construction complexity", PhantomTree.class, "PhantomTree");

    public FloatOption convDeltaOption = new FloatOption("convDelta", 'a',
            "The confidence value for computing true error during the observation period", 0.1, 0.0, 1.0);

    public FloatOption convThresholdOption = new FloatOption("convThreshold", 'b',
            "The convergence threshold for true error during the observation period", 0.1, 0.0, 1.0);

    public IntOption windowSizeOption = new IntOption("windowSize", 'n',
            "The number of instances to observe for testing convergence.",
            50, 0, Integer.MAX_VALUE);

    public FlagOption disablePatchingOption = new FlagOption("disablePatching", 'e', "Disable patching as a benchmark");

    public FlagOption forceEnablePatchingOption = new FlagOption("forceEnablePatching", 'x', "Force enable patching as a benchmark");

    protected AutoExpandVector<Patching> classifierRepo;
    protected Patching classifier;
    protected ChangeDetector driftDetectionMethod;
    protected AutoExpandVector<Instance> obsInstanceStore;
    protected AutoExpandVector<Instance> errorRegionInstanceStore;
    protected TrueError trueError;

    public boolean isRandomizable() {
        return true;
    }

    @Override
    public double[] getVotesForInstance(Instance samoaInstance) {
        return this.classifier.getVotesForInstance(samoaInstance);
    }

    @Override
    public void resetLearningImpl() {
        this.classifierRepo = new AutoExpandVector<>();
        this.classifier= null;
        this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
        this.obsInstanceStore = null;
        this.errorRegionInstanceStore = null;
        this.trueError = null;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (this.classifier == null) {
            this.classifier = (Patching) getPreparedClassOption(this.patchingClassifierOption);
        }

        int errorCount = this.classifier.correctlyClassifies(inst) ? 0 : 1;

        this.driftDetectionMethod.input(errorCount);
        if (this.driftDetectionMethod.getChange()) {
            this.driftDetectionMethod.resetLearning();

            if (this.disablePatchingOption.isSet()) {
                this.classifier = (Patching) getPreparedClassOption(this.patchingClassifierOption);

            } else {
                this.classifierRepo.add(this.classifier);
                this.obsInstanceStore = new AutoExpandVector<>();
                this.trueError = new TrueError(
                        this.windowSizeOption.getValue(),
                        this.convDeltaOption.getValue(),
                        this.convThresholdOption.getValue(),
                        this.classifierRandom);

                if (this.classifierRepo.size() == 1) {
                    // classifier is the transferred model

                } else {
                    // TODO
                    throw new NullPointerException("Not supporting cross stream transfer yet.");
                }
            }
        }

        this.classifier.trainOnInstanceImpl(inst);

        if (this.obsInstanceStore != null) {
            this.obsInstanceStore.add(inst);
            if (errorCount == 1) {
                this.errorRegionInstanceStore.add(inst);
            }

            // weka.classifiers.trees.RandomForest randomForest = (weka.classifiers.trees.RandomForest) this.baseClassifier;
            // randomForest.getMembershipValues();

            if (this.trueError.isStable(errorCount)) {
                PhantomTree phantomTree = (PhantomTree) getPreparedClassOption(this.phantomTreeOption);
                PhantomTree regionalPhantomTree = (PhantomTree) getPreparedClassOption(this.phantomTreeOption);

                double regionalComplexity = regionalPhantomTree.getConstructionComplexity(errorRegionInstanceStore);
                double complexity = phantomTree.getConstructionComplexity(obsInstanceStore);
                System.out.println("regional=" + regionalComplexity + " | complexity=" + complexity);
                if (regionalComplexity < complexity) {
                    this.classifier.setEnablePatching(true);
                } else {
                    this.classifier = (Patching) getPreparedClassOption(this.patchingClassifierOption);
                    for (Instance obsInstance : obsInstanceStore) {
                        this.classifier.trainOnInstanceImpl(obsInstance);
                    }
                }

                this.obsInstanceStore = null;
                this.errorRegionInstanceStore = null;
            }
        }

    }

    class TrueError {
        int sampleSize;
        int windowSize;
        double rc;
        double errorCount;
        double delta;
        double convThreshold;
        double windowSum;
        Random classifierRandom;

        ArrayDeque<Double> window;

        public TrueError(
                int windowSize,
                double delta,
                double convThreshold,
                Random classifierRandom) {

            this.sampleSize = 0;
            this.windowSize = windowSize;
            this.rc = 0;
            this.errorCount = 0;
            this.delta = delta;
            this.convThreshold = convThreshold;
            this.windowSum = 0;
            this.classifierRandom = classifierRandom;
        }

        private boolean isStable(int error) {
            if (this.window.size() == this.windowSize) {
                double val = this.window.pop();
                this.windowSum -= val;
            }
            double trueError = getTrueError(error);
            this.window.add(trueError);
            this.windowSum += trueError;

            double mean = this.windowSum / this.window.size();
            double numerator = 0;
            for (double err : window) {
                numerator += err - mean;
            }

            if (numerator / (this.window.size() - 1) <= this.convThreshold) {
                return true;
            }

            return false;
        }

        public double getTrueError(int error) {
            this.errorCount += error;
            this.sampleSize++;

            int sigma = -1;
            if (this.classifierRandom.nextBoolean()) {
                sigma = 1;
            }
            this.rc += sigma * error;

            double risk = this.errorCount / this.sampleSize;
            this.rc /= this.sampleSize;

            // true error based on Rademacher bound
            double trueErrorBound = risk + 2*this.rc + 3*Math.sqrt(Math.log(2/this.delta) / (2*sampleSize));
            return trueErrorBound;
        }
    }

    protected AttributeClassObserver newNominalClassObserver() {
        return new NominalAttributeClassObserver();
    }

    protected AttributeClassObserver newNumericClassObserver() {
        return new GaussianNumericAttributeClassObserver();
    }

    public void getModelDescription(StringBuilder out, int indent) {
    }

    protected moa.core.Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == TransferFramework.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }

}
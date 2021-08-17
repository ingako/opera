package moa.classifiers.transfer;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
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

public class TransferFramework extends AbstractClassifier implements MultiClassClassifier, CapabilitiesHandler {

    private static final long serialVersionUID = 1L;

    public IntOption windowSizeOption = new IntOption("windowSize", 'w',
            "The number of instances to observe for testing convergence.",
            50, 0, Integer.MAX_VALUE);

    public WEKAClassOption baseClassifierOption = new WEKAClassOption("baseClassifier", 'l',
            "WEKA class to use for the base classifier.", weka.classifiers.Classifier.class, "weka.classifiers.trees.RandomForest");

    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
            "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-3");

    public ClassOption warningDetectionMethodOption = new ClassOption("warningDetectionMethod", 'p',
            "Change detector for warnings (start training bkg learner)", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-2");

    public ClassOption phantomTreeOption = new ClassOption("phantomTree", 'p',
            "Phantom Tree for measuring construction complexity", PhantomTree.class, "PhantomTree");

    protected AutoExpandVector<AdaptiveRandomForest> sourceRepo;
    protected Classifier baseClassifier;
    protected Patching patching;
    protected ChangeDetector driftDetectionMethod;
    protected ChangeDetector warningDetectionMethod;
    protected AutoExpandVector<Instance> obsInstanceStore;

    public boolean isRandomizable() {
        return true;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return new double[0];
    }

    @Override
    public void resetLearningImpl() {
        this.obsInstanceStore = null;
        this.baseClassifier = getBaseClassifier();
        this.sourceRepo = new AutoExpandVector<>();
        this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (patching != null) {
            patching.trainOnInstanceImpl(inst);
            return;
        }

        int errorCount = this.baseClassifier.getClass() == inst.getClass() ? 0 : 1;
        driftDetectionMethod.input(errorCount);
        if (driftDetectionMethod.getChange()) {
            this.obsInstanceStore = new AutoExpandVector<>();
        }

        if (isStable(errorCount)) {
            PhantomTree phantomTree = (PhantomTree) getPreparedClassOption(this.phantomTreeOption);
        }
    }

    private boolean isStable(int errorCount) {
        // if (this.trainingWeightSeenByModel == this.windowSizeOption.getValue()) {
        // }

        int sigma = -1;
        if (this.classifierRandom.nextBoolean()) {
            sigma = 1;
        }
        return false;
    }

    private Classifier getBaseClassifier() {
        try {
            String[] options = weka.core.Utils.splitOptions(baseClassifierOption.getValueAsCLIString());
            String classifierName = options[0];
            String[] newoptions = options.clone();
            newoptions[0] = "";
            Classifier classifier = weka.classifiers.AbstractClassifier.forName(classifierName, newoptions);

            return classifier;

        } catch (Exception e) {
            System.err.println("Error retrieving selected classifier:");
            System.err.println("Chosen classifier: " + this.baseClassifierOption.getValueAsCLIString());
            System.err.println(e.getMessage());
        }

        return null;
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
package moa.classifiers.transfer;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.GaussianNumericAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NominalAttributeClassObserver;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.meta.OzaBag;
import moa.classifiers.meta.OzaBoost;
import moa.core.AutoExpandVector;
import moa.core.Utils;
import moa.options.ClassOption;
import com.yahoo.labs.samoa.instances.Instance;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Random;

public class TransferFramework extends AbstractClassifier implements MultiClassClassifier, CapabilitiesHandler {
    @Override
    public String getPurposeString() { return "Transfer Framework"; }

    private static final long serialVersionUID = 1L;

    public ClassOption baseClassifierOption = new ClassOption("baseClassifier", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
            "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-3");

    public ClassOption phantomTreeOption = new ClassOption("phantomTree", 't',
            "Phantom Tree for measuring construction complexity", PhantomTree.class, "PhantomTree");

    public FloatOption convDeltaOption = new FloatOption("convDelta", 'a',
            "The confidence value for computing true error during the observation period", 0.1, 0.0, 1.0);

    public FloatOption convThresholdOption = new FloatOption("convThreshold", 'b',
            "The convergence threshold for true error during the observation period", 0.15, 0.0, 1.0);

    public IntOption windowSizeOption = new IntOption("windowSize", 'n',
            "The number of instances to observe for testing convergence.",
            50, 0, Integer.MAX_VALUE);

    public FlagOption forceDisableTransferOption = new FlagOption("froceDisableTransfer", 'e', "Force disable transfer");

    public FlagOption forceEnableTransferOption = new FlagOption("forceEnableTransfer", 'x', "Force enable transfer");

    protected AutoExpandVector<Classifier> classifierRepo;
    protected Classifier classifier;
    protected ChangeDetector driftDetectionMethod;
    protected ArrayList<Instance> obsInstanceStore;
    protected ArrayList<Integer> obsPredictionResults;
    protected ArrayList<Instance> errorRegionInstanceStore;
    protected ArrayDeque<Instance> aproposRegionInstanceStore;
    protected TrueError trueError;

    protected Classifier errorRegionClassifier;
    protected Classifier patchClassifier;

    int patchCount = 0;
    int classifierCount = 0;

    protected Classifier emptyClassifier;

    public boolean isRandomizable() { return true; }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (this.patchClassifier != null) {
            Instance newInstance = inst.copy();
            newInstance.insertAttributeAt(0);
            newInstance.setValue(0, inst.classValue());
            if (Utils.maxIndex(this.errorRegionClassifier.getVotesForInstance(newInstance)) == 1) {
                patchCount++;
                System.out.println("patching="+patchCount+"|classifierCount="+classifierCount);
                return this.patchClassifier.getVotesForInstance(inst);
            } else {
                classifierCount++;
                System.out.println("patching="+patchCount+"|classifierCount="+classifierCount);
            }
        }

        return this.classifier.getVotesForInstance(inst);
    }

    @Override
    public void resetLearningImpl() {
        if (this.classifier == null) {
            this.classifier = (Classifier) getPreparedClassOption(this.baseClassifierOption);
            this.emptyClassifier = this.classifier.copy();
        }
        this.classifier.resetLearning();
        this.classifierRepo = new AutoExpandVector<>();
        this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
        this.obsInstanceStore = null;
        this.obsPredictionResults = null;
        this.errorRegionInstanceStore = null;
        this.aproposRegionInstanceStore = null;
        this.trueError = null;

        this.errorRegionClassifier = null;
        this.patchClassifier = null;

        this.patchCount = 0;
        this.classifierCount = 0;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (this.classifier == null) {
            this.classifier = (Classifier) getPreparedClassOption(this.baseClassifierOption);
        }

        int errorCount = this.classifier.correctlyClassifies(inst) ? 0 : 1;

        this.driftDetectionMethod.input(errorCount);
        if (this.driftDetectionMethod.getChange()) {
            this.driftDetectionMethod.resetLearning();

            this.classifierRepo.add(this.classifier);
            this.obsInstanceStore = new ArrayList<>();
            this.obsPredictionResults = new ArrayList<>();
            this.errorRegionInstanceStore = new ArrayList<>();
            this.aproposRegionInstanceStore= new ArrayDeque<>();

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

        if (this.patchClassifier != null) {
            // train the patch or the transferred model based on if instance is in error region
            Instance newInstance = inst.copy();
            newInstance.insertAttributeAt(0);
            newInstance.setValue(0, inst.classValue());
            if (errorCount == 1) {
                newInstance.setClassValue(1);
                this.patchClassifier.trainOnInstance(inst);
            } else {
                newInstance.setClassValue(0);
                // this.classifier.trainOnInstance(inst);
            }
            this.errorRegionClassifier.trainOnInstance(newInstance);

        } else if (this.obsInstanceStore == null) {
            this.classifier.trainOnInstance(inst);

        } else {
            this.obsInstanceStore.add(inst);
            this.obsPredictionResults.add(errorCount);
            if (errorCount == 1) {
                this.errorRegionInstanceStore.add(inst);
            }

            if (this.trueError.isStable(errorCount)) {
                this.driftDetectionMethod.resetLearning();
                System.out.println("instance store size: " + obsInstanceStore.size());

                boolean enableTransfer = false;
                if (this.forceDisableTransferOption.isSet()) {

                } else if (this.forceEnableTransferOption.isSet()) {
                    enableTransfer = true;

                } else {
                    PhantomTree phantomTree = (PhantomTree) getPreparedClassOption(this.phantomTreeOption);
                    PhantomTree regionalPhantomTree = (PhantomTree) phantomTree.copy();

                    double regionalComplexity = regionalPhantomTree.getConstructionComplexity(errorRegionInstanceStore);
                    double complexity = phantomTree.getConstructionComplexity(obsInstanceStore);
                    System.out.println("regional=" + regionalComplexity + " | full=" + complexity);
                    if (regionalComplexity < complexity) {
                        enableTransfer = true;
                    }
                }

                if (enableTransfer) {
                    this.errorRegionClassifier = this.emptyClassifier.copy();
                    this.patchClassifier = this.emptyClassifier.copy();

                    for (int idx = 0; idx < this.obsInstanceStore.size(); idx++) {
                        Instance obsInstance = this.obsInstanceStore.get(idx);
                        Instance newInstance = obsInstance.copy();
                        newInstance.insertAttributeAt(0);
                        newInstance.setValue(0, obsInstance.classValue());

                        this.errorRegionClassifier.trainOnInstance(newInstance);

                        if (this.obsPredictionResults.get(idx) == 1) {
                            // this.patchClassifier.trainOnInstance(newInstance);
                            this.patchClassifier.trainOnInstance(obsInstance);
                        } else  {
                            this.classifier.trainOnInstance(obsInstance);
                        }
                    }

                } else {
                    // this.classifier.resetLearning();
                    this.classifier = emptyClassifier.copy();
                    for (Instance obsInstance : this.obsInstanceStore) {
                        this.classifier.trainOnInstance(obsInstance);
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
            this.window = new ArrayDeque<>();
        }

        private boolean isStable(int error) {
            double trueError = getTrueError(error);
            this.window.add(trueError);
            this.windowSum += trueError;

            if (this.window.size() < this.windowSize) {
                return false;
            }

            if (this.window.size() > this.windowSize) {
                double val = this.window.pop();
                this.windowSum -= val;
            }

            double mean = this.windowSum / this.window.size();
            double numerator = 0;
            for (double err : window) {
                numerator += Math.sqrt(Math.abs(err - mean));
            }

            double convVal = Math.sqrt(numerator / (this.window.size() - 1));
           //  System.out.println(convVal);
            if (convVal <= this.convThreshold) {
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
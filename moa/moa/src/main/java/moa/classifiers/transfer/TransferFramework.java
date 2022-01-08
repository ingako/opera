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
import moa.core.AutoExpandVector;
import moa.core.Measurement;
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

    public IntOption obsWindowSizeOption = new IntOption("obsWindowSize", 'n',
            "The number of instances to observe for testing convergence.",
            50, 0, Integer.MAX_VALUE);

    public IntOption perfWindowSizeOption = new IntOption("perfWindowSize", 'o',
            "The number of instances to observe for assessing the performance of background classifier.",
            500, 0, Integer.MAX_VALUE);

    public IntOption driftLocationOption = new IntOption("driftLocation", 'f',
            "-1 turns on drift detector",
            -1, -1, Integer.MAX_VALUE);

    public IntOption minObsPeriod = new IntOption("minObsPeriod", 'm',
            "The minimum number of instances to observe before testing convergence. -1 to disable.",
            -1, -1, Integer.MAX_VALUE);

    public FlagOption disablePatchingOption = new FlagOption("disablePatching", 'x', "Force disable patching");

    protected AutoExpandVector<Classifier> classifierRepo;
    protected Classifier classifier;
    protected ChangeDetector driftDetectionMethod;
    protected ArrayList<Instance> obsInstanceStore;
    protected ArrayList<Integer> obsPredictionResults;
    protected ArrayList<Instance> errorRegionInstanceStore;
    protected ArrayList<Instance> aproposRegionInstanceStore;
    protected TrueError trueError;

    protected Classifier errorRegionClassifier;
    protected Classifier patchClassifier;

    int patchCount;
    int classifierCount;
    int maxObsPeriodLen;
    int maxErrRegionStoreSize;
    int maxAprRegionStoreSize;

    protected Classifier emptyClassifier;

    InstanceStoreComplexity obsInstanceStoreComplexity;
    InstanceStoreComplexity errorInstanceStoreComplexity;
    InstanceStoreComplexity aproposInstanceStoreComplexity;

    // track performances for both the patch learner and the transferred model
    protected ArrayDeque<Integer> patchErrorWindow;
    protected ArrayDeque<Integer> transErrorWindow;
    protected double patchErrorWindowSum;
    protected double transErrorWindowSum;

    public boolean isRandomizable() { return true; }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (this.patchClassifier == null) {
            return this.classifier.getVotesForInstance(inst);
        }

        Instance newInstance = inst.copy();
        newInstance.insertAttributeAt(0);
        newInstance.setValue(0, inst.classValue());
        if (Utils.maxIndex(this.errorRegionClassifier.getVotesForInstance(newInstance)) == 1) {
            // in error region, check patch performance
            if (turnOnPatchPrediction()) {
                this.patchCount++;
                return this.patchClassifier.getVotesForInstance(inst);
            }

        }

        this.classifierCount++;
        return this.classifier.getVotesForInstance(inst);
    }

    private boolean turnOnPatchPrediction() {
        if (this.patchErrorWindow.size() < this.perfWindowSizeOption.getValue()) {
            return true;
        }

        if (this.patchErrorWindowSum < this.transErrorWindowSum) {
            return true;
        }

        return false;
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
        this.maxObsPeriodLen = 0;
        this.maxErrRegionStoreSize = 0;
        this.maxAprRegionStoreSize = 0;

        this.obsInstanceStoreComplexity = new InstanceStoreComplexity();
        this.errorInstanceStoreComplexity = new InstanceStoreComplexity();
        this.aproposInstanceStoreComplexity = new InstanceStoreComplexity();

        // patch related
        this.patchErrorWindow = new ArrayDeque<>();
        this.transErrorWindow = new ArrayDeque<>();
        this.patchErrorWindowSum = 0;
        this.transErrorWindowSum = 0;

    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (this.classifier == null) {
            this.classifier = (Classifier) getPreparedClassOption(this.baseClassifierOption);
        }

        int errorCount = this.classifier.correctlyClassifies(inst) ? 0 : 1;
        handleDrift(errorCount);

        if (this.patchClassifier != null) {
            // train transferred model
            this.classifier.trainOnInstance(inst);
            // update transferred model performance
            if (this.transErrorWindow.size() > this.perfWindowSizeOption.getValue()){
                this.transErrorWindowSum -= this.transErrorWindow.pollFirst();
            }
            this.transErrorWindow.offerLast(errorCount);
            this.transErrorWindowSum += errorCount;

            // train patch
            int patchErrorCount = 0;
            if (errorCount == 1) {
                // keep track of patch to either turn on/off patch prediction
                if (!this.patchClassifier.correctlyClassifies(inst)) {
                    patchErrorCount = 1;
                }

                this.patchClassifier.trainOnInstance(inst);
            }

            // update patch performance
            if (this.patchErrorWindow.size() > this.perfWindowSizeOption.getValue()){
                this.patchErrorWindowSum -= this.patchErrorWindow.pollFirst();
            }
            this.patchErrorWindow.offerLast(patchErrorCount);
            this.patchErrorWindowSum += patchErrorCount;

        } else if (this.obsInstanceStore == null) {
            // either from source or a new model in target
            this.classifier.trainOnInstance(inst);

        } else {
            this.obsInstanceStore.add(inst);
            this.maxObsPeriodLen = Math.max(this.maxObsPeriodLen, this.obsInstanceStore.size());

            this.obsPredictionResults.add(errorCount);
            if (errorCount == 1) {
                this.errorRegionInstanceStore.add(inst);
                this.maxErrRegionStoreSize = Math.max(this.maxErrRegionStoreSize, this.errorRegionInstanceStore.size());
            } else {
                this.aproposRegionInstanceStore.add(inst);
                this.maxAprRegionStoreSize = Math.max(this.maxAprRegionStoreSize, this.aproposRegionInstanceStore.size());
            }

            if (!this.trueError.isStable(errorCount)) {
                return;
            }

            if (this.minObsPeriod.getValue() > -1 && this.obsInstanceStore.size() < this.minObsPeriod.getValue()) {
                return;
            }

            this.driftDetectionMethod.resetLearning();
            System.out.println("instance store size: " + obsInstanceStore.size());

            boolean enableTransfer = false;
            if (this.disablePatchingOption.isSet()) {

            } else {
                enableTransfer = true;
                measureComplexities();
            }

            if (enableTransfer) {
                this.errorRegionClassifier = this.emptyClassifier.copy();
                this.patchClassifier = this.emptyClassifier.copy();

                for (int idx = 0; idx < this.obsInstanceStore.size(); idx++) {
                    Instance obsInstance = this.obsInstanceStore.get(idx);
                    Instance newInstance = obsInstance.copy();
                    newInstance.insertAttributeAt(0);
                    newInstance.setValue(0, obsInstance.classValue());

                    this.classifier.trainOnInstance(obsInstance);
                    if (this.obsPredictionResults.get(idx) == 1) {
                        this.patchClassifier.trainOnInstance(obsInstance);
                        newInstance.setClassValue(1);
                    } else  {
                        newInstance.setClassValue(0);
                    }

                    this.errorRegionClassifier.trainOnInstance(newInstance);
                }
            } else {
                this.classifier = this.emptyClassifier.copy();
                for (int idx = 0; idx < this.obsInstanceStore.size(); idx++) {
                    Instance obsInstance = this.obsInstanceStore.get(idx);
                    this.classifier.trainOnInstance(obsInstance);
                }
            }

            this.obsInstanceStore = null;
            this.errorRegionInstanceStore = null;
            this.aproposRegionInstanceStore = null;
        }
    }

    private void handleDrift(int errorCount) {
        if (this.driftLocationOption.getValue() == -1) {
            this.driftDetectionMethod.input(errorCount);
            if (!this.driftDetectionMethod.getChange()) {
                return;
            }
            this.driftDetectionMethod.resetLearning();
        } else {
            if (this.trainingWeightSeenByModel() != this.driftLocationOption.getValue()) {
                return;
            }
        }

        this.classifierRepo.add(this.classifier);
        this.obsInstanceStore = new ArrayList<>();
        this.obsPredictionResults = new ArrayList<>();
        this.errorRegionInstanceStore = new ArrayList<>();
        this.aproposRegionInstanceStore = new ArrayList<>();

        this.trueError = new TrueError(
                this.obsWindowSizeOption.getValue(),
                this.convDeltaOption.getValue(),
                this.convThresholdOption.getValue(),
                this.classifierRandom);

        if (this.classifierRepo.size() == 1) {
            // hack; classifier is the transferred model

        } else {
            // TODO
            throw new NullPointerException("Not supporting cross stream transfer yet.");
        }
        // }
    }

    private void measureComplexities() {
        PhantomTree emptyPhantomTree = (PhantomTree) getPreparedClassOption(this.phantomTreeOption);
        this.obsInstanceStoreComplexity.measure(this.obsInstanceStore, (PhantomTree) emptyPhantomTree.copy());
        this.errorInstanceStoreComplexity.measure(this.errorRegionInstanceStore, (PhantomTree) emptyPhantomTree.copy());
        this.aproposInstanceStoreComplexity.measure(this.aproposRegionInstanceStore, (PhantomTree) emptyPhantomTree.copy());
    }

    class InstanceStoreComplexity {
        long time;
        double avgDepth;
        double minDepth;
        double maxDepth;

        public InstanceStoreComplexity() {
            this.time = -1;
            this.avgDepth = -1;
            this.minDepth = -1;
            this.maxDepth = -1;
        }

        public void measure(ArrayList<Instance> instanceStore, PhantomTree phantomTree) {
            long startTime = System.nanoTime();
            phantomTree.getConstructionComplexity(instanceStore);
            this.time = System.nanoTime() - startTime;

            this.avgDepth = phantomTree.avgPhantomBranchDepth;
            this.minDepth = phantomTree.minPhantomBranchDepth;
            this.maxDepth = phantomTree.maxPhantomBranchDepth;
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
        return new Measurement[]{
                new Measurement("instance store size", this.maxObsPeriodLen),
                new Measurement("error region instance store size",
                        this.maxErrRegionStoreSize),
                new Measurement("apropos region instance store size",
                        this.maxAprRegionStoreSize),

                new Measurement("full region time",
                        this.obsInstanceStoreComplexity.time),
                new Measurement("error region time",
                        this.errorInstanceStoreComplexity.time),
                new Measurement("apropos region time",
                        this.aproposInstanceStoreComplexity.time),

                new Measurement("full region depth avg",
                        this.obsInstanceStoreComplexity.avgDepth),
                new Measurement("error region depth avg",
                        this.errorInstanceStoreComplexity.avgDepth),
                new Measurement("apropos region depth avg",
                        this.aproposInstanceStoreComplexity.avgDepth),

                new Measurement("full region depth min",
                        this.obsInstanceStoreComplexity.minDepth),
                new Measurement("error region depth min",
                        this.errorInstanceStoreComplexity.minDepth),
                new Measurement("apropos region depth min",
                        this.aproposInstanceStoreComplexity.minDepth),

                new Measurement("full region depth max",
                        this.obsInstanceStoreComplexity.maxDepth),
                new Measurement("error region depth max",
                        this.errorInstanceStoreComplexity.maxDepth),
                new Measurement("apropos region depth max",
                        this.aproposInstanceStoreComplexity.maxDepth),

                new Measurement("patch count",
                        this.patchCount),
                new Measurement("base classifier count",
                        this.classifierCount)
        };
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == TransferFramework.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }

}

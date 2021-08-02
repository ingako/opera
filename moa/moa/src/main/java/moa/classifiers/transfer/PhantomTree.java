package moa.classifiers.transfer;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Attribute;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.GaussianNumericAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NominalAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NullAttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalBinaryTest;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.conditionaltests.NominalAttributeBinaryTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.classifiers.patching.Patching;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import com.yahoo.labs.samoa.instances.Instance;
import moa.core.SizeOf;

import java.util.*;

public class PhantomTree extends HoeffdingTree implements MultiClassClassifier, CapabilitiesHandler {

    private static final long serialVersionUID = 1L;

    public IntOption obsPeriodOption = new IntOption("obsPeriod", 'o',
            "The number of instances to observe before growing phantom branches.",
            1000, 0, Integer.MAX_VALUE);

    public IntOption numPhantomBranchOption = new IntOption("numPhantomBranch", 'k',
            "The number of phantom branches to grow.",
            9, 1, Integer.MAX_VALUE);

    ArrayDeque<Instance> instanceStore = new ArrayDeque<>();

    class PhantomBranch {
        int depth;
        double foil_info_gain;
        PhantomNode leaf;
        HashSet<Integer> usedNominalAttributes;
        ArrayDeque<Instance> reachedInstances;

        protected DoubleVector observedClassDistribution;

        public PhantomBranch(PhantomNode leaf, int depth, HashSet<Integer> usedNominalAttributes) {
            this.leaf = leaf;
            this.depth = depth;
            this.usedNominalAttributes = usedNominalAttributes;
            this.reachedInstances = new ArrayDeque<>();
            this.foil_info_gain = -1;
        }
    }

    public static class PhantomNode extends ActiveLearningNode {

        private int postiveCount = 0;
        private int negativeCount = 0;

        public PhantomNode(double[] initialClassObservations) {
            super(initialClassObservations);
            this.weightSeenAtLastSplitEvaluation = getWeightSeen();
            this.isInitialized = false;
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {
            if (this.isInitialized == false) {
                this.attributeObservers = new AutoExpandVector<AttributeClassObserver>(inst.numAttributes());
                this.isInitialized = true;
            }
            this.observedClassDistribution.addToValue((int) inst.classValue(),
                    inst.weight());
            for (int i = 0; i < inst.numAttributes() - 1; i++) {
                int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs == null) {
                    obs = inst.attribute(instAttIndex).isNominal() ? ht.newNominalClassObserver() : ht.newNumericClassObserver();
                    this.attributeObservers.set(i, obs);
                }
                obs.observeAttributeClass(inst.value(instAttIndex), (int) inst.classValue(), inst.weight());
            }
        }

        public AttributeSplitSuggestion[] getBestSplitSuggestions(SplitCriterion criterion) {
            List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<AttributeSplitSuggestion>();
            double[] preSplitDist = this.observedClassDistribution.getArrayCopy();

            for (int i = 0; i < this.attributeObservers.size(); i++) {
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs != null) {
                    AttributeSplitSuggestion bestSuggestion = obs.getBestEvaluatedSplitSuggestion(
                                                                    criterion,
                                                                    preSplitDist,
                                                                    i,
                                                                    true);
                    if (bestSuggestion != null) {
                        bestSuggestions.add(bestSuggestion);
                    }
                }
            }
            return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
        }
    }

    private ArrayDeque<PhantomBranch> growPhantomBranch() {
        // init candidate phantom branches
        addInstancesToLeaves(this.instanceStore);
        ArrayDeque<PhantomBranch> rootPhantomBranches = initPhantomBranches();
        ArrayDeque<PhantomBranch> candidatePhantomBranches = new ArrayDeque<>();
        for (PhantomBranch pb : rootPhantomBranches) {
            for (PhantomBranch candidate: phantomSplit(pb)) {
                candidatePhantomBranches.offer(candidate);
            }
        }

        ArrayDeque<PhantomBranch> weightedRandomBranches = new ArrayDeque<>();
        for (int i = 0; i < numPhantomBranchOption.getValue(); i++) {
            PhantomBranch phantomRoot = getWeightedRandomBranch(candidatePhantomBranches);
            weightedRandomBranches.offer(growPhantomBranch(phantomRoot));
        }

        return weightedRandomBranches;
    }

    private ArrayDeque<PhantomBranch> phantomSplit(PhantomBranch phantomBranches) {
        ArrayDeque<PhantomBranch> phantomChildren = new ArrayDeque<>();
        // ArrayDeque<> splitCandidates = new ArrayDeque<>();
        // for (PhantomBranch phantomBranch : phantomBranches) {
        //     for (Instance inst : phantomBranch.reachedInstances) {
        //         for (int i = 0; i < inst.numAttributes(); i++) {
        //             for (int j = 0; j < inst.numAttributes(); j++) {
        //                 Attribute att = inst.inputAttribute(j);
        //                 if (att.isNumeric()) continue;

        //                 if (phantomBranch.usedAttributeValues.get(i).contains(j)) {
        //                     continue;
        //                 }

        //                 List<String> values= att.getAttributeValues();
        //                 for (int k = 0; k < att.numValues(); j++) {

        //                 }
        //             }

        //         }
        //     }
        // }

        return phantomSplit(phantomBranches);
    }

    private PhantomBranch getWeightedRandomBranch(ArrayDeque<PhantomBranch> phantomBranches) {
        double sum = 0;
        for (PhantomBranch pb : phantomBranches) {
            if (pb.foil_info_gain <= 0) {
                continue;
            }
            sum += pb.foil_info_gain;
        }

        double rand = Math.random();
        double partial_sum = 0;
        for (PhantomBranch pb : phantomBranches) {
            if (pb.foil_info_gain <= 0) {
                continue;
            }
            partial_sum += (pb.foil_info_gain / sum);
            if (partial_sum > rand) {
                return pb;
            }
        }

        return null;
    }

    private PhantomBranch growPhantomBranch(PhantomBranch phantomBranch) {
        if (phantomBranch.depth > 5) {
            return phantomBranch;
        }

        ArrayDeque<PhantomBranch> candidateBranches = new ArrayDeque<>();
        for (int i = 0; i < numPhantomBranchOption.getValue(); i++) {
            ArrayDeque<PhantomBranch> phantomChildren = phantomSplit(phantomBranch);
            while (!phantomChildren.isEmpty()) {
                candidateBranches.offer(phantomChildren.poll());
            }
        }

        return growPhantomBranch(getWeightedRandomBranch(candidateBranches));
    }

    private ArrayDeque<PhantomBranch> initPhantomBranches() {
        ArrayDeque<Node> nodes = new ArrayDeque<>();
        ArrayDeque<HashSet<Integer>> usedNominalAttributes = new ArrayDeque<>();
        ArrayDeque<PhantomBranch> phantomBranches = new ArrayDeque<>();
        nodes.offer(super.treeRoot);
        usedNominalAttributes.offer(new HashSet<>());
        int depth = 0;

        while (!nodes.isEmpty()) {
            Node curNode = nodes.pop();
            depth += 1;

            if (curNode instanceof LearningNode) {
                phantomBranches.offer(new PhantomBranch(new PhantomNode(curNode.getObservedClassDistribution()),
                                                            depth, usedNominalAttributes.pop()));

            } else {
                HashSet<Integer> usedNominalAttributeSet = usedNominalAttributes.pop();

                SplitNode splitNode = (SplitNode) curNode;
                InstanceConditionalTest condition = splitNode.getSplitTest();
                if (condition instanceof NominalAttributeBinaryTest) {
                    usedNominalAttributeSet.add(condition.getAttributeIndex());
                }

                for (int j = 0; j < splitNode.numChildren(); j++) {
                    nodes.offer(splitNode.getChild(j));
                    usedNominalAttributes.offer((HashSet<Integer>) usedNominalAttributeSet.clone());
                }
            }
        }

        return phantomBranches;
    }

    private void addInstancesToLeaves(ArrayDeque<Instance> instanceStore) {
        for (Instance inst : instanceStore) {
            filter(super.treeRoot, inst);
        }
    }

    private void filter(Node node, Instance inst) {
        if (node instanceof LearningNode) {
            node.instanceStore.offer(inst);
            return;
        }
        int childIndex = ((SplitNode) node).instanceChildIndex(inst);
        if (childIndex < 0) return;
        Node child = ((SplitNode) node).getChild(childIndex);
        filter(child, inst);
    }

    public void printPhantomBranches(ArrayDeque<PhantomBranch> phantomBranches) {
        System.out.println("Phantom Branches:");
        for (PhantomBranch pb : phantomBranches) {
            System.out.println(pb.usedNominalAttributes);
        }
    }

    public void printTree() {
        if (super.treeRoot == null) {
            System.exit(1);
        }

        System.out.println("Custom print tree:");

        ArrayDeque<Node> nodes = new ArrayDeque<>();
        nodes.offer(super.treeRoot);
        StringBuilder sb = new StringBuilder();

        while (!nodes.isEmpty()) {
            int size = nodes.size();
            for (int i = 0; i < size; i++) {
                Node curNode = nodes.pop();

                if (curNode instanceof LearningNode) {
                    double[] observations = curNode.getObservedClassDistribution();
                    int labelIndex = -1;
                    double maxObservation = -1;
                    for (int j = 0; j < observations.length; j++) {
                        if (maxObservation < observations[j]) {
                            labelIndex = j;
                            maxObservation = observations[j];
                        }
                    }
                    sb.append("[label]" + labelIndex + " ");

                } else {
                    SplitNode splitNode = (SplitNode) curNode;
                    InstanceConditionalTest condition = splitNode.getSplitTest();

                    int attIndex = condition.getAttributeIndex();
                    double attValue = condition.getAttributeValue();
                    sb.append("[" + attIndex + "]" + attValue + " ");

                    for (int j = 0; j < splitNode.numChildren(); j++) {
                        nodes.offer((Node) splitNode.getChild(j));
                    }
                }
            }
            sb.append("\n");
        }
        System.out.println(sb.toString());

        System.out.println("MOA description:");
        StringBuilder description = new StringBuilder();
        this.treeRoot.describeSubtree(this, description, 4);
        System.out.println(description.toString());
    }

    public boolean isRandomizable() {
        return false;
    }

    @Override
    public void resetLearningImpl() {
        // Force majority count and binary splits
        this.leafpredictionOption.setChosenLabel("MC");
        super.resetLearningImpl();
        this.binarySplitsOption.setValue(true);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (this.trainingWeightSeenByModel < obsPeriodOption.getValue()) {
            super.trainOnInstanceImpl(inst);
            instanceStore.offer(inst);

        } else if (this.trainingWeightSeenByModel == obsPeriodOption.getValue()) {
            ArrayDeque<PhantomBranch> phantomBranches = growPhantomBranch();
            printPhantomBranches(phantomBranches);

            printTree();
        }
    }

    // public void getModelDescription(StringBuilder out, int indent) {
    // }

    // protected moa.core.Measurement[] getModelMeasurementsImpl() {
    //     return null;
    // }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == PhantomTree.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }

}
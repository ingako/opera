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
import moa.options.ClassOption;
import org.jfree.util.ArrayUtils;

import java.util.*;

public class PhantomTree extends HoeffdingTree implements MultiClassClassifier, CapabilitiesHandler {

    private static final long serialVersionUID = 1L;

    public IntOption obsPeriodOption = new IntOption("obsPeriod", 'o',
            "The number of instances to observe before growing phantom branches.",
            1000, 0, Integer.MAX_VALUE);

    public IntOption numPhantomBranchOption = new IntOption("numPhantomBranch", 'k',
            "The number of phantom branches to grow.",
            9, 1, Integer.MAX_VALUE);

    public ClassOption phantomSplitCriterionOption = new ClassOption("splitCriterion",
            'y', "Split criterion to use.", SplitCriterion.class,
            "FoilInfoGainSplitCriterion");


    ArrayDeque<Instance> instanceStore = new ArrayDeque<>();

    public static class PhantomNode extends ActiveLearningNode {

        public ArrayDeque<Instance> reachedInstances;
        int depth;
        double foil_info_gain; // TODO from parent

        public PhantomNode(double[] initialClassObservations, int depth) {
            super(initialClassObservations);
            this.weightSeenAtLastSplitEvaluation = getWeightSeen();
            this.isInitialized = false;
            this.depth = depth;
        }

        public AttributeSplitSuggestion[] getAllSplitSuggestions(SplitCriterion criterion) {
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

    private ArrayDeque<PhantomNode> growPhantomBranches() {
        // init candidate phantom branches
        addInstancesToLeaves(this.instanceStore);
        ArrayDeque<PhantomNode> phantomRoots = initPhantomBranches();
        ArrayDeque<PhantomNode> candidatePhantomRoots = new ArrayDeque<>();
        // TODO perform first level phantom splits to find candidates.

        ArrayDeque<PhantomNode> phantomLeaves = new ArrayDeque<>();
        for (int i = 0; i < numPhantomBranchOption.getValue(); i++) {
            // TODO select candidate phantom roots by weighted selection.
            // PhantomNode phantomRoot = getWeightedRandomChild(candidatePhantomRoots);
            // phantomLeaves.offer(growPhantomBranches(phantomRoot));
        }

        return phantomLeaves;
    }

    private PhantomNode phantomSplitFirstLevel() {
        // TODO
        return null;
    }

    private PhantomNode phantomSplit(PhantomNode node) {

        // TODO skip split if phantom children already exits

        if (node.observedClassDistributionIsPure()) {
            return node;
        }

        SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.phantomSplitCriterionOption);
        AttributeSplitSuggestion[] allSplitSuggestions = node.getAllSplitSuggestions(splitCriterion);
        Arrays.sort(allSplitSuggestions);

        // TODO split by foil information gain?
        AttributeSplitSuggestion splitDecision = getWeightedRandomPhantomChild(allSplitSuggestions);
        if (splitDecision.splitTest == null) {
            System.exit(1);
        }

        SplitNode newSplit = newSplitNode(
                splitDecision.splitTest,
                node.getObservedClassDistribution(),
                splitDecision.numSplits());

        // TODO for each candidate child, if class distribution > 30% then create child
        for (int i = 0; i < splitDecision.numSplits(); i++) {
            double[] resultingClassDistribution = splitDecision.resultingClassDistributionFromSplit(i);
            boolean shouldStop = true;
            for (double classDistribution : resultingClassDistribution) {
                if (classDistribution >= 0.3) {
                    shouldStop = false;
                    break;
                }
            }
            if (shouldStop) continue;

            PhantomNode newChild = new PhantomNode(resultingClassDistribution, node.depth + 1);
            newSplit.setChild(i, newChild);

            // TODO train the phantom children
            for (Instance inst : node.reachedInstances) {
                newChild.learnFromInstance(inst, this);
            }
        }

        // TODO pick a child with the highest foil information gain (by weighted selection)
        PhantomNode selectedPhantomChild = null;
        for (Node phantomChild : newSplit.getChildren()) {
            // TODO compute foil information gain before weighted selection

        }


        return phantomSplit(selectedPhantomChild);
    }

    private AttributeSplitSuggestion getWeightedRandomPhantomChild(AttributeSplitSuggestion[] allSplitSuggestions) {
        // merit: foil information gain

        double sum = 0;
        double invalid_suggestion_count = 0;
        for (AttributeSplitSuggestion suggestion : allSplitSuggestions) {
            if (suggestion.merit == Double.NEGATIVE_INFINITY) {
                continue;
            }
            if (suggestion.merit <= 0) {
                continue;
            }
            sum += suggestion.merit;
        }

        if (invalid_suggestion_count == allSplitSuggestions.length) {
            return null;
        }

        double rand = Math.random();
        double partial_sum = 0;
        for (AttributeSplitSuggestion suggestion : allSplitSuggestions) {
            if (suggestion.merit <= 0) {
                continue;
            }
            partial_sum += (suggestion.merit / sum);
            if (partial_sum > rand) {
                return suggestion;
            }
        }

        return null;
    }

    private ArrayDeque<PhantomNode> initPhantomBranches() {
        ArrayDeque<Node> nodes = new ArrayDeque<>();
        ArrayDeque<HashSet<Integer>> usedNominalAttributes = new ArrayDeque<>();
        ArrayDeque<PhantomNode> phantomRoots = new ArrayDeque<>();
        nodes.offer(super.treeRoot);
        usedNominalAttributes.offer(new HashSet<>());
        int depth = 0;

        // TODO remove redundancy
        while (!nodes.isEmpty()) {
            Node curNode = nodes.pop();
            depth += 1;

            if (curNode instanceof LearningNode) {
                phantomRoots.offer(
                        new PhantomNode(
                                curNode.getObservedClassDistribution(),
                                depth));

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

        return phantomRoots;
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

    public void printPhantomBranches(ArrayDeque<PhantomNode> phantomLeaves) {
        System.out.println("Phantom Nodes:");
        for (PhantomNode pn: phantomLeaves) {
            // TODO
            // System.out.println();
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
            ArrayDeque<PhantomNode> phantomNodes = growPhantomBranches();
            printPhantomBranches(phantomNodes);

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
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
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.classifiers.patching.Patching;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.options.ClassOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.recommender.rc.utils.Hash;

import java.util.ArrayDeque;
import java.util.HashSet;

public class PhantomTree extends HoeffdingTree implements MultiClassClassifier, CapabilitiesHandler {

    private static final long serialVersionUID = 1L;

    public IntOption obsPeriodOption = new IntOption("obsPeriod", 'o',
            "The number of instances to observe before growing phantom branches.",
            10000, 0, Integer.MAX_VALUE);

    public IntOption numPhantomBranchOption = new IntOption("numPhantomBranch", 'p',
            "The number of phantom branches to grow.",
            9, 1, Integer.MAX_VALUE);

    public IntOption gracePeriodOption = new IntOption("gracePeriod", 'g',
            "The number of instances to observe between model changes.",
            1000, 0, Integer.MAX_VALUE);

    public FlagOption binarySplitsOption = new FlagOption("binarySplits", 'b',
            "Only allow binary splits.");

    public ClassOption splitCriterionOption = new ClassOption("splitCriterion",
            'c', "Split criterion to use.", SplitCriterion.class,
            "InfoGainSplitCriterion");

    protected AttributeSplitSuggestion bestSplit;

    protected DoubleVector observedClassDistribution;

    protected AutoExpandVector<AttributeClassObserver> attributeObservers;

    //
    ArrayDeque<Instance> instanceStore;

    class PhantomBranch {
        SplitNode leaf;
        HashSet<String> usedFeatureValuePairs;
        ArrayDeque<Instance> reachedInstances;

        public PhantomBranch(SplitNode leaf, HashSet<String> branchDecisions) {
            this.leaf = leaf;
            this.usedFeatureValuePairs = branchDecisions;
            reachedInstances = new ArrayDeque<>();
        }
    }

    public void growPhantomBranch() {
        addInstancesToLeaves(this.instanceStore);
        ArrayDeque<SplitNode> nodes = new ArrayDeque<>();
        ArrayDeque<HashSet<String>> branchDecisions = new ArrayDeque<>();
        ArrayDeque<PhantomBranch> phantomBranches = new ArrayDeque<>();
        nodes.push((SplitNode) this.treeRoot);
        branchDecisions.push(new HashSet<>());

        while (!nodes.isEmpty()) {
            SplitNode curNode = nodes.pop();

            if (curNode.isLeaf()) {
                phantomBranches.push(new PhantomBranch(curNode, branchDecisions.pop()));

            } else {
                HashSet<String> branchDecision = branchDecisions.pop();
                branchDecision.add(curNode.getSplitTest().getAttIndex() + "#" + curNode.getSplitTest().getAttValue());

                for (int j = 0; j < curNode.numChildren(); j++) {
                    nodes.push((SplitNode) curNode.getChild(j));
                    branchDecisions.push((HashSet<String>) branchDecision.clone());
                }
            }
        }

        ArrayDeque<PhantomBranch> topBranches = new ArrayDeque<>();
        for (int i = 0; i < numPhantomBranchOption.getValue(); i++) {
            topBranches.push(phantomSplit(phantomBranches));
        }
    }

    private void addInstancesToLeaves(ArrayDeque<Instance> instanceStore) {
        for (Instance inst : instanceStore) {
            filter((SplitNode) this.treeRoot, inst);
        }
    }

    private void filter(SplitNode node, Instance inst) {
        int childIndex = node.instanceChildIndex(inst);
        if (childIndex >= 0) {
            Node child = node.getChild(childIndex);
            if (child == null) {
                node.instanceStore.push(inst);
            } else {
                filter(node, inst);
            }
        }
    }

    private PhantomBranch phantomSplit(ArrayDeque<PhantomBranch> phantomBranches) {
        PhantomBranch bestPhantomBranch = null;
        for (PhantomBranch phantomBranch : phantomBranches) {

        }
        return bestPhantomBranch;
    }

    public String printTree() {
        if (this.treeRoot == null) {
            System.exit(1);
        }

        ArrayDeque<SplitNode> nodes = new ArrayDeque<>();
        nodes.push((SplitNode) this.treeRoot);
        StringBuilder sb = new StringBuilder();

        while (!nodes.isEmpty()) {
            int size = nodes.size();
            for (int i = 0; i < size; i++) {
                SplitNode curNode = nodes.pop();

                if (curNode.isLeaf()) {
                    int labelIndex = this.observedClassDistribution.maxIndex();
                    sb.append("[label]" + labelIndex + " ");

                } else {
                    int attIndex = curNode.getSplitTest().getAttIndex();
                    int attValue = curNode.getSplitTest().getAttValue();
                    sb.append("[" + attIndex  + "]" + attValue + " ");

                    for (int j = 0; j < curNode.numChildren(); j++) {
                        nodes.push((SplitNode) curNode.getChild(j));
                    }
                }
                sb.append("\n");
            }
        }
        return sb.toString();
    }

    public boolean isRandomizable() {
        return false;
    }

    @Override
    public void resetLearningImpl() {
        this.bestSplit = null;
        this.observedClassDistribution = new DoubleVector();
        this.attributeObservers = new AutoExpandVector<AttributeClassObserver>();
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        super.trainOnInstanceImpl(inst);
        instanceStore.push(inst);

        if (this.trainingWeightSeenByModel > obsPeriodOption.getValue()) {
            growPhantomBranch();
        }
    }

    public double[] getVotesForInstance(Instance inst) {
        if (this.bestSplit != null) {
            int branch = this.bestSplit.splitTest.branchForInstance(inst);
            if (branch >= 0) {
                return this.bestSplit
                        .resultingClassDistributionFromSplit(branch);
            }
        }
        return this.observedClassDistribution.getArrayCopy();
    }

    protected AttributeClassObserver newNominalClassObserver() {
        return new NominalAttributeClassObserver();
    }

    protected AttributeClassObserver newNumericClassObserver() {
        return new GaussianNumericAttributeClassObserver();
    }

    protected AttributeSplitSuggestion findBestSplit(SplitCriterion criterion) {
        AttributeSplitSuggestion bestFound = null;
        double bestMerit = Double.NEGATIVE_INFINITY;
        double[] preSplitDist = this.observedClassDistribution.getArrayCopy();
        for (int i = 0; i < this.attributeObservers.size(); i++) {
            AttributeClassObserver obs = this.attributeObservers.get(i);
            if (obs != null) {
                AttributeSplitSuggestion suggestion =
                        obs.getBestEvaluatedSplitSuggestion(
                                criterion,
                                preSplitDist,
                                i,
                                this.binarySplitsOption.isSet());
                if (suggestion.merit > bestMerit) {
                    bestMerit = suggestion.merit;
                    bestFound = suggestion;
                }
            }
        }
        return bestFound;
    }

    public void getModelDescription(StringBuilder out, int indent) {
    }

    protected moa.core.Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == PhantomTree.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }

}
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
import moa.classifiers.core.conditionaltests.NumericAttributeBinaryTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.classifiers.patching.Patching;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import com.yahoo.labs.samoa.instances.Instance;
import moa.core.SizeOf;
import moa.core.Utils;
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

    ArrayDeque<Instance> instanceStore = new ArrayDeque<>();

    public static class PhantomNode extends ActiveLearningNode {

        int depth;
        double num_positive;
        double num_negative;
        double foil_info_gain;
        // public boolean isPhantomLeaf;

        public AutoExpandVector<PhantomNode> children;
        public InstanceConditionalTest splitTest;
        public StringBuilder branchStringBuilder;

        public PhantomNode(double[] initialClassObservations,
                           int depth,
                           StringBuilder branchStringBuilder) {
            super(initialClassObservations);
            // this.weightSeenAtLastSplitEvaluation = getWeightSeen();
            // this.isInitialized = false;
            this.depth = depth;
            this.num_positive = 0;
            this.num_negative = 0;
            this.foil_info_gain = -1;
            // this.isPhantomLeaf = false;

            this.children = new AutoExpandVector<>();
            this.splitTest = null;
            this.branchStringBuilder = branchStringBuilder;
        }

        private void passInstanceToChild(Instance inst, HoeffdingTree ht) {
            int childIndex = this.splitTest.branchForInstance(inst);
            if (childIndex < 0) {
                System.exit(1);
            }
            PhantomNode child = this.children.get(childIndex);
            child.instanceStore.offer(inst);
            child.learnFromInstance(inst, ht);
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
        ArrayDeque<PhantomNode> phantomRootParents = initPhantomBranches();
        if (phantomRootParents.size() == 0) {
            System.out.print("No phantom roots constructed.");
            System.exit(1);
        }

        // perform first level phantom splits to find phantom roots
        AutoExpandVector<PhantomNode> phantomRoots = new AutoExpandVector<>();
        for (PhantomNode parent : phantomRootParents) {
            phantomSplit(parent);

            for (PhantomNode child : parent.children) {
                calcFoilInfoGain(parent, child);
                phantomRoots.addAll(parent.children);
            }
        }

        ArrayDeque<PhantomNode> phantomLeaves = new ArrayDeque<>();
        for (int i = 0; i < numPhantomBranchOption.getValue(); i++) {
            PhantomNode phantomRoot = getWeightedRandomPhantomChild(phantomRoots);
            phantomLeaves.offer(growPhantomBranch(phantomRoot));
        }

        return phantomLeaves;
    }

    private PhantomNode growPhantomBranch(PhantomNode node) {
        if (node.observedClassDistributionIsPure()) {
            return node;
        }

        // split if phantom children do not exist
        if (node.children.size() == 0) {
            phantomSplit(node);
        }

        // compute foil information gain before weighted selection
        for (Node child : node.children) {
            PhantomNode phantomChild = (PhantomNode) child;
            phantomChild.foil_info_gain = calcFoilInfoGain(node, phantomChild);
        }

        PhantomNode selectedPhantomChild = getWeightedRandomPhantomChild(node.children);

        return growPhantomBranch(selectedPhantomChild);
    }

    private void phantomSplit(PhantomNode node) {
        SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
        AttributeSplitSuggestion[] allSplitSuggestions = node.getAllSplitSuggestions(splitCriterion);
        if (allSplitSuggestions.length == 0) {
            return;
        }
        Arrays.sort(allSplitSuggestions);

        // TODO make best split suggestions by foil information gain?
        // AttributeSplitSuggestion splitDecision = getWeightedRandomPhantomChild(allSplitSuggestions);
        AttributeSplitSuggestion splitDecision = allSplitSuggestions[allSplitSuggestions.length - 1];
        if (splitDecision.splitTest == null) {
            System.exit(1);
        }
        node.splitTest = splitDecision.splitTest;

        for (int i = 0; i < splitDecision.numSplits(); i++) {
            double[] resultingClassDistribution = splitDecision.resultingClassDistributionFromSplit(i);

            // boolean shouldStop = true;
            // TODO for each candidate child, if class distribution > 30% then create child
            // for (double classDistribution : resultingClassDistribution) {
            //     if (classDistribution >= 0.3) {
            //         shouldStop = false;
            //         break;
            //     }
            // }
            // if (shouldStop) continue;

            PhantomNode newChild = new PhantomNode(
                    resultingClassDistribution,
                    node.depth + 1,
                    new StringBuilder(node.branchStringBuilder));

            InstanceConditionalTest condition = node.splitTest;
            if (condition instanceof NominalAttributeBinaryTest) {
                newChild.branchStringBuilder.append(condition.getAttributeIndex());
            } else if (condition instanceof NumericAttributeBinaryTest) {
                newChild.branchStringBuilder.append(condition.getAttributeValue());
            } else {
                System.out.print("Multiway test is not supported.");
                System.exit(1);
            }
            node.children.add(newChild);

            // TODO only train the selected phantom children?
            for (Instance inst : node.instanceStore) {
                node.passInstanceToChild(inst, this);
            }
        }
    }

    private double calcFoilInfoGain(PhantomNode parent, PhantomNode child) {
        return getNumMutualPositives(parent, child)
                * (child.num_positive / (child.num_positive + child.num_negative)
                - parent.num_positive / (parent.num_positive + parent.num_negative));
    }

   private double getNumMutualPositives(PhantomNode parent, PhantomNode child) {
        int count = 0;
        for (Instance inst : child.instanceStore) {
            int trueClass = (int) inst.classValue();
            int parentPrediction = Utils.maxIndex(parent.getClassVotes(inst, this));
            int childPrediction = Utils.maxIndex(child.getClassVotes(inst, this));
            if  (parentPrediction == trueClass && parentPrediction == childPrediction) {
                count++;
            }
        }

        return count;
   }

    private PhantomNode getWeightedRandomPhantomChild(AutoExpandVector<PhantomNode> phantomChildren) {
        double sum = 0;
        int invalid_child_count = 0;
        for (PhantomNode child : phantomChildren) {
            if (child.foil_info_gain <= 0) {
                invalid_child_count++;
                continue;
            }
            sum += child.foil_info_gain;
        }

        if (invalid_child_count == phantomChildren.size()) {
            return null;
        }

        double rand = Math.random();
        double partial_sum = 0;
        for (PhantomNode child : phantomChildren) {
            if (child.foil_info_gain <= 0) {
                continue;
            }
            partial_sum += (child.foil_info_gain / sum);
            if (partial_sum > rand) {
                return child;
            }
        }

        return null;
    }

    private ArrayDeque<PhantomNode> initPhantomBranches() {
        ArrayDeque<Node> nodes = new ArrayDeque<>();
        ArrayDeque<PhantomNode> phantomRoots = new ArrayDeque<>();
        ArrayDeque<Integer> depths = new ArrayDeque<>();
        ArrayDeque<StringBuilder> branchStringBuilders = new ArrayDeque<>();
        nodes.offer(super.treeRoot);
        depths.offer(1);
        branchStringBuilders.offer(new StringBuilder());

        while (!nodes.isEmpty()) {
            Node curNode = nodes.poll();
            StringBuilder branchStringBuilder = branchStringBuilders.poll();
            int depth = depths.poll();

            if (curNode instanceof LearningNode) {
                branchStringBuilder.append(curNode.getClass());
                phantomRoots.offer(
                        new PhantomNode(
                                curNode.getObservedClassDistribution(),
                                depth + 1,
                                branchStringBuilder));

            } else {
                branchStringBuilder.append("|");

                SplitNode splitNode = (SplitNode) curNode;
                InstanceConditionalTest condition = splitNode.getSplitTest();
                if (condition instanceof NominalAttributeBinaryTest) {
                    branchStringBuilder.append(condition.getAttributeIndex());
                } else if (condition instanceof NumericAttributeBinaryTest) {
                    branchStringBuilder.append(condition.getAttributeValue());
                } else {
                    System.out.print("Multiway test is not supported.");
                    System.exit(1);
                }

                for (int j = 0; j < splitNode.numChildren(); j++) {
                    nodes.offer(splitNode.getChild(j));
                    branchStringBuilders.offer(new StringBuilder(branchStringBuilder));
                    depths.offer(depth + 1);
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
        for (PhantomNode pl: phantomLeaves) {
            System.out.println("Depth= " + pl.depth);
            System.out.println(pl.branchStringBuilder.toString());
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
                Node curNode = nodes.poll();

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
                        nodes.offer(splitNode.getChild(j));
                    }
                }
            }
            sb.append("\n");
        }
        System.out.println(sb.toString());
    }

    public boolean isRandomizable() {
        return false;
    }

    @Override
    public void resetLearningImpl() {
        super.resetLearningImpl();
        this.leafpredictionOption.setChosenLabel("MC");
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

            // printTree();
            System.out.println("MOA description:");
            StringBuilder description = new StringBuilder();
            this.treeRoot.describeSubtree(this, description, 4);
            System.out.println(description.toString());
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
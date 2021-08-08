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

    // public ClassOption splitCriterionOption = new ClassOption("splitCriterion",
    //         'd', "Split criterion to use.", SplitCriterion.class,
    //         "PhantomSplitCriterion");

    ArrayDeque<Instance> instanceStore = new ArrayDeque<>();

    public class PhantomNode extends ActiveLearningNode {

        int depth;
        double foil_info_gain;
        // public boolean isPhantomLeaf;
        String branchPrefix = "";

        public AutoExpandVector<InstanceConditionalTest> splitTests;
        public AutoExpandVector<PhantomNode> splitChildrenPairs;
        public ArrayDeque<Instance> instanceStore;

        public PhantomNode(
                double[] initialClassObservations,
                int depth,
                String branchPrefix,
                ArrayDeque<Instance> instanceStore) {

            this(initialClassObservations, depth);
            this.instanceStore = instanceStore;
            this.branchPrefix = branchPrefix;
        }

        public PhantomNode(
                double[] initialClassObservations,
                int depth) {

            super(initialClassObservations);
            this.depth = depth;
            this.foil_info_gain = -1;

            this.splitTests = new AutoExpandVector<>();
            this.splitChildrenPairs = new AutoExpandVector<>();
            this.instanceStore = new ArrayDeque<>();
        }

        private void passInstanceToChild(
                Instance inst,
                HoeffdingTree ht,
                InstanceConditionalTest splitTest,
                AutoExpandVector<PhantomNode> children) {
            // int childIndex = this.splitTest.branchForInstance(inst);
            int childIndex = splitTest.branchForInstance(inst);
            if (childIndex < 0) {
                System.exit(1);
            }
            PhantomNode child = children.get(childIndex);
            if (child == null) {
                throw new NullPointerException("passInstanceTodChild child is null");
            }
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

        @Override
        public String toString() {
            return "";
        }
    }

    private ArrayDeque<PhantomNode> growPhantomBranches() {
        // init candidate phantom branches
        addInstancesToLeaves(this.instanceStore);
        ArrayDeque<PhantomNode> phantomRootParents = initPhantomBranches();
        if (phantomRootParents.size() == 0) {
            throw new NullPointerException("No phantom root parent constructed.");
        }

        // perform first level phantom splits to find phantom roots
        AutoExpandVector<PhantomNode> phantomRoots = new AutoExpandVector<>();
        for (PhantomNode parent : phantomRootParents) {
            phantomSplit(parent);
            phantomRoots.addAll(parent.splitChildrenPairs);
        }

        ArrayDeque<PhantomNode> phantomLeaves = new ArrayDeque<>();
        for (int i = 0; i < numPhantomBranchOption.getValue(); i++) {
            int nodeIdx = getWeightedRandomPhantomNodeIdx(phantomRoots);
            if (nodeIdx == -1) {
                throw new NullPointerException("getWeightedRandomPhantomNodeIdx returns -1");
            }
            PhantomNode phantomRoot = phantomRoots.get(nodeIdx);

            StringBuilder branchStringBuilder = new StringBuilder(phantomRoot.branchPrefix);
            phantomLeaves.offer(growPhantomBranch(phantomRoot, branchStringBuilder));
            System.out.println(branchStringBuilder.toString());
        }

        return phantomLeaves;
    }

    private PhantomNode growPhantomBranch(PhantomNode node, StringBuilder branchStringBuilder) {
        if  (node == null) {
            throw new NullPointerException("growPhantomBranch node is null");
        }

        if (node.observedClassDistributionIsPure()) {
            return node;
        }

        // TODO cache leaf node info
        // split if phantom children do not exist
        if (node.splitChildrenPairs.size() == 0) {
            phantomSplit(node);
        }

        int childIdx = getWeightedRandomPhantomNodeIdx(node.splitChildrenPairs);
        if (childIdx == -1) {
            throw new NullPointerException("getWeightedRandomPhantomChildIdx returns -1");
        }
        PhantomNode selectedPhantomChild = node.splitChildrenPairs.get(childIdx);
        InstanceConditionalTest condition = node.splitTests.get(childIdx);
        if (condition instanceof NominalAttributeBinaryTest) {
            branchStringBuilder.append(condition.getAttributeIndex());
        } else if (condition instanceof NumericAttributeBinaryTest) {
            branchStringBuilder.append(condition.getAttributeValue());
        } else {
            throw new NullPointerException("Multiway test is not supported.");
        }

        return growPhantomBranch(selectedPhantomChild, branchStringBuilder);
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
        for (AttributeSplitSuggestion splitDecision : allSplitSuggestions) {
            // AttributeSplitSuggestion splitDecision = allSplitSuggestions[allSplitSuggestions.length - 1];
            if (splitDecision.splitTest == null) {
                throw new NullPointerException("PhantomSplit splitTest is null.");
            }
            // node.splitTest = splitDecision.splitTest;

            InstanceConditionalTest curSplitTest = splitDecision.splitTest;
            AutoExpandVector<PhantomNode> newChildren = new AutoExpandVector<>();
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
                        node.depth + 1);

                newChildren.add(newChild);
            }

            // TODO only train the selected phantom children?
            // for each splitTest, pass instances & train
            for (Instance inst : node.instanceStore) {
                node.passInstanceToChild(inst, this, curSplitTest, newChildren);
            }

            // compute foil information gain for weighted selection
            for (Node child : newChildren) {
                PhantomNode phantomChild = (PhantomNode) child;
                phantomChild.foil_info_gain = calcFoilInfoGain(node, phantomChild);
            }
            node.splitTests.add(curSplitTest);
            node.splitChildrenPairs.addAll(newChildren);
        }
    }

    private double calcFoilInfoGain(PhantomNode parent, PhantomNode child) {
        double child_num_positive = 0;
        double parent_num_positive = 0;
        double count = 0;
        for (Instance inst : child.instanceStore) {
            int trueClass = (int) inst.classValue();
            int parentPrediction = Utils.maxIndex(parent.getClassVotes(inst, this));
            int childPrediction = Utils.maxIndex(child.getClassVotes(inst, this));
            if  (parentPrediction == trueClass) {
                parent_num_positive++;
            }
            if  (childPrediction == trueClass) {
                child_num_positive++;
            }
            if (parentPrediction == childPrediction) {
                count++;
            }
        }

        double total = child.instanceStore.size();
        double gain = total
                * (Math.log(child_num_positive / total) / Math.log(2)
                   - Math.log(parent_num_positive / total) / Math.log(2));
        return gain;
    }

    private int getWeightedRandomPhantomNodeIdx(AutoExpandVector<PhantomNode> phantomChildren) {
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
            return -1;
        }

        double rand = Math.random();
        double partial_sum = 0;
        for (int i = 0; i < phantomChildren.size(); i++) {
            PhantomNode child = phantomChildren.get(i);
            if (child.foil_info_gain <= 0) {
                continue;
            }
            partial_sum += (child.foil_info_gain / sum);
            if (partial_sum > rand) {
                return i;
            }
        }

        return -1;
    }

    private ArrayDeque<PhantomNode> initPhantomBranches() {
        ArrayDeque<Node> nodes = new ArrayDeque<>();
        ArrayDeque<PhantomNode> phantomRoots = new ArrayDeque<>();
        ArrayDeque<Integer> depths = new ArrayDeque<>();
        ArrayDeque<StringBuilder> branchStringBuilders = new ArrayDeque<>();
        nodes.offer(super.treeRoot);
        depths.offer(0);
        branchStringBuilders.offer(new StringBuilder("#"));

        while (!nodes.isEmpty()) {
            Node curNode = nodes.poll();
            StringBuilder branchStringBuilder = branchStringBuilders.poll();
            int depth = depths.poll();

            if (curNode instanceof LearningNode) {
                branchStringBuilder.append("#");
                PhantomNode root = new PhantomNode(
                        curNode.getObservedClassDistribution(),
                        depth + 1,
                        branchStringBuilder.toString(),
                        curNode.instanceStore);
                root.attributeObservers = ((ActiveLearningNode) curNode).attributeObservers;
                phantomRoots.offer(root);

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
            System.out.println(pl.branchPrefix);
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
            // printPhantomBranches(phantomNodes);

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
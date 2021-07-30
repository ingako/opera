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
import moa.classifiers.core.conditionaltests.InstanceConditionalBinaryTest;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.conditionaltests.NominalAttributeBinaryTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.classifiers.patching.Patching;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import com.yahoo.labs.samoa.instances.Instance;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

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
        Node leaf;
        HashSet<Integer> usedNominalAttributes;
        ArrayDeque<Instance> reachedInstances;

        public PhantomBranch(Node leaf, HashSet<Integer> usedNominalAttributes) {
            this.leaf = leaf;
            this.usedNominalAttributes = usedNominalAttributes;
            reachedInstances = new ArrayDeque<>();
        }
    }

    public ArrayDeque<PhantomBranch> growPhantomBranch() {
        addInstancesToLeaves(this.instanceStore);
        ArrayDeque<PhantomBranch> phantomBranches = initPhantomBranches();

        // ArrayDeque<PhantomBranch> topBranches = new ArrayDeque<>();
        // for (int i = 0; i < numPhantomBranchOption.getValue(); i++) {
        //     topBranches.push(phantomSplit(phantomBranches));
        // }

        return phantomBranches;
    }

    private PhantomBranch phantomSplit(ArrayDeque<PhantomBranch> phantomBranches) {
        PhantomBranch bestPhantomBranch = null;
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

        return bestPhantomBranch;
    }

    private ArrayDeque<PhantomBranch> initPhantomBranches() {
        ArrayDeque<Node> nodes = new ArrayDeque<>();
        ArrayDeque<HashSet<Integer>> usedNominalAttributes = new ArrayDeque<>();
        ArrayDeque<PhantomBranch> phantomBranches = new ArrayDeque<>();
        nodes.push(super.treeRoot);
        usedNominalAttributes.push(new HashSet<>());

        while (!nodes.isEmpty()) {
            Node curNode = nodes.pop();

            if (curNode instanceof LearningNode) {
                phantomBranches.push(new PhantomBranch(curNode, usedNominalAttributes.pop()));

            } else {
                HashSet<Integer> usedNominalAttributeSet = usedNominalAttributes.pop();

                SplitNode splitNode = (SplitNode) curNode;
                InstanceConditionalTest condition = splitNode.getSplitTest();
                if (condition instanceof NominalAttributeBinaryTest) {
                    usedNominalAttributeSet.add(condition.getAttributeIndex());
                }

                for (int j = 0; j < splitNode.numChildren(); j++) {
                    nodes.push(splitNode.getChild(j));
                    usedNominalAttributes.push((HashSet<Integer>) usedNominalAttributeSet.clone());
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
            node.instanceStore.push(inst);
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

        System.out.println("Print Tree");

        ArrayDeque<Node> nodes = new ArrayDeque<>();
        nodes.push(super.treeRoot);
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
                        nodes.push((Node) splitNode.getChild(j));
                    }
                }
            }
            sb.append("\n");
        }
        System.out.println(sb.toString());

        StringBuilder description = new StringBuilder();
        this.treeRoot.describeSubtree(this, description, 4);
        System.out.println("description:" + description.toString());
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
            instanceStore.push(inst);

        } else if (this.trainingWeightSeenByModel == obsPeriodOption.getValue()) {
            ArrayDeque<PhantomBranch> phantomBranches = growPhantomBranch();
            printPhantomBranches(phantomBranches);

            printTree();

        } else {
            return;
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
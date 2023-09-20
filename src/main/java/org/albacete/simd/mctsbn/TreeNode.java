package org.albacete.simd.mctsbn;

import java.util.Arrays;
import org.albacete.simd.utils.Problem;
import org.jetbrains.annotations.NotNull;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Objects;
import java.util.Set;

public class TreeNode implements Comparable<TreeNode>{

    private final State state;
    private final TreeNode parent;
    private Set<TreeNode> children = new HashSet<>();
    
    private int numVisits = 0;
    private double totalReward = 0;

    private double UCTSCore = 0;
    
    private final HillClimbingEvaluator hc;
    
    private boolean fullyExpanded;
    private boolean isExpanded = false;

    MCTSBN mctsbn;

    public TreeNode(State state, TreeNode parent, MCTSBN mctsbn) {
        this.state = state;
        this.hc = state.getHC();
        this.parent = parent;
        this.numVisits = 0;
        this.totalReward = 0;
        this.fullyExpanded = state.isTerminal();
        this.mctsbn = mctsbn;
        
        if (this.parent != null) {
            this.parent.addChild(this);
            this.numVisits = 1;
        }
    }

    public boolean isTerminal() {
        return state.isTerminal();
    }

    public State getState() {
        return state;
    }

    public TreeNode getParent() {
        return parent;
    }

    public int getNumVisits() {
        return numVisits;
    }

    public double getTotalReward() {
        return totalReward;
    }
    
    public double getUCTScore() {
        return UCTSCore;
    }

    public Set<TreeNode> getChildren() {
        return children;
    }

    public Set<Integer> getChildrenAction(){
        Set<Integer> actions = new HashSet<>();
        for (TreeNode child: children) {
            actions.add(child.state.getNode());
        }
        return actions;
    }

    public void setChildren(Set<TreeNode> children) {
        this.children = children;
    }

    public void addChild(TreeNode child){
        this.children.add(child);
        this.isExpanded = true;
    }

    public boolean isFullyExpanded() {
        return fullyExpanded;
    }

    public void setFullyExpanded(boolean fullyExpanded) {
        this.fullyExpanded = fullyExpanded;
    }

    public void setNumVisits(int numVisits) {
        this.numVisits = numVisits;
    }

    public void setTotalReward(double totalReward) {
        this.totalReward = totalReward;
    }

    public void addReward(double reward){
        totalReward += reward;
    }

    public void incrementOneVisit(){
        this.numVisits++;
    }
    
    public void decrementOneVisit(){
        this.numVisits--;
    }

    public boolean isExpanded() {
        return isExpanded;
    }

    public void setExpanded(boolean expanded) {
        isExpanded = expanded;
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder(50);
        print(buffer, "", "");
        return buffer.toString();
    }

    private void print(StringBuilder buffer, String prefix, String childrenPrefix) {
        buffer.append(prefix);
        buffer.append("N" + state.getNode());
        
        String results;
        
        double exploitationScore = mctsbn.EXPLOITATION_CONSTANT * (this.getTotalReward() / this.getNumVisits());

        if(this.parent == null){
            results = "  \t" + this.getNumVisits() + "   BDeu " + exploitationScore;
        } else {
            double explorationScore = mctsbn.EXPLORATION_CONSTANT * Math.sqrt(Math.log(this.parent.getNumVisits()) / this.getNumVisits());
            results = "  \t" + this.getNumVisits() + "   UCT " + UCTSCore + ",   BDeu " + exploitationScore + ",   EXP " + explorationScore;
        }
        buffer.append(results);
        
        buffer.append('\n');
        for (Iterator<TreeNode> it = children.iterator(); it.hasNext();) {
            TreeNode next = it.next();
            if (it.hasNext()) {
                next.print(buffer, childrenPrefix + "├── ", childrenPrefix + "│   ");
            } else {
                next.print(buffer, childrenPrefix + "└── ", childrenPrefix + "    ");
            }
        }
    }
    
    public void updateUCT(double bestAStar) {
        if(this.parent == null && this.fullyExpanded){
            UCTSCore = Double.NEGATIVE_INFINITY;
        } else {
            double exploitationScore = mctsbn.EXPLOITATION_CONSTANT * (this.getTotalReward() / this.getNumVisits());
            double explorationScore = mctsbn.EXPLORATION_CONSTANT * Math.sqrt(Math.log(this.parent.getNumVisits()) / this.getNumVisits());
            /*double aStar = state.getLocalScore();
            for (Integer node : state.getPossibleActions()) {
                aStar += hc.bestBDeuForNode[node];
            }
            //System.out.println("localScore: " + state.getLocalScore() + ", aStar: " + aStar + ", bestAStar: " + bestAStar);
            aStar -= bestAStar;
            aStar /= Problem.nInstances;
            aStar *= MCTSBN.A_STAR_CONSTANT;*/

            UCTSCore = exploitationScore + explorationScore;
            //System.out.println("UCT: " + UCTSCore + ".   \t" + exploitationScore + ".   \t" + explorationScore + ".   \t" + aStar);
        }
    }

    @Override
    public int compareTo(@NotNull TreeNode o) {
        return Double.compare(this.UCTSCore, o.UCTSCore);
    }

    @Override
    public boolean equals(Object obj) {
        if(this == obj)
            return true;
        if(!(obj instanceof TreeNode))
            return false;
        TreeNode other = (TreeNode) obj;

        return this.state.equals(other.state);
    }

    @Override
    public int hashCode() {
        int hash = 3;
        hash = 29 * hash + Objects.hashCode(this.state);
        return hash;
    }

}

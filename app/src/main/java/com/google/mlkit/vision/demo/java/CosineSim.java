package com.google.mlkit.vision.demo.java;

public class CosineSim implements Comparable<CosineSim> {
    public String subject;
    public float similarity;

    public CosineSim(String label, float similarity) {
        this.subject = label;
        this.similarity = similarity;
    }

    @Override
    public int compareTo(CosineSim target) {
        return this.similarity <= target.similarity ? 1 : - 1;
    }

    @Override
    public String toString() {
        return "Subject Name : " + subject + ", Similarity : " + similarity;
    }
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NeuralNetwork;

import Control.MathFx;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Asus
 */
public class ConfusionMatrix {

    private int totalSamples;
    private int[][] matrix = new int[1][1];
    private double accuracy;
    private double precision;
    private double recall;
    private double f1score;
    
    public ConfusionMatrix() {
        this.reset();
    }
    
    public void reset() {
        this.totalSamples = 0;
        this.accuracy = 0.0;
        this.precision = 0.0;
        this.recall = 0.0;
        this.f1score = 0.0;
        this.matrix = new int[1][1];
    }
    
    public void update(int actual, int predicted) {
        while ((actual + 1) >= this.matrix.length || 
                (predicted + 1) >= this.matrix.length) {
            this.matrix = this.increaseMatrixSize(this.matrix);
        }
        
        this.matrix[actual][predicted]++;
        this.totalSamples++;
        
        this.calculateAccuracy();
        this.calculatePrecision();
        this.calculateRecall();
        this.calculateF1score();
    }
    
    public void showMatrix() {
        for (int i = 0; i < this.matrix.length; i++) {
            for (int j = 0; j < this.matrix[i].length; j++) {
                System.out.print(this.matrix[i][j] + " ");
            }
            System.out.println();
        }
    }
    
    public double getAccuracy() {
        return this.accuracy;
    }
    
    private void calculateAccuracy() {
        int truePositives = 0;
        for (int i = 0; i < this.matrix.length; i++) {
            truePositives += this.matrix[i][i];
        }
        this.accuracy = (double)truePositives / (double)this.totalSamples;
    }
    
    public double getRecall() {
        return this.recall;
    }
    
    private void calculateRecall() {
        List<Double> recalls = new ArrayList<>();
        
        for (int i = 0; i < this.matrix.length; i++) {
            int n = 0;
            for (int j = 0; j < this.matrix[i].length; j++) {
                n += this.matrix[j][i];
            }
            
            recalls.add(n == 0 ? 0.0 : (double)this.matrix[i][i] / (double)n);
        }
        if (recalls.size() <= 0) {
            this.recall = 0.0;
        }
        else {
            this.recall = MathFx.sum(new ArrayList<>(recalls)) / 
                (double)recalls.size();
        }
        
    }
    
    public double getPrecision() {
        return this.precision;
    }
    
    private void calculatePrecision() {
        List<Double> precisions = new ArrayList<>();
        
        for (int i = 0; i < this.matrix.length; i++) {
            int n = 0;
            for (int j = 0; j < this.matrix[i].length; j++) {
                n += this.matrix[i][j];
            }
            
            precisions.add(n == 0 ? 0.0 : (double)this.matrix[i][i] / (double)n);
        }
        
        if (precisions.size() <= 0) {
            this.precision = 0.0;
        }
        else {
            this.precision = MathFx.sum(new ArrayList<>(precisions)) / 
                (double)precisions.size();
        }
        
    }
    
    public double getF1score() {
        return this.f1score;
    }
    
    private void calculateF1score() {
        if ((this.precision + this.recall) <= 0) {
            this.f1score = 0.0;
        }
        else {
            this.f1score = 2 * ((this.precision * this.recall) / 
                (this.precision + this.recall));
        }
        
    }
    
    private int[][] increaseMatrixSize(int[][] matrix) {
        int[][] newMatrix = new int[matrix.length + 1][matrix[0].length + 1];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                newMatrix[i][j] = matrix[i][j];
            }
        }
        return newMatrix;
    }
}
